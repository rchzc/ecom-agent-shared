"""RAG 服务：把切分、向量化、检索、重排编排成一条可调用的链路。

链路的完整顺序（面试问"RAG 完整流程"就照这个答，每一步为什么这么做都说得出）：

    文档 → 按域分目录 → 语义切分 → 向量化 → 入库
                                              ↓
    用户问题 → 向量化 → 按域过滤召回 top_k×2 → 混合重排 → 取 top_k → 交给模型

两个设计决定值得单独解释：

1. **先召回 top_k×2 再重排取 top_k。** 只召回 top_k 就没得选了 —— 重排的作用是
   "在候选里重新排序"，候选集本身太小的话，该提上来的片段根本没进来。
   放大一倍是成本和效果的折中：再多就是给重排送噪声。

2. **按域过滤在召回阶段做，不在重排阶段做。** 域是硬边界（选品的问题不该召回
   广告知识），过滤放在重排后面会让"为什么这条没出现"变得没法解释。

这个类不关心向量库是哪种后端 —— 换后端是 `build_backend()` 的事，
这里只依赖它暴露的五个方法。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from ..config import Settings
from ..errors import KnowledgeBaseError
from .backends import RetrievedChunk, VectorBackend, build_backend
from .chunking import Chunk, chunk_corpus, load_documents
from .embedder import Embedder
from .rerank import DEFAULT_ALPHA, rerank

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    """入库结果。返回结构化结果而不是只打日志 —— 调用方（CLI / API / 流水线）
    需要拿这些数字做断言和展示。"""

    documents: int
    chunks: int
    domains: dict[str, int]
    backend: str
    embedding_mode: str

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "domains": self.domains,
            "backend": self.backend,
            "embedding_mode": self.embedding_mode,
        }


class RagService:
    """检索增强的对外门面。业务 Agent 只调 `search()`。"""

    def __init__(
        self,
        settings: Settings,
        *,
        backend: VectorBackend | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        self.backend = backend or build_backend(settings)

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    async def ingest_documents(
        self,
        documents: Sequence[tuple[str, str, str]],
        *,
        rebuild: bool = True,
    ) -> IngestReport:
        """把 (domain, source, content) 三元组切分、向量化、入库。"""
        chunks = chunk_corpus(
            list(documents),
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
        )
        if not chunks:
            raise KnowledgeBaseError("没有可入库的内容（文档为空或目录下没有 .md 文件）")

        if rebuild:
            self.backend.reset()

        vectors = await self.embedder.embed([c.text for c in chunks])
        self.backend.add(
            ids=[c.chunk_id for c in chunks],
            texts=[c.text for c in chunks],
            metadatas=[c.to_metadata() for c in chunks],
            embeddings=vectors or None,
        )

        domains: dict[str, int] = {}
        for chunk in chunks:
            domains[chunk.domain] = domains.get(chunk.domain, 0) + 1

        report = IngestReport(
            documents=len(documents),
            chunks=len(chunks),
            domains=domains,
            backend=self.backend.name,
            embedding_mode=self.embedder.mode,
        )
        logger.info(
            "rag.ingest",
            extra={
                "documents": report.documents,
                "chunks": report.chunks,
                "backend": report.backend,
                "embedding_mode": report.embedding_mode,
            },
        )
        return report

    async def ingest_dir(self, docs_dir: str, *, rebuild: bool = True) -> IngestReport:
        """从目录入库。目录约定 ``<docs_dir>/<domain>/*.md``。"""
        documents = load_documents(docs_dir)
        if not documents:
            raise KnowledgeBaseError(f"目录 {docs_dir} 下没有找到任何 .md 文档")
        return await self.ingest_documents(documents, rebuild=rebuild)

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    async def search(
        self,
        query: str,
        *,
        domain: str = "*",
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> list[RetrievedChunk]:
        """检索并重排，返回最终 top_k。"""
        k = top_k or self.settings.top_k
        if k < 1:
            raise KnowledgeBaseError("top_k 必须 >= 1")
        if not query.strip():
            # 空 query 去检索等于让后端随便返回几条，看起来"有结果"其实全是噪声。
            # 明确报错比返回噪声好 —— 这个坑在本地 embedding 降级时踩过。
            raise KnowledgeBaseError("检索 query 不能为空")

        # 召回放大一倍，给重排留出选择空间
        recall_k = max(k * 2, k + 2)

        vectors = await self.embedder.embed([query])
        candidates = self.backend.query(
            vectors[0] if vectors else None,
            domain=domain,
            top_k=recall_k,
            query_text=query,
        )
        if not candidates:
            return []

        ranked = rerank(
            query,
            candidates,
            top_k=k,
            alpha=alpha if alpha is not None else self.settings.rerank_alpha,
        )
        logger.info(
            "rag.search",
            extra={
                "domain": domain,
                "recall": len(candidates),
                "returned": len(ranked),
                "top_score": ranked[0].score if ranked else 0,
            },
        )
        return ranked

    async def search_as_context(
        self, query: str, *, domain: str = "*", top_k: int | None = None
    ) -> list[dict]:
        """检索并转成「可喂给模型的上下文」格式，带来源标记以便前端展示引用。"""
        chunks = await self.search(query, domain=domain, top_k=top_k)
        return [
            {
                "text": c.text,
                "score": c.score,
                "source": c.source,
                "domain": c.domain,
            }
            for c in chunks
        ]

    # ------------------------------------------------------------------
    # 运维
    # ------------------------------------------------------------------
    def count(self) -> int:
        return self.backend.count()

    def reset(self) -> None:
        self.backend.reset()

    def stats(self) -> dict:
        return {
            "backend": self.backend.name,
            "embedding_mode": self.embedder.mode,
            "collection": self.settings.collection,
            "chunks": self.count(),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "top_k": self.settings.top_k,
            "rerank_alpha": self.settings.rerank_alpha,
        }
