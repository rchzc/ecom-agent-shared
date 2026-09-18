"""向量库后端：同一套检索语义，三种实现可切换。

为什么要有后端抽象（面试可以讲这个取舍）：

- **chroma** —— 生产默认。HNSW 近似最近邻索引 + 持久化，规模上去后检索仍是
  亚毫秒级；代价是引入一个不算轻的依赖。
- **numpy** —— 零外部依赖的暴力检索。矩阵乘一次算完所有相似度，几万条以内
  其实比 HNSW 还快（没有图遍历开销），而且完全可解释、可复现，
  离线评测脚本用它最合适。需要调用方自己提供向量。
- **lexical** —— 纯词法检索（BM25），**不需要任何向量**。
  存在的意义：一是单元测试和离线演示需要一个不依赖模型下载的确定性后端，
  二是它同时是"向量检索失效时"的降级路径 —— 真出问题时至少还能关键词检索。

三者对上层暴露完全相同的接口，所以切换后端不会改动任何业务代码。
`build_backend()` 按配置选择，并在不可用时给出明确的降级/报错，不静默换实现。
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..config import Settings
from ..errors import KnowledgeBaseError

logger = logging.getLogger(__name__)

# BM25 参数：k1 控制词频饱和速度，b 控制文档长度归一化强度。这两个是文献里的常用默认值。
_BM25_K1 = 1.5
_BM25_B = 0.75

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class RetrievedChunk:
    """检索结果。score 统一归一化到 0~1，这样重排的加权公式对所有后端都成立。"""

    text: str
    source: str
    domain: str
    score: float


class VectorBackend(Protocol):
    """后端契约。上层的 RagService 只依赖这五个方法。"""

    name: str

    def count(self) -> int: ...

    def reset(self) -> None: ...

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None: ...

    def query(
        self,
        query_embedding: Sequence[float] | None,
        *,
        domain: str,
        top_k: int,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]: ...


# ---------------------------------------------------------------------------
# ChromaDB 后端
# ---------------------------------------------------------------------------
class ChromaBackend:
    """ChromaDB 持久化向量库。"""

    name = "chroma"

    def __init__(self, settings: Settings) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:  # pragma: no cover - 取决于安装环境
            raise KnowledgeBaseError(
                "VECTOR_BACKEND=chroma 但未安装 chromadb。"
                "请 `pip install chromadb`，或改用 VECTOR_BACKEND=numpy / lexical。"
            ) from exc
        self.settings = settings
        self._client = chromadb.PersistentClient(
            path=settings.vector_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._get_collection()

    def _get_collection(self):
        return self._client.get_or_create_collection(
            name=self.settings.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """清空并重建集合（用于知识库重建）。"""
        try:
            self._client.delete_collection(self.settings.collection)
        except Exception:  # 集合不存在时忽略 —— 这不是错误，是正常分支
            pass
        self._collection = self._get_collection()

    def _matched_count(self, where: dict | None) -> int:
        """统计满足 where 条件的切片数（不取回内容，开销很小）。"""
        got = self._collection.get(where=where, include=[])
        return len(got.get("ids") or [])

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """写入切片。embeddings 为空时由 ChromaDB 用内置模型生成。"""
        kwargs: dict[str, Any] = {
            "ids": list(ids),
            "documents": list(texts),
            "metadatas": list(metadatas),
        }
        if embeddings:
            kwargs["embeddings"] = [list(e) for e in embeddings]
        try:
            self._collection.add(**kwargs)
        except Exception as exc:
            raise KnowledgeBaseError(f"写入向量库失败: {exc}") from exc

    def query(
        self,
        query_embedding: Sequence[float] | None,
        *,
        domain: str,
        top_k: int,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        """按域过滤检索。domain="*" 表示不限制域（跨域检索）。

        query_embedding 为 None 时（本地/无 embedding 场景）退回文本检索，
        此时必须用真实 query_text —— 之前这里写死传空串，等于用空文本去检索，
        返回的是库里排在最前面的任意切片，所谓"降级可用"实际是"降级即失效"。
        """
        total = self.count()
        if total == 0:
            raise KnowledgeBaseError(
                "知识库为空，请先调用 ingest() 或 CLI 的 rebuild 命令建立索引"
            )

        where = None if domain in ("*", "", "all") else {"domain": domain}
        kwargs: dict[str, Any] = {
            "n_results": max(1, min(top_k, total)),
            "where": where,
        }
        if query_embedding is not None:
            kwargs["query_embeddings"] = [list(query_embedding)]
        else:
            kwargs["query_texts"] = [query_text or ""]

        try:
            result = self._collection.query(**kwargs)
        except Exception as exc:
            # 部分 ChromaDB 版本在「域内切片数 < n_results」时直接报错。
            # 这时不是检索失败，只是该域知识较少 —— 按域内实际数量重试一次。
            try:
                matched = self._matched_count(where)
            except Exception:  # 连计数都失败，说明是真故障
                raise KnowledgeBaseError(f"检索失败: {exc}") from exc
            if matched <= 0:
                return []
            kwargs["n_results"] = matched
            try:
                result = self._collection.query(**kwargs)
            except Exception as exc2:
                raise KnowledgeBaseError(f"检索失败: {exc2}") from exc2

        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]

        chunks: list[RetrievedChunk] = []
        for doc, meta, dist in zip(documents[0], metadatas[0], distances[0]):
            meta = meta or {}
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    source=meta.get("source", "unknown"),
                    domain=meta.get("domain", "unknown"),
                    # ChromaDB 返回的是余弦距离，转成相似度更直观
                    score=round(1 - float(dist), 4),
                )
            )
        return chunks


# ---------------------------------------------------------------------------
# numpy 暴力检索后端
# ---------------------------------------------------------------------------
class NumpyBackend:
    """矩阵乘一次算完全部相似度。零外部依赖，几万条以内性能足够。

    持久化用 .npy（向量）+ .json（文本与元数据），目录结构简单到可以直接肉眼检查，
    排查"到底入库了几条"的时候比 HNSW 索引文件友好得多。
    """

    name = "numpy"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.dir = os.path.join(settings.vector_dir, settings.collection)
        os.makedirs(self.dir, exist_ok=True)
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._vectors: list[list[float]] = []
        self._load()

    # -- 持久化 --
    def _vectors_path(self) -> str:
        return os.path.join(self.dir, "vectors.npy")

    def _meta_path(self) -> str:
        return os.path.join(self.dir, "chunks.json")

    def _save(self) -> None:
        import numpy as np

        np.save(self._vectors_path(), np.array(self._vectors, dtype=np.float32))
        with open(self._meta_path(), "w", encoding="utf-8") as fh:
            json.dump({"texts": self._texts, "metas": self._metas}, fh, ensure_ascii=False)

    def _load(self) -> None:
        import numpy as np

        if not os.path.exists(self._vectors_path()):
            return
        self._vectors = np.load(self._vectors_path()).tolist()
        with open(self._meta_path(), encoding="utf-8") as fh:
            data = json.load(fh)
        self._texts = data["texts"]
        self._metas = data["metas"]

    # -- 契约 --
    def count(self) -> int:
        return len(self._texts)

    def reset(self) -> None:
        self._texts, self._metas, self._vectors = [], [], []
        for path in (self._vectors_path(), self._meta_path()):
            if os.path.exists(path):
                os.remove(path)

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        if not embeddings:
            # 不静默生成假向量：没有真实向量就应该明确报错，让调用方去配 embedding
            raise KnowledgeBaseError(
                "numpy 后端需要调用方提供向量。当前 provider 未提供 embedding 能力，"
                "请配置支持 embedding 的厂商（dashscope / openai / ollama），"
                "或改用 VECTOR_BACKEND=chroma（自带本地模型）/ lexical（纯词法）。"
            )
        if len(embeddings) != len(texts):
            raise KnowledgeBaseError(
                f"向量数量({len(embeddings)})与文本数量({len(texts)})不一致"
            )
        self._texts.extend(texts)
        self._metas.extend(metadatas)
        self._vectors.extend([list(e) for e in embeddings])
        self._save()

    def query(
        self,
        query_embedding: Sequence[float] | None,
        *,
        domain: str,
        top_k: int,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        import numpy as np

        if not self._vectors:
            raise KnowledgeBaseError("知识库为空，请先 ingest 建立索引")
        if query_embedding is None:
            raise KnowledgeBaseError(
                "numpy 后端需要 query 向量；当前无 embedding 能力，请改用 lexical 后端"
            )

        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-9)
        mat = np.asarray(self._vectors, dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
        sims = (mat / norms) @ q  # 归一化后点积即余弦相似度

        order = np.argsort(-sims)
        out: list[RetrievedChunk] = []
        for i in order:
            meta = self._metas[int(i)]
            if domain not in ("*", "", "all") and meta.get("domain") != domain:
                continue
            out.append(
                RetrievedChunk(
                    text=self._texts[int(i)],
                    source=meta.get("source", "unknown"),
                    domain=meta.get("domain", "unknown"),
                    score=round(float(sims[int(i)]), 4),
                )
            )
            if len(out) >= top_k:
                break
        return out


# ---------------------------------------------------------------------------
# 纯词法后端（BM25）
# ---------------------------------------------------------------------------
class LexicalBackend:
    """BM25 检索，完全不依赖向量与模型下载。

    为什么值得单独实现一个：它是整个 RAG 链路里**唯一没有外部依赖的一环**，
    因此单测、CI、离线演示都靠它保证"检索这条路一定能通"。
    同时它也是真实可用的降级方案 —— 很多"向量检索效果不好"的场景，
    换回 BM25 反而更稳（尤其是 query 里有明确型号、SKU、专业名词的时候）。

    score 归一化到 0~1（除以本次最高分），这样重排的加权公式对三种后端都成立。
    """

    name = "lexical"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict] = []
        self._tokens: list[Counter] = []
        self._lengths: list[int] = []
        self._df: Counter = Counter()

    def count(self) -> int:
        return len(self._texts)

    def reset(self) -> None:
        self._ids, self._texts, self._metas = [], [], []
        self._tokens, self._lengths, self._df = [], [], Counter()

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        for cid, text, meta in zip(ids, texts, metadatas):
            tokens = _tokenize(text)
            self._ids.append(cid)
            self._texts.append(text)
            self._metas.append(meta)
            self._tokens.append(Counter(tokens))
            self._lengths.append(max(len(tokens), 1))
            for token in set(tokens):
                self._df[token] += 1

    def query(
        self,
        query_embedding: Sequence[float] | None,
        *,
        domain: str,
        top_k: int,
        query_text: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._texts:
            raise KnowledgeBaseError("知识库为空，请先 ingest 建立索引")
        if not query_text:
            raise KnowledgeBaseError("lexical 后端必须传入 query_text（它不做向量检索）")

        n = len(self._texts)
        avg_len = sum(self._lengths) / n
        q_tokens = set(_tokenize(query_text))

        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tokens):
            if domain not in ("*", "", "all") and self._metas[i].get("domain") != domain:
                continue
            score = 0.0
            for token in q_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                # BM25 的 IDF 项：出现在越少文档里的词权重越高
                idf = math.log(1 + (n - self._df[token] + 0.5) / (self._df[token] + 0.5))
                denom = freq + _BM25_K1 * (1 - _BM25_B + _BM25_B * self._lengths[i] / avg_len)
                score += idf * freq * (_BM25_K1 + 1) / denom
            if score > 0:
                scored.append((score, i))

        if not scored:
            return []
        scored.sort(reverse=True)
        top_score = scored[0][0] or 1.0
        return [
            RetrievedChunk(
                text=self._texts[i],
                source=self._metas[i].get("source", "unknown"),
                domain=self._metas[i].get("domain", "unknown"),
                score=round(score / top_score, 4),
            )
            for score, i in scored[:top_k]
        ]


_BACKENDS: dict[str, type] = {
    "chroma": ChromaBackend,
    "numpy": NumpyBackend,
    "lexical": LexicalBackend,
}


def build_backend(settings: Settings) -> VectorBackend:
    """按配置建后端。

    不在这里做"chroma 装不上就偷偷换成 numpy"这种事 —— 后端换了，检索质量、
    分数分布、评测结果全都会变。要么按用户的配置建，要么明确报错。
    """
    cls = _BACKENDS.get(settings.vector_backend)
    if cls is None:
        raise KnowledgeBaseError(
            f"未知的向量库后端 {settings.vector_backend!r}，可选：{', '.join(_BACKENDS)}"
        )
    logger.info("vectorstore.backend", extra={"backend": settings.vector_backend})
    return cls(settings)
