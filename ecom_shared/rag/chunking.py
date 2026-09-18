"""文档切分：语义优先，重叠只补在真正被切断的地方。

切分策略（RAG 面试第一问通常就是这里）：

- **段落优先** —— 先按空行分段。固定长度切分会把完整段落拦腰截断，破坏语义完整性，
  检索出来的片段读着就是半句话。
- **长段按句切** —— 段落超过阈值时按句末标点切句，滚动累积到接近阈值为止。
- **重叠只补在切断处** —— 相邻块尾部保留 overlap 个字符，缓解跨块语义断裂。
  一句话正好卡在块边界时，重叠保证它至少完整地出现在某一个切片里。

第三条是这里最容易做错的地方，代码注释里写了踩过的坑。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")


@dataclass(frozen=True)
class Chunk:
    """一个语义切片。

    frozen 而不是普通 dataclass：切片一旦生成就不该被改写，
    检索链路里多个 Agent 会同时持有同一批切片，可变对象容易出诡异 bug。
    """

    text: str
    index: int
    source: str
    domain: str
    #: 文档在语料中的全局序号。用于生成全局唯一的 chunk_id ——
    #: 同一域下多篇文档的 index 都从 0 开始，只靠 domain+index 会撞 ID，
    #: 导致写入向量库时被拒绝（ChromaDB 要求 ID 唯一）。
    doc_ordinal: int = 0

    @property
    def chunk_id(self) -> str:
        return f"{self.domain}-{self.doc_ordinal:03d}-{self.index:04d}"

    def to_metadata(self) -> dict[str, str | int]:
        return {
            "source": self.source,
            "domain": self.domain,
            "index": self.index,
            "doc_ordinal": self.doc_ordinal,
        }


def split_sentences(paragraph: str) -> list[str]:
    """按句末标点切句，保留标点。"""
    parts = _SENTENCE_SPLIT_RE.split(paragraph)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    text: str,
    *,
    source: str,
    domain: str,
    chunk_size: int = 600,
    overlap: int = 50,
    doc_ordinal: int = 0,
) -> list[Chunk]:
    """把一篇文档切成语义切片。

    重叠只加在"确实被切断"的地方：

    - 段落本身没超阈值 → 整段独立成块，**不**拼接上一段的尾巴。
      之前无差别给每个块都加前缀，导致知识库里几乎每个切片都被上一段内容污染
      （中文文档大多整段都短于 600 字），向量被无关文本稀释，召回精度下降。
    - 长段被切成多块 → 只在**同一段内**的相邻块之间补重叠，
      保证卡在边界上的那句话至少完整地出现在某一个块里。
      之前这段逻辑重复加了两次重叠（滚动累积时加一次，最后统一又加一次）。

    这个 bug 很隐蔽：切分数量完全正常、检索也不报错，只是召回内容里混着
    上一段的尾巴 —— 不看切片原文根本发现不了。
    """
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]

    segments: list[str] = []
    # joins[i] 表示第 i 块是否「与第 i-1 块同属一个被切断的长段」，
    # 只有这种情况才需要补重叠
    joins: list[bool] = []

    for para in paragraphs:
        if len(para) <= chunk_size:
            segments.append(para)
            joins.append(False)
            continue
        # 长段落：按句滚动累积
        buffer = ""
        first_in_para = True
        for sentence in split_sentences(para):
            if not buffer:
                buffer = sentence
                continue
            if len(buffer) + len(sentence) <= chunk_size:
                buffer += sentence
            else:
                segments.append(buffer)
                joins.append(not first_in_para)
                first_in_para = False
                buffer = sentence
        if buffer:
            segments.append(buffer)
            joins.append(not first_in_para)

    chunks: list[Chunk] = []
    for idx, segment in enumerate(segments):
        body = segment
        if idx > 0 and overlap and joins[idx]:
            body = segments[idx - 1][-overlap:] + segment
        chunks.append(
            Chunk(
                text=body,
                index=idx,
                source=source,
                domain=domain,
                doc_ordinal=doc_ordinal,
            )
        )
    return chunks


def chunk_corpus(
    documents: list[tuple[str, str, str]],
    *,
    chunk_size: int = 600,
    overlap: int = 50,
) -> list[Chunk]:
    """把 (domain, filename, content) 三元组整批切分，并维护全局文档序号。

    文档序号在整批范围内递增，而不是每篇从 0 开始 —— 这是 chunk_id 全局唯一的前提。
    """
    all_chunks: list[Chunk] = []
    for ordinal, (domain, source, content) in enumerate(documents):
        all_chunks.extend(
            chunk_document(
                content,
                source=source,
                domain=domain,
                chunk_size=chunk_size,
                overlap=overlap,
                doc_ordinal=ordinal,
            )
        )
    return all_chunks


def load_documents(docs_dir: str) -> list[tuple[str, str, str]]:
    """扫描目录下的 .md 文件，返回 (domain, filename, content)。

    目录约定：``<docs_dir>/<domain>/*.md``，每个子目录是一个业务域。
    业务域会成为切片的元数据，检索时按域过滤，避免跨领域噪声 ——
    选品的问题不该召回广告投放的知识。

    直接放在根目录下的 .md 归入 ``general`` 域。
    """
    import os

    results: list[tuple[str, str, str]] = []
    if not os.path.isdir(docs_dir):
        return results
    for entry in sorted(os.listdir(docs_dir)):
        sub = os.path.join(docs_dir, entry)
        if os.path.isdir(sub):
            for name in sorted(os.listdir(sub)):
                if not name.endswith(".md"):
                    continue
                with open(os.path.join(sub, name), encoding="utf-8") as fh:
                    results.append((entry, name, fh.read()))
        elif entry.endswith(".md"):
            with open(sub, encoding="utf-8") as fh:
                results.append(("general", entry, fh.read()))
    return results
