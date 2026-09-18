"""切分逻辑单测。重点验证「重叠只补在真正被切断的地方」这条规则。

这条规则是最容易写错、写错了又最难发现的地方：
切出来的块数完全正常、检索也不报错，只是每个切片都被上一段的尾巴污染了。
"""
from __future__ import annotations

import pytest

from ecom_shared.rag.chunking import (
    Chunk,
    chunk_corpus,
    chunk_document,
    load_documents,
    split_sentences,
)


def test_short_paragraphs_become_standalone_chunks():
    """整段短于阈值时，每段独立成块 —— 不该拼接上一段。"""
    text = "第一段内容。\n\n第二段内容。\n\n第三段内容。"
    chunks = chunk_document(text, source="d.md", domain="ads", chunk_size=600, overlap=50)
    assert [c.text for c in chunks] == ["第一段内容。", "第二段内容。", "第三段内容。"]


def test_long_paragraph_split_by_sentence_with_overlap():
    """长段被切断时，后续块要带上上一块尾巴，保证边界句至少完整出现在某一块里。"""
    sentences = [f"这是第{i}句用于测试的句子，长度足够撑满阈值。" for i in range(12)]
    text = "".join(sentences)
    chunks = chunk_document(text, source="d.md", domain="ads", chunk_size=100, overlap=20)
    assert len(chunks) > 2
    # 第二块起应包含上一块的尾部内容
    tail = chunks[0].text[-20:]
    assert tail in chunks[1].text


def test_overlap_not_applied_across_paragraphs():
    """跨段落不补重叠：不同段之间没有"被切断的语义"，补了只是噪声。"""
    para_a = "A" * 80
    para_b = "B" * 80
    chunks = chunk_document(f"{para_a}\n\n{para_b}", source="d.md", domain="ads",
                            chunk_size=100, overlap=20)
    assert len(chunks) == 2
    assert not chunks[1].text.startswith("A")


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_document("abc", source="d", domain="x", chunk_size=50, overlap=50)


def test_chunk_id_is_globally_unique_across_documents():
    """同域下多篇文档如果只靠 domain+index 编号会撞 ID，写库直接失败。"""
    docs = [("ads", "a.md", "内容一" * 50), ("ads", "b.md", "内容二" * 50)]
    chunks = chunk_corpus(docs, chunk_size=200, overlap=20)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0].startswith("ads-000-")


def test_chunk_is_frozen():
    """切片是跨 Agent 共享的只读对象，被意外改写会出难查的 bug。"""
    chunk = Chunk(text="x", index=0, source="d", domain="ads")
    with pytest.raises(Exception):
        chunk.text = "y"  # type: ignore[misc]


def test_split_sentences_keeps_punctuation():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_load_documents_groups_by_domain(tmp_path):
    (tmp_path / "ads").mkdir()
    (tmp_path / "ads" / "a.md").write_text("广告内容", encoding="utf-8")
    (tmp_path / "selection").mkdir()
    (tmp_path / "selection" / "b.md").write_text("选品内容", encoding="utf-8")
    (tmp_path / "root.md").write_text("根目录文档", encoding="utf-8")

    docs = load_documents(str(tmp_path))
    by_domain = {d: len([1 for dom, _, _ in docs if dom == d]) for d in ("ads", "selection", "general")}
    assert by_domain == {"ads": 1, "selection": 1, "general": 1}


def test_load_documents_missing_dir_returns_empty():
    assert load_documents("this/does/not/exist") == []
