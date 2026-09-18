"""RAG 服务端到端单测（全离线：provider=mock + VECTOR_BACKEND=lexical）。

覆盖的是"链路能不能通、分数分布合不合理、边界情况有没有兜住"，
不覆盖"检索准不准" —— 那是评测的事，见 `scripts/eval_retrieval.py`。
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ecom_shared.errors import KnowledgeBaseError
from ecom_shared.rag.backends import LexicalBackend, build_backend
from ecom_shared.rag.service import RagService

from tests.fixtures import SAMPLE_DOCS


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def svc(tmp_path, settings):
    return RagService(replace(settings, vector_dir=str(tmp_path / "vs")))


def test_backend_factory_respects_config(settings, tmp_path):
    backend = build_backend(replace(settings, vector_dir=str(tmp_path / "x")))
    assert isinstance(backend, LexicalBackend)


def test_backend_factory_rejects_unknown(tmp_path, settings):
    with pytest.raises(KnowledgeBaseError, match="未知的向量库后端"):
        build_backend(replace(settings, vector_backend="faiss", vector_dir=str(tmp_path)))


def test_ingest_reports_documents_and_domains(svc):
    report = run(svc.ingest_documents(SAMPLE_DOCS))
    assert report.documents == 3
    assert report.chunks >= 3
    assert set(report.domains) == {"ads", "selection", "logistics"}
    assert report.backend == "lexical"


def test_ingest_empty_corpus_raises(svc):
    with pytest.raises(KnowledgeBaseError):
        run(svc.ingest_documents([]))


def test_ingest_dir_missing_docs_raises(svc, tmp_path):
    with pytest.raises(KnowledgeBaseError):
        run(svc.ingest_dir(str(tmp_path / "nothing")))


def test_search_finds_the_right_domain(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    hits = run(svc.search("ACOS 过高怎么优化"))
    assert hits
    assert hits[0].domain == "ads"
    assert 0.0 <= hits[0].score <= 1.0


def test_search_domain_filter_excludes_other_domains(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    hits = run(svc.search("优化", domain="selection"))
    assert all(h.domain == "selection" for h in hits)


def test_search_empty_query_raises(svc):
    """空 query 去检索会让后端随便返回几条，看起来"有结果"其实全是噪声。
    明确报错好过返回噪声 —— 这个坑在本地 embedding 降级时踩过。"""
    run(svc.ingest_documents(SAMPLE_DOCS))
    with pytest.raises(KnowledgeBaseError, match="不能为空"):
        run(svc.search("   "))


def test_search_on_empty_kb_raises(svc):
    with pytest.raises(KnowledgeBaseError, match="知识库为空"):
        run(svc.search("ACOS"))


def test_search_respects_top_k(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    assert len(run(svc.search("选品 物流 广告", top_k=2))) <= 2


def test_search_as_context_returns_source_metadata(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    ctx = run(svc.search_as_context("ACOS", top_k=1))
    assert ctx[0]["source"] and ctx[0]["domain"] and "text" in ctx[0]


def test_reingest_with_rebuild_does_not_duplicate(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    first = svc.count()
    run(svc.ingest_documents(SAMPLE_DOCS))
    assert svc.count() == first


def test_ingest_without_rebuild_appends(svc):
    run(svc.ingest_documents(SAMPLE_DOCS))
    first = svc.count()
    run(svc.ingest_documents(SAMPLE_DOCS, rebuild=False))
    assert svc.count() == first * 2


def test_stats_exposes_retrieval_configuration(svc):
    stats = svc.stats()
    assert stats["backend"] == "lexical"
    assert stats["chunk_size"] > 0
    assert stats["rerank_alpha"] == 0.7


def test_numpy_backend_requires_vectors(tmp_path, settings):
    """numpy 后端缺向量时报错而不是静默造假向量 —— 假向量会让检索变成随机排序。"""
    from ecom_shared.rag.backends import NumpyBackend

    backend = NumpyBackend(replace(settings, vector_dir=str(tmp_path)))
    with pytest.raises(KnowledgeBaseError, match="需要调用方提供向量"):
        backend.add(ids=["1"], texts=["t"], metadatas=[{}], embeddings=None)


def test_lexical_backend_normalizes_score_to_unit_range(tmp_path, settings):
    backend = LexicalBackend(replace(settings, vector_dir=str(tmp_path)))
    backend.add(
        ids=["1", "2"],
        texts=["ACOS 优化投放预算", "选品看市场容量"],
        metadatas=[{"domain": "ads"}, {"domain": "selection"}],
    )
    hits = backend.query(None, domain="*", top_k=2, query_text="ACOS 优化")
    assert hits[0].score == 1.0  # 最高分归一化到 1
    assert all(0.0 <= h.score <= 1.0 for h in hits)
