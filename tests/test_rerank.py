"""重排单测：验证它确实起到"纠偏"作用，而不只是换个排序。"""
from __future__ import annotations

from dataclasses import dataclass

from ecom_shared.rag.rerank import DEFAULT_ALPHA, lexical_score, rerank, tokenize


@dataclass
class _C:
    text: str
    score: float = 0.5
    source: str = "t"
    domain: str = "ads"


def test_tokenize_splits_cjk_and_ascii():
    tokens = tokenize("ACOS 优化 acos")
    assert "acos" in tokens  # 统一小写
    assert "优" in tokens and "化" in tokens


def test_lexical_score_full_hit():
    assert lexical_score("ACOS", "广告 ACOS 过高") == 1.0


def test_lexical_score_empty_query_is_zero():
    """query 没有有效 token 时返回 0，不硬打分 —— 否则会给出误导性的高相关度。"""
    assert lexical_score("", "任意内容") == 0.0
    assert lexical_score("!!!", "任意内容") == 0.0


def test_rerank_promotes_lexical_hit_over_slightly_higher_vector_score():
    """核心场景：语义分略低但字面命中关键词的片段，应该被提上来。"""
    candidates = [
        _C("物流时效和海运清关的关系，影响备货节奏", score=0.66),
        _C("客户要求退款时的合规话术与升级判断流程", score=0.60),
        _C("选品时市场容量与竞争强度的评估框架", score=0.50),
    ]
    top = rerank("退款话术怎么写", candidates, top_k=1)
    assert "退款" in top[0].text


def test_rerank_rewrites_score_in_place():
    """前端展示的"相关度"必须反映重排后的结果，否则用户看到的排序无法解释。"""
    c = _C("ACOS 优化", score=0.1)
    rerank("ACOS 优化", [c], top_k=1)
    assert c.score > 0.1


def test_alpha_one_falls_back_to_pure_vector_order():
    """alpha=1 时退化为纯向量排序 —— 评测扫参时要用这个端点做基准。"""
    low_vec_high_lex = _C("ACOS 优化投放", score=0.2)
    high_vec = _C("完全不相关的内容", score=0.9)
    top = rerank("ACOS 优化投放", [low_vec_high_lex, high_vec], top_k=1, alpha=1.0)
    assert top[0] is high_vec


def test_default_alpha_is_zero_point_seven():
    assert DEFAULT_ALPHA == 0.7


def test_rerank_empty_candidates():
    assert rerank("q", [], top_k=3) == []


def test_rerank_top_k_floor_is_one():
    """top_k 传 0 或负数时兜到 1，而不是返回空 —— 调用方意图是"要结果"。"""
    got = rerank("ACOS", [_C("ACOS 优化")], top_k=0)
    assert len(got) == 1
