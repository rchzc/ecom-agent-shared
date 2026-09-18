"""检索重排：向量召回 + 词面重排的轻量融合。

**为什么需要重排**（面试可以主动讲这个）：

纯向量召回按语义相似度排序，但中文短 query 有两个典型问题：

- **同义不同词**：用户问"退款怎么弄"，文档里写"退货流程"，向量能对上（这是向量的强项）。
- **字面命中却被排后**：用户问"ACOS 怎么优化"，含"ACOS"的那篇文档语义相似度略低，
  被一堆泛泛讲"广告投放策略"的片段挤到后面 —— 关键词明明命中了，却没排前面。

重排就是在召回的候选里用"词面重叠"做二次打分，把真正含关键词的片段提上来。

**设计原则：**

- 纯本地、无外部依赖、不联网、确定可复现 —— 离线评测脚本可以直接跑，不需要模型和网络。
- 融合分 = α·语义分 + (1-α)·词面分。语义分来自向量库相似度（已归一化到 0~1），
  词面分来自 query 与 chunk 的字符/词重叠率，对中文友好。
- α 默认 0.7，让向量召回主导方向，词面分只做纠偏。**不做成 1.0 是有意的**：
  纯词面匹配会退化成关键词搜索，丢掉向量的最大价值。
- 本模块不 import 向量库后端：只依赖传入对象的 .text / .score 字段（鸭子类型）。
  这样离线评测脚本不需要加载 chromadb 也能复用同一套打分逻辑 ——
  评测逻辑和线上检索逻辑是同一份代码，评测结果才有意义。
"""
from __future__ import annotations

import re
from typing import Any, Sequence

# 语义分权重。0.7 让向量召回主导方向，0.3 的词面分用于"字面命中却被排后"的纠偏。
DEFAULT_ALPHA = 0.7

# 英文/数字词保留原词，中文按单字切（中文无空格，字符级重叠更稳）。
# 单字切分的代价是"的/了/是"这类虚词也参与重叠，但因为用集合去重且除以 query
# token 种类数，虚词带来的噪声被摊薄，实测影响可忽略。
_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+|[一-鿿]")


def tokenize(text: str) -> list[str]:
    """把文本切成可比较的 token：ASCII 词 + 汉字单字，统一小写。"""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def lexical_score(query: str, chunk_text: str) -> float:
    """query 与 chunk 的词面重叠率（命中 token 种类数 / query token 种类数）。

    用集合去重，避免 query 里重复出现的词抬高分数；chunk 里命中即算。
    返回 0~1。query 无有效 token 时返回 0（不强行打分）。
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return 0.0
    text = chunk_text.lower()
    hit = sum(1 for t in q_tokens if t in text)
    return hit / len(q_tokens)


def rerank(
    query: str,
    candidates: Sequence[Any],
    *,
    top_k: int,
    alpha: float = DEFAULT_ALPHA,
) -> list[Any]:
    """在召回候选里做轻量重排，返回前 top_k。

    candidates 需具备 .text（原文）与 .score（语义相似度，0~1）两个字段；
    返回的是同一批对象（按融合分排序），并把 .score 原地改写为融合分，
    使前端展示的"相关度"反映重排后的结果。

    **只重排序、不跨域。** 调用方应在传入前就按 domain 过滤好候选 ——
    重排不负责过滤，两件事混在一起后就没法解释"为什么这条没出现"了。

    alpha 暴露成参数而不是写死常量，是为了让评测脚本能扫参：
    在领域语料上跑 α ∈ {0.5, 0.6, 0.7, 0.8, 0.9}，看命中率怎么变，
    而不是凭感觉拍一个 0.7。
    """
    if not candidates:
        return []
    if top_k < 1:
        top_k = 1

    scored: list[tuple[float, Any]] = []
    for c in candidates:
        lex = lexical_score(query, c.text)
        fused = alpha * float(getattr(c, "score", 0.0)) + (1 - alpha) * lex
        # 原地改写 score，让下游透明拿到重排后的相关度
        try:
            c.score = round(fused, 4)
        except (AttributeError, TypeError):
            pass
        scored.append((fused, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:top_k]]


if __name__ == "__main__":
    # 自测：不依赖 chromadb / 网络，直接跑即可验证重排把"字面命中"的片段提上来。
    from dataclasses import dataclass

    @dataclass
    class _C:
        text: str
        score: float = 0.5
        source: str = "self-test"

    # 场景：一个语义分略高(0.66)但无关键词的干扰片段，
    # 和一个语义分接近(0.60)、但字面命中"退款/话术"的片段。
    # rerank 作为"纠偏器"，应把字面命中的片段提上来（而非无脑跟语义分）。
    cand = [
        _C("客户要求退款时的合规话术与升级判断流程", score=0.60),
        _C("物流时效和海运清关的关系，影响备货节奏", score=0.66),
        _C("选品时市场容量与竞争强度的评估框架", score=0.50),
    ]
    out = rerank("退款话术怎么写", cand, top_k=2)
    assert "退款" in out[0].text, "重排失败：字面命中关键词的片段应排到第一"
    print("rerank 自测通过：top1 =", out[0].text[:20], "| fused =", out[0].score)
    print("lexical_score('ACOS 优化', '广告 ACOS 过高优化投放预算') =",
          lexical_score("ACOS 优化", "广告 ACOS 过高优化投放预算"))
