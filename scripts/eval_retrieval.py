"""检索离线评测：把「检索准不准」变成一个可复现的数字。

**为什么需要一个独立脚本，而不是写在测试里：**

单测回答"代码有没有按预期运行"（切分对不对、重排有没有把字面命中的提上来），
评测回答"检索效果好不好"。后者依赖具体语料，换个知识库结果就变，
不适合当断言 —— 一旦语料更新测试就红，最后大家会把测试删掉。

用法：

    python scripts/eval_retrieval.py                           # 用内置样例
    python scripts/eval_retrieval.py --cases cases.json        # 用自己的用例
    python scripts/eval_retrieval.py --docs ../某仓库/data/docs # 用真实语料
    python scripts/eval_retrieval.py --sweep-alpha             # 扫参找最优 α

用例文件格式（JSON 数组）：

    [
      {"id": "acos-01", "domain": "ads", "question": "ACOS 太高怎么优化",
       "expect_source": "01_ACOS与广告结构优化.md"},
      ...
    ]

`expect_source` 是**必须命中的文件名**。这是评测的关键设计：
不能只看"返回了结果"，要看"返回的是不是对的那篇"。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 评测必须离线可复现，否则同一份语料两次跑出不同数字，评测就失去意义
os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("VECTOR_BACKEND", "lexical")
os.environ.setdefault("LOG_LEVEL", "ERROR")

from ecom_shared import SharedCluster  # noqa: E402

# ---------------------------------------------------------------------------
# 内置样例语料与用例：在没有外部语料时也能跑出一份有意义的报告
# ---------------------------------------------------------------------------
DEFAULT_DOCS = [
    ("ads", "01_ACOS与广告结构优化.md",
     "广告 ACOS 高于毛利率时应先做搜索词归因：导出搜索词报告，把高花费零转化的词加否定。\n\n"
     "结构上遵循「一个广告活动一个目标」，不要把拓词和打爆款混在同一个活动里。"),
    ("ads", "02_关键词投放与出价策略.md",
     "关键词分三类：核心词保排名、长尾词要转化、竞品词做卡位。\n\n"
     "出价用「动态竞价 - 仅降低」起步，拿到 2 周数据后再按实际转化调。"),
    ("selection", "01_选品核心指标与筛选框架.md",
     "选品看市场容量、竞争强度、利润空间三个维度，三者不能同时差。\n\n"
     "毛利率低于 25% 的品类在当前物流成本下基本不做。"),
    ("selection", "02_市场容量与竞争度评估.md",
     "竞争度用头部卖家的评论数判断：TOP10 平均评论数低于 300 属于可切入，高于 2000 很难拿自然位。"),
    ("logistics", "01_头程物流渠道对比.md",
     "海运整柜单位成本最低但备货周期 35-45 天，适合销量稳定的常规款。\n\n"
     "空运只在断货救急或测新款时用，单位成本是海运的 4-6 倍。"),
    ("logistics", "03_清关关税与合规.md",
     "欧盟 VAT 与关税起征点、美国 800 美元免税额度、WEEE 与包装法注册，都要在发货前确认。"),
    ("support", "02_退款退货与纠纷处理.md",
     "客户要求退款时先确认是否符合退货政策，符合则直接同意并给出退货面单，不要反复挽留。\n\n"
     "纠纷升级到 A-to-Z 之前必须先给出可执行的解决方案。"),
    ("review", "01_评论情感分析与痛点归类.md",
     "把差评按「产品质量 / 物流时效 / 描述不符 / 客服响应」四类归因，才能定位到可改进的动作。"),
]

DEFAULT_CASES = [
    {"id": "ads-01", "domain": "ads", "question": "ACOS 太高怎么优化",
     "expect_source": "01_ACOS与广告结构优化.md"},
    {"id": "ads-02", "domain": "ads", "question": "关键词出价策略怎么定",
     "expect_source": "02_关键词投放与出价策略.md"},
    {"id": "sel-01", "domain": "selection", "question": "怎么判断一个类目竞争激不激烈",
     "expect_source": "02_市场容量与竞争度评估.md"},
    {"id": "sel-02", "domain": "selection", "question": "选品要看哪几个维度",
     "expect_source": "01_选品核心指标与筛选框架.md"},
    {"id": "log-01", "domain": "logistics", "question": "海运和空运怎么选",
     "expect_source": "01_头程物流渠道对比.md"},
    {"id": "log-02", "domain": "logistics", "question": "欧盟 VAT 和清关合规要注意什么",
     "expect_source": "03_清关关税与合规.md"},
    {"id": "sup-01", "domain": "support", "question": "客户要求退款怎么处理",
     "expect_source": "02_退款退货与纠纷处理.md"},
    {"id": "rev-01", "domain": "review", "question": "差评怎么归类分析",
     "expect_source": "01_评论情感分析与痛点归类.md"},
]


@dataclass
class CaseResult:
    case_id: str
    question: str
    domain: str
    top_source: str
    top_score: float
    hit: bool


def load_cases(path: str | None) -> list[dict]:
    if not path:
        return DEFAULT_CASES
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_docs(docs_dir: str | None) -> list[tuple[str, str, str]]:
    if not docs_dir:
        return DEFAULT_DOCS
    from ecom_shared.rag.chunking import load_documents

    docs = load_documents(docs_dir)
    if not docs:
        raise SystemExit(f"目录 {docs_dir} 下没有找到 .md 文档（约定 <dir>/<domain>/*.md）")
    return docs


async def run_eval(cluster, cases: list[dict], docs, top_k: int) -> list[CaseResult]:
    report = await cluster.ingest_documents(docs)
    print(f"语料：{report.documents} 篇文档 → {report.chunks} 个切片，"
          f"域分布 {report.domains}")
    print(f"后端：{report.backend}   向量化：{report.embedding_mode}")
    print(f"用例：{len(cases)} 条   top_k={top_k}   α={cluster.settings.rerank_alpha}")
    print("-" * 78)

    results: list[CaseResult] = []
    for case in cases:
        hits = await cluster.search(
            case["question"], domain=case.get("domain", "*"), top_k=top_k
        )
        top = hits[0] if hits else None
        hit = bool(top and top.source == case["expect_source"])
        results.append(
            CaseResult(
                case_id=case.get("id", case["question"][:12]),
                question=case["question"],
                domain=case.get("domain", "*"),
                top_source=top.source if top else "(无结果)",
                top_score=top.score if top else 0.0,
                hit=hit,
            )
        )
        mark = "命中" if hit else "未命中"
        print(f"  [{mark}] {results[-1].case_id:8s} 「{case['question'][:22]}」")
        if not hit:
            print(f"           期望={case['expect_source']}")
            print(f"           实际={results[-1].top_source} (score={results[-1].top_score})")
    return results


def summarize(results: list[CaseResult]) -> int:
    total = len(results)
    hits = sum(1 for r in results if r.hit)
    rate = hits / total * 100 if total else 0.0

    print("-" * 78)
    print(f"命中率：{hits}/{total} = {rate:.1f}%")

    by_domain: dict[str, list[CaseResult]] = {}
    for r in results:
        by_domain.setdefault(r.domain, []).append(r)
    for domain, items in sorted(by_domain.items()):
        ok = sum(1 for i in items if i.hit)
        print(f"  {domain:12s} {ok}/{len(items)}")

    print("-" * 78)
    print("说明：命中 = top1 的 source 与期望文件一致。")
    print("      这个口径比「返回了结果就算命中」严格 —— 不看排序的评测没有意义。")
    return 0 if hits == total else 1


async def sweep_alpha(cluster, cases: list[dict], docs, top_k: int) -> None:
    """扫参：α 从 0.5 到 1.0，看命中率怎么变。

    这是为了让 α=0.7 这个默认值有依据，而不是拍脑袋定的。
    α=1.0 是纯向量（本评测里 lexical 后端下即纯词法）的基准端点。
    """
    await cluster.ingest_documents(docs)
    print(f"扫参：top_k={top_k}，后端={cluster.settings.vector_backend}")
    print(f"{'α':>6}  {'命中率':>8}   说明")
    print("-" * 60)
    best = (0.0, 0.0)
    for alpha in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        hits = 0
        for case in cases:
            got = await cluster.search(
                case["question"], domain=case.get("domain", "*"), top_k=top_k, alpha=alpha
            )
            if got and got[0].source == case["expect_source"]:
                hits += 1
        rate = hits / len(cases) * 100
        note = "纯语义端点（重排不起作用）" if alpha == 1.0 else ""
        if rate > best[1]:
            best = (alpha, rate)
        print(f"{alpha:>6.1f}  {rate:>7.1f}%   {note}")
    print("-" * 60)
    print(f"最优 α = {best[0]}（{best[1]:.1f}%）。默认值是 "
          f"{cluster.settings.rerank_alpha}，两者是否一致可据此判断要不要调。")


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG 检索离线评测")
    parser.add_argument("--cases", help="用例 JSON 文件路径")
    parser.add_argument("--docs", help="语料目录（约定 <dir>/<domain>/*.md）")
    parser.add_argument("--top-k", type=int, default=3, help="检索返回条数（默认 3）")
    parser.add_argument("--sweep-alpha", action="store_true", help="扫参找最优 α")
    args = parser.parse_args()

    cases = load_cases(args.cases)
    docs = load_docs(args.docs)
    cluster = SharedCluster.build()

    if args.sweep_alpha:
        asyncio.run(sweep_alpha(cluster, cases, docs, args.top_k))
        return 0
    results = asyncio.run(run_eval(cluster, cases, docs, args.top_k))
    return summarize(results)


if __name__ == "__main__":
    raise SystemExit(main())
