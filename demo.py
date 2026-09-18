"""共享集群一键演示：不需要任何 API Key，不联网。

    python demo.py

演示的顺序是**从底层往上**，正好对应"共享集群支撑上层业务"这句话：

    1. 集群组装      —— 一次 build，五块能力全到位
    2. 知识库检索    —— 切分 / 入库 / 召回 / 重排
    3. MCP 工具调用  —— 走 JSON-RPC，和真实客户端完全同一条路径
    4. Prompt 渲染   —— 模板注册与安全渲染
    5. 记忆          —— 会话记忆（多轮指代）+ 长期事实记忆（时间衰减）
    6. 模型路由      —— 轻/重模型选择与成本记账

每一步都打印可核对的实际数字，没有"看起来在跑"的演示。
"""
from __future__ import annotations

import asyncio
import os
import sys

# 允许直接 `python demo.py` 而不必先 pip install
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 没有 API Key 时自动切离线模式，保证 clone 下来就能跑。
# 显式设置 LLM_PROVIDER 可以覆盖（想连真模型就自己填 .env）
if not (os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")):
    os.environ.setdefault("LLM_PROVIDER", "mock")
    os.environ.setdefault("VECTOR_BACKEND", "lexical")
# 演示要的是干净的输出，日志压到 WARNING。
# 想看完整的结构化日志（每行一条 JSON）就设 LOG_LEVEL=INFO 再跑一次 ——
# 那是项目里日志体系的真实形态，值得单独看一遍。
os.environ.setdefault("LOG_LEVEL", "WARNING")

from ecom_shared import SharedCluster  # noqa: E402
from ecom_shared.mcp.builtin import build_shared_registry  # noqa: E402

LINE = "=" * 74


def step(n: int, title: str) -> None:
    print(f"\n{LINE}\n  {n}. {title}\n{LINE}")


SAMPLE_DOCS = [
    (
        "ads",
        "01_ACOS与广告结构优化.md",
        "广告 ACOS 高于毛利率时应先做搜索词归因：导出搜索词报告，"
        "把高花费零转化的词加否定，把高出单低 ACOS 的词单独提价。\n\n"
        "结构上遵循「一个广告活动一个目标」，不要把拓词和打爆款混在同一个活动里，"
        "否则预算会被低效词吃掉。",
    ),
    (
        "selection",
        "02_市场容量与竞争度评估.md",
        "评估市场容量看三个指标：核心词搜索量、类目 TOP100 销量分布、客单价区间。\n\n"
        "竞争度用头部卖家的评论数判断：TOP10 平均评论数低于 300 属于可切入，"
        "高于 2000 说明新链接很难拿到自然位。",
    ),
    (
        "logistics",
        "01_头程物流渠道对比.md",
        "海运整柜单位成本最低但备货周期 35-45 天，适合销量稳定的常规款；\n\n"
        "空运只在断货救急或测新款时用，单位成本是海运的 4-6 倍，"
        "超过 15% 的货值就不该走空运。",
    ),
]


async def main() -> None:
    print(f"{LINE}\n  电商 AI 生态 · 共享集群（ecom-agent-shared）演示\n{LINE}")

    # ---------------------------------------------------------------- 1
    step(1, "集群组装：一次 build，五块能力全到位")
    cluster = SharedCluster.build()
    info = cluster.describe()
    print(f"  模型接入    : {info['provider_label']}（{info['provider']}）"
          f"{'  ← 离线模式，内容带 [MOCK] 前缀' if info['mock'] else ''}")
    print(f"  轻/重模型   : {info['model_light']} / {info['model_heavy']}")
    print(f"  检索        : 后端={info['retrieval']['backend']}  "
          f"向量化={info['retrieval']['embedding_mode']}  "
          f"切片={info['retrieval']['chunk_size']}/{info['retrieval']['chunk_overlap']}  "
          f"α={info['retrieval']['rerank_alpha']}")
    print(f"  MCP 工具    : {len(info['tools'])} 个 -> "
          f"{', '.join(t['name'] for t in info['tools'])}")
    print(f"  Prompt 模板 : {len(info['prompts'])} 个 -> "
          f"{', '.join(p['name'] for p in info['prompts'])}")

    # ---------------------------------------------------------------- 2
    step(2, "知识库检索：切分 → 入库 → 按域召回 → 混合重排")
    report = await cluster.ingest_documents(SAMPLE_DOCS)
    print(f"  入库        : {report.documents} 篇文档 → {report.chunks} 个切片")
    print(f"  业务域分布  : {report.domains}")

    for query, domain in (("ACOS 太高怎么办", "*"), ("怎么判断竞争度", "selection"),
                          ("海运还是空运", "logistics")):
        hits = await cluster.search(query, domain=domain)
        top = hits[0] if hits else None
        print(f"\n  查询「{query}」（域={domain}）")
        if top:
            print(f"    命中 {len(hits)} 条，top1 融合分={top.score}  域={top.domain}  "
                  f"来源={top.source}")
            print(f"    {top.text[:52].replace(chr(10), ' ')}…")
        else:
            print("    无命中")

    print("\n  说明：融合分 = 0.7 × 向量/词法相似度 + 0.3 × 词面重叠率。")
    print("        α=0.7 让语义主导方向，词面分只做「字面命中却被排后」的纠偏。")

    # ---------------------------------------------------------------- 3
    step(3, "MCP 工具调用：走 JSON-RPC，与真实客户端同一条路径")
    resp = await cluster.tools.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize"}
    )
    print(f"  initialize  : protocolVersion={resp['result']['protocolVersion']}  "
          f"server={resp['result']['serverInfo']['name']}")

    resp = await cluster.tools.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
    )
    print(f"  tools/list  : 返回 {len(resp['result']['tools'])} 个工具定义")
    for tool in resp["result"]["tools"][:3]:
        print(f"                - {tool['name']:16s} {tool['description'][:34]}")

    resp = await cluster.tools.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "rag_search",
                   "arguments": {"query": "海运备货周期多久", "top_k": 1}},
    })
    print(f"  tools/call  : isError={resp['result']['isError']}  "
          f"content={resp['result']['content'][0]['text'][:70]}…")

    # 故意调用一个不存在的工具，演示错误处理
    resp = await cluster.tools.handle({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "no_such_tool", "arguments": {}},
    })
    print(f"  失败语义    : RPC 本身成功，content.isError={resp['result']['isError']}")
    print(f"                {resp['result']['content'][0]['text'][:60]}")
    print("                （工具失败不该让整次 RPC 挂掉 —— Agent 循环要把")
    print("                  失败原因当成观察结果喂回模型，让它自己换策略）")

    # ---------------------------------------------------------------- 4
    step(4, "Prompt 注册中心：模板化、可覆写、缺失变量不炸")
    print(f"  已注册      : {cluster.prompts.names()}")
    rendered = cluster.prompts.render(
        "rag_answer",
        persona="跨境电商售前顾问",
        question="ACOS 太高怎么办",
        context="[1] 先做搜索词归因，否定高花费零转化词",
    )
    print("  渲染结果（前 3 行）：")
    for line in rendered.splitlines()[:3]:
        print(f"    {line}")
    missing = cluster.prompts.render("rag_answer", persona="顾问")
    print(f"  缺变量时    : 保留字面量而非抛异常 -> {'{question}' in missing}")

    # ---------------------------------------------------------------- 5
    step(5, "记忆：会话记忆（多轮）+ 长期事实记忆（时间衰减）")
    mem = cluster.session_memory
    mem.append("demo", "user", "帮我看看这款降噪耳机的广告数据")
    mem.append("demo", "assistant", "好的，正在拉取该 ASIN 的广告报表")
    mem.append("demo", "user", "它有什么优惠活动")
    print(f"  会话记忆    : 共 {len(mem.history('demo'))} 轮")
    print(f"                最近提问 = {mem.last_user_query('demo')}")
    print("                「它」要靠上面两轮才能解析出指的是哪款耳机")
    print(f"  转录文本    : {mem.transcript('demo').splitlines()[-1][:40]}")

    lt = cluster.longterm_memory
    lt.remember("主营站点", "北美站（US + CA）", scope="shop_001")
    lt.remember("价格敏感度", "对价格敏感，依赖优惠券转化", scope="shop_001")
    recalled = lt.recall("站点 价格", scope="shop_001", top_k=2)
    print(f"\n  长期记忆    : {len(lt.all('shop_001'))} 条事实，召回 {len(recalled)} 条")
    for fact in recalled:
        print(f"                - {fact.text}（{fact.age_days():.1f} 天前，"
              f"衰减权重 {fact.decay():.2f}）")
    print("                半衰期 14 天：过期事实即使字面匹配也排不过新事实")

    # ---------------------------------------------------------------- 6
    step(6, "模型路由与成本记账：简单任务不占用大模型额度")
    for text in ("判断这句话是不是退款诉求", "帮我分析竞品定价并给出策略方案"):
        model, tier, score = cluster.gateway.resolve_route(text)
        _, meta = await cluster.gateway.complete("你是运营助手", text)
        print(f"  「{text[:16]}…」")
        print(f"    → 档位={tier}  得分={score}  实际调用={meta['model']}")

    usage = cluster.usage.snapshot()
    print(f"\n  用量汇总    : {usage['calls']} 次调用 / {usage['total_tokens']} tokens")
    print(f"    按模型    : {usage['by_model']}")
    print(f"    按档位    : {usage['by_tier']}")
    print(f"    估算成本  : ￥{usage['estimated_cost_cny']}（仅估算，真实账单以厂商后台为准）")

    # ---------------------------------------------------------------- 边界
    step(7, "已知边界：明确说清楚哪些是没做的")
    print("  · 会话记忆存在进程内，多实例部署需换 Redis（接口已收敛成 MemoryStore）")
    print("  · 用量统计不落库，进程重启即清零，做真实成本核算需接时序库")
    print("  · MCP HTTP 端点无鉴权，默认只绑 127.0.0.1；对外提供必须先加鉴权")
    print("  · 规则路由靠关键词表，换业务域要重调词表（要大改可换小模型做分类）")

    print(f"\n{LINE}\n  演示结束。换成真实模型：在 .env 填 LLM_API_KEY，代码零改动。\n{LINE}")


if __name__ == "__main__":
    asyncio.run(main())
