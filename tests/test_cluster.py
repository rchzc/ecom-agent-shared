"""共享集群门面单测：验证一次组装之后，业务侧要用的东西都齐了。

这一层是"共享集群"这个词的实物对应 —— 它塌了，六个业务包就各自复制一份组装代码。
"""
from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from ecom_shared import SharedCluster
from ecom_shared.memory.longterm import JsonFileStore

from tests.fixtures import SAMPLE_DOCS


def run(coro):
    return asyncio.run(coro)


def test_build_assembles_all_parts(cluster):
    assert cluster.gateway is not None
    assert cluster.rag is not None
    assert cluster.prompts.has("rag_answer")
    assert cluster.session_memory is not None
    assert cluster.longterm_memory is not None
    assert cluster.tools.has("rag_search")


def test_describe_exposes_mock_flag(cluster):
    """把 mock / 后端 / embedding 模式全暴露出来，避免
    "看起来在跑真模型其实是 mock" 这种自欺。"""
    info = cluster.describe()
    assert info["mock"] is True
    assert info["retrieval"]["backend"] == "lexical"
    assert info["provider_label"]
    assert len(info["tools"]) == len(cluster.tools.names())
    assert info["longterm_memory"]["enabled"] is True


def test_cluster_delegates_ingest_and_search(cluster):
    report = run(cluster.ingest_documents(SAMPLE_DOCS))
    assert report.chunks > 0
    hits = run(cluster.search("选品维度"))
    assert hits and hits[0].domain == "selection"


def test_tools_work_end_to_end_through_cluster(cluster):
    run(cluster.ingest_documents(SAMPLE_DOCS))
    out = run(cluster.tools.call("rag_search", {"query": "ACOS 怎么降"}))
    assert out["ok"] is True
    assert out["result"]["count"] >= 1
    assert out["result"]["contexts"][0]["source"]


def test_memory_tools_round_trip(cluster):
    run(cluster.tools.call("memory_remember", {"key": "价格敏感度", "value": "对价格敏感", "scope": "u1"}))
    out = run(cluster.tools.call("memory_recall", {"query": "价格", "scope": "u1"}))
    assert out["result"]["count"] == 1
    assert out["result"]["facts"][0]["value"] == "对价格敏感"


def test_session_context_tool(cluster):
    cluster.session_memory.append("s1", "user", "推荐一款降噪耳机")
    cluster.session_memory.append("s1", "assistant", "推荐 A 款")
    out = run(cluster.tools.call("session_context", {"session_id": "s1"}))
    assert out["result"]["last_user_query"] == "推荐一款降噪耳机"


def test_product_provider_not_injected_is_explicit(cluster):
    """共享层不认识业务数据。没注入时要明确说"没注入"，
    不能返回一个空 dict 让 Agent 以为"查了但没这个商品"。"""
    out = run(cluster.tools.call("product_query", {"name": "某耳机"}))
    assert out["ok"] is True
    assert out["result"]["found"] is False
    assert "未注入" in out["result"]["reason"]


def test_product_provider_injection_works(tmp_path, settings):
    c = SharedCluster.build(
        settings=replace(settings, vector_dir=str(tmp_path / "vs")),
        product_provider=lambda name: {"found": True, "name": name, "price": 199},
        setup_log=False,
    )
    out = run(c.tools.call("product_query", {"name": "A 耳机"}))
    assert out["result"] == {"found": True, "name": "A 耳机", "price": 199}


def test_notifier_not_injected_marks_undelivered(cluster):
    """不假装通知发出去了 —— 这是"故障可见"的基本要求。"""
    out = run(cluster.tools.call("notify", {"message": "测试"}))
    assert out["result"]["delivered"] is False
    assert "未注入" in out["result"]["reason"]


def test_notifier_injection_works(tmp_path, settings):
    sent: list[tuple[str, str]] = []

    def fake_notifier(message: str, channel: str) -> dict:
        sent.append((message, channel))
        return {"message_id": "m1"}

    c = SharedCluster.build(
        settings=replace(settings, vector_dir=str(tmp_path / "vs")),
        notifier=fake_notifier,
        setup_log=False,
    )
    out = run(c.tools.call("notify", {"message": "库存告警", "channel": "feishu"}))
    assert out["result"]["delivered"] is True
    assert out["result"]["message_id"] == "m1"
    assert sent == [("库存告警", "feishu")]


def test_async_notifier_is_awaited(tmp_path, settings):
    async def fake_async_notifier(message: str, channel: str) -> dict:
        await asyncio.sleep(0)
        return {"async": True}

    c = SharedCluster.build(
        settings=replace(settings, vector_dir=str(tmp_path / "vs")),
        notifier=fake_async_notifier,
        setup_log=False,
    )
    out = run(c.tools.call("notify", {"message": "x"}))
    assert out["result"]["async"] is True


def test_register_tool_extends_cluster(cluster):
    cluster.register_tool(
        "custom_tool",
        "业务自定义工具",
        {"type": "object", "properties": {}, "required": []},
        lambda: {"custom": True},
        owner="presale",
    )
    assert cluster.tools.has("custom_tool")
    out = run(cluster.tools.call("custom_tool", {}))
    assert out["result"] == {"custom": True}


def test_memory_file_store_survives_rebuild(tmp_path, settings):
    path = str(tmp_path / "mem.json")
    local = replace(settings, vector_dir=str(tmp_path / "vs"))
    c1 = SharedCluster.build(settings=local, memory_store=JsonFileStore(path), setup_log=False, env_file=None)
    c1.longterm_memory.remember("主营站点", "北美站", scope="u9")

    c2 = SharedCluster.build_from_memory_file(path, settings=local, setup_log=False)
    assert c2.longterm_memory.recall("站点", scope="u9")[0].value == "北美站"


def test_usage_accumulates_from_gateway(cluster):
    run(cluster.gateway.complete("系统", "分析这个竞品"))
    snapshot = cluster.usage.snapshot()
    assert snapshot["calls"] >= 1
    assert snapshot["total_tokens"] > 0
