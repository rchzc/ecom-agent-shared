"""MCP 工具协议单测：注册、调用、JSON-RPC 分帧、失败语义。

最关键的断言是 **工具失败不能抛异常** —— Agent 循环靠这个把失败当成
"这次尝试没成功"继续往下走，而不是整轮对话中断。
"""
from __future__ import annotations

import asyncio

import pytest

from ecom_shared.mcp.registry import ToolRegistry


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def reg():
    registry = ToolRegistry(server_name="test")

    def add(a: int, b: int = 1) -> dict:
        return {"sum": a + b}

    async def slow(x: str) -> dict:
        await asyncio.sleep(0)
        return {"echo": x}

    def boom() -> dict:
        raise RuntimeError("内部炸了")

    registry.register(
        "add",
        "求和",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}, "required": ["a"]},
        add,
    )
    registry.register(
        "async_echo",
        "异步回显",
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        slow,
    )
    registry.register(
        "boom",
        "会抛异常的工具",
        {"type": "object", "properties": {}, "required": []},
        boom,
    )
    return registry


# ---------------------------------------------------------------------------
# 注册
# ---------------------------------------------------------------------------
def test_register_and_list(reg):
    assert reg.names() == ["add", "async_echo", "boom"]
    listed = {t["name"]: t for t in reg.list_tools()}
    assert "inputSchema" in listed["add"]  # 字段名对齐 MCP 协议


def test_duplicate_tool_registration_requires_override(reg):
    with pytest.raises(ValueError, match="已注册"):
        reg.register("add", "另一个", {}, lambda: {})


def test_invalid_tool_name_rejected(reg):
    with pytest.raises(ValueError, match="不合法"):
        reg.register("bad name!", "x", {}, lambda: {})


# ---------------------------------------------------------------------------
# 调用
# ---------------------------------------------------------------------------
def test_sync_tool_call(reg):
    out = run(reg.call("add", {"a": 1, "b": 2}))
    assert out["ok"] and out["result"] == {"sum": 3}


def test_async_tool_call(reg):
    out = run(reg.call("async_echo", {"x": "hi"}))
    assert out["result"] == {"echo": "hi"}


def test_unknown_tool_returns_error_not_exception(reg):
    out = run(reg.call("nope", {}))
    assert out["ok"] is False
    assert "unknown tool" in out["error"]
    assert "add" in out["available"]


def test_missing_required_argument_reported(reg):
    out = run(reg.call("add", {}))
    assert out["ok"] is False
    assert "a" in out["error"]


def test_unexpected_argument_returns_accepted_names(reg):
    """模型臆造参数名时，错误信息里带上正确参数名，它下一轮才能自我纠正。"""
    out = run(reg.call("add", {"a": 1, "c": 9}))
    assert out["ok"] is False
    assert "参数不匹配" in out["error"]


def test_tool_exception_is_swallowed_into_error_result(reg):
    """这是整个模块最重要的一条：工具炸了不能把 Agent 循环带崩。"""
    out = run(reg.call("boom", {}))
    assert out["ok"] is False
    assert "内部炸了" in out["error"]


def test_call_history_is_recorded(reg):
    run(reg.call("add", {"a": 1}))
    run(reg.call("boom", {}))
    history = reg.history()
    assert len(history) == 2
    assert history[0]["name"] == "boom"  # 最新的在前
    assert history[0]["ok"] is False


# ---------------------------------------------------------------------------
# JSON-RPC 分帧
# ---------------------------------------------------------------------------
def test_initialize_returns_protocol_version(reg):
    resp = run(reg.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
    assert resp["result"]["protocolVersion"]
    assert resp["result"]["capabilities"]["tools"] is not None


def test_tools_list_matches_mcp_shape(reg):
    resp = run(reg.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
    assert {t["name"] for t in resp["result"]["tools"]} == {"add", "async_echo", "boom"}


def test_tools_call_success_content(reg):
    resp = run(reg.handle({
        "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": {"name": "add", "arguments": {"a": 2, "b": 3}},
    }))
    assert resp["result"]["isError"] is False
    assert '"sum": 5' in resp["result"]["content"][0]["text"]


def test_tools_call_failure_still_returns_successful_rpc(reg):
    """按 MCP 约定：工具执行失败是"成功的 RPC + isError=true"，
    这样客户端能把失败原因交给模型，而不是让整次 RPC 挂掉。"""
    resp = run(reg.handle({
        "jsonrpc": "2.0", "id": 4, "method": "tools/call",
        "params": {"name": "boom", "arguments": {}},
    }))
    assert "error" not in resp
    assert resp["result"]["isError"] is True
    assert "内部炸了" in resp["result"]["content"][0]["text"]


def test_unknown_method_returns_minus_32601(reg):
    resp = run(reg.handle({"jsonrpc": "2.0", "id": 5, "method": "resources/list"}))
    assert resp["error"]["code"] == -32601


def test_bad_jsonrpc_version_rejected(reg):
    resp = run(reg.handle({"jsonrpc": "1.0", "id": 6, "method": "ping"}))
    assert resp["error"]["code"] == -32600


def test_ping(reg):
    assert run(reg.handle({"jsonrpc": "2.0", "id": 7, "method": "ping"}))["result"] == {}
