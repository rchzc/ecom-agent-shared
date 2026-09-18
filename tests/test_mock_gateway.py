"""离线网关单测。

它验证的是"管道通不通"，不是"答得好不好" —— mock 的意义就在于此：
CI 里没有任何 API Key 也能把整条编排链路跑一遍。
"""
from __future__ import annotations

import asyncio
import json

import pytest

from ecom_shared.gateway.llm import LLMGateway
from ecom_shared.gateway.mock import MOCK_TAG, MockLLMGateway, get_gateway
from ecom_shared.gateway.router import parse_json_lenient

PROMPT_WITH_SCHEMA = (
    "你是运营分析师。\n"
    "输出格式示例：\n"
    '{"summary": "一句话结论", "score": 0, "actions": ["动作"], "urgent": false}\n'
    "请生成分析。"
)


def run(coro):
    return asyncio.run(coro)


def test_factory_returns_mock_gateway_when_provider_is_mock(settings):
    gw = get_gateway(settings)
    assert isinstance(gw, MockLLMGateway)
    assert settings.is_mock


def test_factory_returns_real_gateway_otherwise(settings):
    from dataclasses import replace

    gw = get_gateway(replace(settings, provider="ollama", api_key="x", api_base="http://localhost:1"))
    assert isinstance(gw, LLMGateway)


def test_mock_output_is_parseable_json_echoing_prompt_schema(settings):
    """mock 会从 prompt 的 JSON 示例反推结构 —— 这样下游的解析、校验、
    前端渲染全都能真跑，只是内容没有语义价值。"""
    gw = MockLLMGateway(settings)
    content, meta = run(gw.complete("系统提示", PROMPT_WITH_SCHEMA))
    parsed = parse_json_lenient(content)
    assert set(parsed) == {"summary", "score", "actions", "urgent"}
    assert parsed["urgent"] is False          # 保持类型
    assert isinstance(parsed["score"], int)
    assert meta["mock"] is True


def test_mock_marks_placeholder_content(settings):
    """占位内容带 [MOCK] 前缀，演示时一眼能看出哪段不是真模型产的。"""
    gw = MockLLMGateway(settings)
    content, _ = run(gw.complete("s", PROMPT_WITH_SCHEMA))
    assert MOCK_TAG in content


def test_mock_emits_tool_decision_when_tools_declared(settings):
    """Agent 循环第一轮：system 声明了可用工具，应该先发起检索而不是直接答。"""
    gw = MockLLMGateway(settings)
    content, _ = run(gw.complete("你可以使用以下可用工具：rag_search", "ACOS 怎么优化"))
    decision = parse_json_lenient(content)
    assert decision["action"] == "tool"
    assert decision["tool"] == "rag_search"
    assert decision["args"]["query"]


def test_mock_emits_answer_after_tool_result(settings):
    """工具结果回灌后该收尾，不能又发起一次工具调用（会死循环）。"""
    gw = MockLLMGateway(settings)
    content, _ = run(gw.complete("s", "工具 rag_search 返回：[1] ACOS 优化建议……"))
    decision = parse_json_lenient(content)
    assert decision["action"] == "answer"
    assert decision["content"]


# -- 工具报错时重试：这条决定了"反复撞同一个工具报错"能不能被造出来 ---------------
# 循环的错误预算（同一个工具失败到上限就禁用）只有在**真有第二次失败**时才会触发，
# 所以 mock 必须区分"工具成功返回"和"工具报错"，否则自进化演示永远造不出失败样本。

_ERROR_OBSERVATION = '工具 rag_search 返回：{"ok": false, "error": "知识库服务不可用"}。请继续。'


def test_mock_retries_tool_when_observation_is_error(settings):
    gw = MockLLMGateway(settings)
    content, _ = run(gw.complete("你可以使用以下可用工具：rag_search", _ERROR_OBSERVATION))
    decision = parse_json_lenient(content)
    assert decision["action"] == "tool", "工具报错时不该拿失败记录当依据作答"
    assert decision["tool"] == "rag_search"


def test_mock_answers_on_error_once_tools_are_forbidden(settings):
    """工具被禁用后必须收尾 —— 否则重试会变成新的死循环。"""
    gw = MockLLMGateway(settings)
    forbidden = "本轮不允许再调用任何工具，请直接基于已知信息作答。\n只输出 JSON 示例："
    content, _ = run(gw.complete(forbidden, _ERROR_OBSERVATION))
    decision = parse_json_lenient(content)
    assert decision["action"] != "tool"


def test_mock_answers_on_successful_result_even_with_tools_declared(settings):
    """反例：成功的结果不该触发重试，否则正常流程也会撞迭代上限。"""
    gw = MockLLMGateway(settings)
    content, _ = run(
        gw.complete("你可以使用以下可用工具：rag_search", "工具 rag_search 返回：3 条命中。请继续。")
    )
    assert parse_json_lenient(content)["action"] == "answer"


def test_mock_template_registration_wins(settings):
    gw = MockLLMGateway(settings)
    gw.register_template("评分", {"score": 88, "level": "A"})
    content, _ = run(gw.complete("请给这段对话评分", "对话内容"))
    assert parse_json_lenient(content) == {"score": 88, "level": "A"}


def test_mock_stream_yields_multiple_chunks(settings):
    async def collect():
        gw = MockLLMGateway(settings)
        return [c async for c in gw.stream("s", PROMPT_WITH_SCHEMA)]

    chunks = run(collect())
    assert len(chunks) > 1
    assert MOCK_TAG in "".join(chunks)


def test_mock_records_usage_and_routing(settings):
    gw = MockLLMGateway(settings)
    run(gw.complete("系统", "帮我分析竞品的定价策略并给出方案"))
    run(gw.complete("系统", "判断是否"))
    snapshot = gw.usage.snapshot()
    assert snapshot["calls"] == 2
    assert set(snapshot["by_tier"]) <= {"light", "heavy"}
    assert snapshot["total_tokens"] > 0


def test_mock_route_can_be_forced(settings):
    gw = MockLLMGateway(settings)
    _, meta = run(gw.complete("s", "随便", force_tier="heavy"))
    assert meta["tier"] == "heavy"
    assert meta["model"] == settings.model_heavy


def test_mock_returns_plain_text_when_json_mode_off(settings):
    gw = MockLLMGateway(settings)
    content, _ = run(gw.complete("s", "你好，介绍一下你自己", json_mode=False))
    with pytest.raises(Exception):
        json.loads(content)  # 确实不是 JSON
    assert MOCK_TAG in content
