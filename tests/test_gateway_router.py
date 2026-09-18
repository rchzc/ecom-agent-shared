"""模型路由与输出解析的单测。

这两个是纯函数，所以能测得很彻底 —— 这本身就是把它们从 Gateway 类里拆出来的理由：
如果它们长在类里，测试就得先造一个 AsyncOpenAI 客户端。
"""
from __future__ import annotations

import pytest

from ecom_shared.errors import ModelOutputError
from ecom_shared.gateway.router import (
    classify_complexity,
    is_unsupported_json_mode,
    parse_json_array_lenient,
    parse_json_lenient,
)


# ---------------------------------------------------------------------------
# 复杂度路由
# ---------------------------------------------------------------------------
def test_heavy_keywords_route_to_heavy():
    tier, score = classify_complexity("帮我分析一下这个竞品的定价策略")
    assert tier == "heavy"
    assert score > 0


def test_light_keywords_route_to_light():
    tier, score = classify_complexity("判断这句话是不是退款诉求")
    assert tier == "light"
    assert score <= 0


def test_long_text_alone_can_trigger_heavy():
    # 没有重量关键词，但文本很长 —— 长文本通常上下文复杂，值得给重模型
    tier, _ = classify_complexity("嗯" * 250)
    assert tier == "heavy"


def test_short_plain_text_stays_light():
    tier, _ = classify_complexity("你好")
    assert tier == "light"


# ---------------------------------------------------------------------------
# 三级 JSON 容错解析
# ---------------------------------------------------------------------------
def test_parse_level1_direct():
    assert parse_json_lenient('{"a": 1}') == {"a": 1}


def test_parse_level2_markdown_fence():
    raw = '```json\n{"summary": "ok", "score": 3}\n```'
    assert parse_json_lenient(raw) == {"summary": "ok", "score": 3}


def test_parse_level3_surrounded_by_prose():
    raw = '好的，这是我的分析结果：{"answer": "降低竞价", "confidence": 0.8} 希望有帮助。'
    assert parse_json_lenient(raw)["answer"] == "降低竞价"


def test_parse_handles_braces_inside_strings():
    """字符串里出现花括号时不能截错 —— 这是用正则做第三级解析会踩的坑。"""
    raw = '前言 {"template": "话术：您好{客户名}，为您推荐{商品}", "ok": true} 结语'
    parsed = parse_json_lenient(raw)
    assert parsed["ok"] is True
    assert "{客户名}" in parsed["template"]


def test_parse_raises_on_garbage():
    with pytest.raises(ModelOutputError):
        parse_json_lenient("这完全不是 JSON")


def test_parse_raises_on_empty():
    with pytest.raises(ModelOutputError):
        parse_json_lenient("")


def test_parse_array_unwraps_dict_container():
    assert parse_json_array_lenient('{"items": [1, 2]}') == [1, 2]


# ---------------------------------------------------------------------------
# JSON 模式降级判定
# ---------------------------------------------------------------------------
class _FakeErr(Exception):
    def __init__(self, msg: str, status: int | None = None) -> None:
        super().__init__(msg)
        if status is not None:
            self.status_code = status


def test_unsupported_json_mode_detected_for_400():
    assert is_unsupported_json_mode(_FakeErr("Unsupported parameter: response_format", 400))


def test_auth_error_is_not_treated_as_unsupported():
    """401 不该被当成"不支持 JSON 模式"去重试 —— 重试注定再失败一次，
    还会把真正的鉴权错误伪装成降级，排查会被带偏。"""
    assert not is_unsupported_json_mode(_FakeErr("Invalid API key", 401))


def test_rate_limit_is_not_treated_as_unsupported():
    assert not is_unsupported_json_mode(_FakeErr("rate limit exceeded", 429))


def test_timeout_without_status_is_not_unsupported():
    assert not is_unsupported_json_mode(TimeoutError("connection timed out"))
