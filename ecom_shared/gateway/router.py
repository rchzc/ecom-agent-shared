"""模型路由 + 输出解析。

这两件事放在一起，是因为它们解决的是同一个问题的两面：
**大模型的输出不可信**。

- 路由：不是所有任务都值得调大模型。分类、提取、判断这类任务用轻量模型就够，
  简单任务走重量模型是纯粹的浪费。
- 解析：模型返回 JSON 时经常带 Markdown 围栏、前后加解释文字、或者干脆吐
  Python 风格的单引号字典。没有容错解析，前端迟早会拿到脏数据崩掉。

两者都是纯函数，不依赖网络和 SDK，因此可以单独写单测 —— 这是把它们从
Gateway 类里拆出来的主要动机。
"""
from __future__ import annotations

import json
import re
from typing import Any

from ..config import COMPLEX_LENGTH_THRESHOLD, HEAVY_HINTS, LIGHT_HINTS
from ..errors import ModelOutputError

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

# 厂商不支持 response_format / json_object 时的错误特征
_UNSUPPORTED_MARKERS = (
    "response_format",
    "json_object",
    "json mode",
    "json_mode",
    "unsupported",
    "not supported",
    "unknown parameter",
    "invalid_request",
    "unrecognized request argument",
)


# ---------------------------------------------------------------------------
# 模型路由
# ---------------------------------------------------------------------------
def classify_complexity(text: str) -> tuple[str, int]:
    """按关键词加权 + 文本长度判定任务复杂度，返回 (档位, 得分)。

    得分 > 0 走重量模型。诚实说明：这是**规则路由**，不是训练出来的分类器，
    准确率完全依赖关键词表，换个领域就要重调词表。

    要更准可以先用一个小模型做一次分类，代价是多一次调用（多一次延迟 + 一点 token），
    在"省下的重模型额度"和"多花的分类调用"之间要算账。当前规模下规则足够，
    因为业务是固定的 6 个域，词表可控。

    之所以返回得分而不是只返回档位：前端成本看板要把得分展示出来，
    不然用户看到"走了轻量模型"却不知道为什么，没法判断要不要强制切重模型。
    """
    score = 0
    for word in HEAVY_HINTS:
        if word in text:
            score += 2
    for word in LIGHT_HINTS:
        if word in text:
            score -= 1
    if len(text) > COMPLEX_LENGTH_THRESHOLD:
        score += 1
    return ("heavy" if score > 0 else "light"), score


def is_unsupported_json_mode(exc: Exception) -> bool:
    """判断异常是否来自「厂商不支持强制 JSON 模式」这类请求参数问题。

    只有这类错误才值得去掉 response_format 重试一次。

    之前这里捕获所有异常都重试：401 鉴权失败、429 限流、网络超时
    也会被当成"不支持 JSON 模式"，结果是多花一次注定失败的调用，
    还把真正的错误伪装成降级，排查时被误导到完全错误的方向。
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            if int(status) not in (400, 404, 415, 422):
                return False
        except (TypeError, ValueError):
            return False
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _UNSUPPORTED_MARKERS)


# ---------------------------------------------------------------------------
# 三级 JSON 容错解析
# ---------------------------------------------------------------------------
def parse_json_lenient(raw: str) -> dict[str, Any]:
    """三级容错解析模型输出。

    第一级：直接解析。
    第二级：去掉 Markdown 代码块围栏再解析（模型经常包一层 ```json）。
    第三级：截取文本中第一个完整的 {...} 再解析（模型常在 JSON 前后加解释文字）。

    三级都失败就抛 ModelOutputError，由上层转成 502。
    关键原则：宁可报错，也不把脏数据透传给前端。
    """
    text = (raw or "").strip()
    if not text:
        raise ModelOutputError("模型返回内容为空")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    extracted = _extract_first_object(text)
    if extracted is not None:
        try:
            parsed = json.loads(extracted)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ModelOutputError(
        "模型输出无法解析为 JSON 对象（已尝试三级容错）",
        detail=text[:300],
    )


def parse_json_array_lenient(raw: str) -> list[Any]:
    """数组版容错解析。用于「一次生成多条素材」这类输出。"""
    text = (raw or "").strip()
    if not text:
        raise ModelOutputError("模型返回内容为空")
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        # 模型有时把数组包在 {"items": [...]} 里，这里顺手兼容
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    return value
    except json.JSONDecodeError:
        pass
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        try:
            parsed = json.loads(fenced.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ModelOutputError("模型输出无法解析为 JSON 数组", detail=text[:300])


def _extract_first_object(text: str) -> str | None:
    """按括号配对扫描出第一个完整的 {...}，跳过字符串内的花括号。

    不用正则的原因：JSON 字符串里完全可能出现 `{` 和 `}`（比如生成的话术里
    带占位符模板），正则会被这种内容骗到，截出一个残缺的对象。
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        start = text.find("{", start + 1)
    return None
