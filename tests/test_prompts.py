"""Prompt 注册中心单测。重点是"渲染不能炸"这件事。"""
from __future__ import annotations

import pytest

from ecom_shared.prompts.registry import (
    BUILTIN_PROMPTS,
    PromptRegistry,
    PromptTemplate,
    build_default_registry,
)


def test_render_fills_variables():
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="t", template="你好 {name}", variables=("name",)))
    assert reg.render("t", name="张三") == "你好 张三"


def test_missing_variable_does_not_raise():
    """Prompt 少传一个变量时留下 {key} 字面量，而不是整条链路 500。
    前者一眼能发现，后者要翻日志。"""
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="t", template="你好 {name}，问 {q}"))
    out = reg.render("t", name="张三")
    assert out == "你好 张三，问 {q}"


def test_json_braces_in_template_survive():
    """Prompt 里经常要写 JSON 输出示例。用 str.format 会被花括号炸掉，
    所以模板里用 {{ }} 转义，渲染后必须还原成单个花括号。"""
    reg = PromptRegistry()
    reg.register(
        PromptTemplate(name="t", template='只输出 JSON：{{"a": "{v}"}}')
    )
    assert reg.render("t", v="x") == '只输出 JSON：{"a": "x"}'


def test_duplicate_registration_raises_unless_override():
    """同名静默覆盖会导致"改 A 的结果 B 变了"，必须显式声明意图。"""
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="t", template="v1", owner="a"))
    with pytest.raises(ValueError, match="已存在"):
        reg.register(PromptTemplate(name="t", template="v2", owner="b"))
    reg.register(PromptTemplate(name="t", template="v2", owner="b"), override=True)
    assert reg.render("t") == "v2"


def test_get_unknown_prompt_lists_registered():
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="a", template="x"))
    with pytest.raises(KeyError, match="已注册"):
        reg.get("nope")


def test_builtin_prompts_render_without_error():
    """内置 Prompt 必须能带全部变量渲染通过 —— 否则线上第一次调用就炸。"""
    reg = build_default_registry()
    reg.render("intent_route", domains="selection/ads", question="q", history="")
    reg.render("rag_answer", persona="顾问", question="q", context="ctx")
    reg.render("memory_extract", transcript="客户：太贵了")


def test_describe_hides_template_body():
    """清单接口不该带模板正文：一个响应塞十几个长 Prompt 会很难用。"""
    reg = build_default_registry()
    described = reg.describe()
    assert len(described) == len(BUILTIN_PROMPTS)
    for item in described:
        assert "template" not in item
        assert item["chars"] > 0


def test_by_owner_filters():
    reg = PromptRegistry()
    reg.register(PromptTemplate(name="a", template="x", owner="shared"))
    reg.register(PromptTemplate(name="b", template="x", owner="presale"))
    assert [t.name for t in reg.by_owner("shared")] == ["a"]
