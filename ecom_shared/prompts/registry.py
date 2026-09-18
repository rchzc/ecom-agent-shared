"""Prompt 注册中心：把 Prompt 当资产管，而不是散落在各 Agent 文件里的字符串。

**为什么值得单独做一个模块：**

Prompt 是 AI 应用里改得最频繁的东西 —— 业务方说"回答太啰嗦"，你要改的是
Prompt 不是代码。如果它散在 8 个 Agent 文件里，每次调整都要：
定位 → 改字符串 → 跑测试 → 发版。而集中注册后，至少能做到：

1. **一处可查** —— 前端可以直接拉出"当前系统里有哪些 Prompt"，方便非技术同事审阅。
2. **版本可追溯** —— 每条 Prompt 带 version，改了就知道改了哪版。
3. **渲染与定义分离** —— 模板里写占位符，调用方传参，避免手工 f-string 拼出
   花括号转义错误（Prompt 里经常要输出 JSON 示例，f-string 拼 JSON 是灾难）。
4. **可覆写** —— 业务仓可以注册同名 Prompt 覆盖共享包的默认值，
   不用 fork 共享包。这是"共享包能长期存活"的关键：
   如果业务方改不了你的默认 Prompt，他们迟早会把整个包复制走。

**渲染用 `str.format_map` + 缺失键占位策略**，而不是 `str.format`：
format 遇到未提供的键会直接 KeyError，而 Prompt 里的花括号经常是**字面量**
（JSON 输出示例 `{"action": "..."}`），format 会把它当成占位符炸掉。
所以模板里用 `{{` `}}` 转义，渲染时用 format_map 并容忍缺失键返回原样。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

logger = logging.getLogger(__name__)


class _SafeDict(dict):
    """缺失键返回 {key} 原文而不是抛 KeyError。

    这样"Prompt 里少传了一个变量"表现为模型看到 {key} 字面量（一眼能发现），
    而不是整条链路 500（排查要翻日志）。可观测性上后者更差。
    """

    def __missing__(self, key: str) -> str:
        logger.warning("prompt.missing_variable", extra={"variable": key})
        return "{" + key + "}"


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: str
    description: str = ""
    version: str = "1.0"
    #: 声明的变量名，用于调用前校验与前端展示
    variables: tuple[str, ...] = ()
    #: 归属模块（哪个 Agent / 哪条链路），前端按这个分组展示
    owner: str = "shared"

    def render(self, **kwargs: object) -> str:
        return self.template.format_map(_SafeDict(kwargs))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "variables": list(self.variables),
            "owner": self.owner,
            "chars": len(self.template),
        }


class PromptRegistry:
    """全局 Prompt 注册表。进程内单例式使用（各 Agent 共享同一个实例）。"""

    def __init__(self) -> None:
        self._templates: dict[str, PromptTemplate] = {}

    def register(self, template: PromptTemplate, *, override: bool = False) -> None:
        """注册模板。同名默认报错，显式 override=True 才覆盖。

        不让同名静默覆盖是有原因的：两个 Agent 各注册了一个 `answer`，
        后注册的悄悄把前面的顶掉，表现是"改 A 的结果 B 变了"，极难排查。
        """
        if template.name in self._templates and not override:
            raise ValueError(
                f"Prompt {template.name!r} 已存在（owner="
                f"{self._templates[template.name].owner}）。"
                f"如需覆盖请显式传 override=True。"
            )
        self._templates[template.name] = template

    def register_many(self, templates: Iterable[PromptTemplate], *, override: bool = False) -> None:
        for t in templates:
            self.register(t, override=override)

    def get(self, template_name: str) -> PromptTemplate:
        try:
            return self._templates[template_name]
        except KeyError as exc:
            raise KeyError(
                f"未注册的 Prompt {template_name!r}，已注册：{sorted(self._templates)}"
            ) from exc

    def render(self, template_name: str, **kwargs: object) -> str:
        """按名渲染。

        第一个参数叫 `template_name` 而不是 `name` 是有实际原因的：
        Prompt 变量里 `name` 太常见了（"你好 {name}"），
        如果这里叫 `name`，`render("greeting", name="张三")` 会直接抛
        `TypeError: got multiple values for argument 'name'` —— 一个完全配得上
        "为什么这个 API 这么难用"的坑。参数名和最常见的变量名必须错开。
        """
        return self.get(template_name).render(**kwargs)

    def has(self, template_name: str) -> bool:
        return template_name in self._templates

    def names(self) -> list[str]:
        return sorted(self._templates)

    def by_owner(self, owner: str) -> list[PromptTemplate]:
        return [t for t in self._templates.values() if t.owner == owner]

    def describe(self) -> list[dict]:
        """给前端 / 文档用的清单，不含模板正文（正文可能有很长，不适合直接塞响应）。"""
        return [t.to_dict() for t in sorted(self._templates.values(), key=lambda x: x.name)]

    def dump(self) -> dict[str, str]:
        """导出全部模板正文。用于备份与人工审阅。"""
        return {t.name: t.template for t in self._templates.values()}


# ---------------------------------------------------------------------------
# 共享层内置 Prompt：三个跨业务通用的底座能力
# ---------------------------------------------------------------------------
BUILTIN_PROMPTS: tuple[PromptTemplate, ...] = (
    PromptTemplate(
        name="intent_route",
        owner="shared",
        version="1.2",
        description="把用户输入路由到业务域，供 Agent 决定检索哪个知识域",
        variables=("domains", "question", "history"),
        template=(
            "你是跨境电商运营助手的意图路由器。\n"
            "可选业务域：{domains}\n\n"
            "历史对话（可能为空）：\n{history}\n\n"
            "当前用户问题：{question}\n\n"
            "请判断该问题属于哪个业务域，并抽取用于检索的关键词。\n"
            "只输出 JSON，不要任何解释文字：\n"
            '{{"domain": "业务域名称", "keywords": ["关键词1", "关键词2"], '
            '"confidence": 0.0, "reason": "一句话理由"}}'
        ),
    ),
    PromptTemplate(
        name="rag_answer",
        owner="shared",
        version="1.3",
        description="基于检索片段作答，要求标注来源、无依据时明确说不知道",
        variables=("question", "context", "persona"),
        template=(
            "你是{persona}。\n\n"
            "以下是检索到的知识库片段：\n{context}\n\n"
            "用户问题：{question}\n\n"
            "回答要求：\n"
            "1. 只使用上面片段中的信息，不要凭常识补充具体数字或规则；\n"
            "2. 如果片段不足以回答，明确说「知识库中没有相关内容」，不要编造；\n"
            "3. 在引用具体做法时，用 [来源: 文件名] 标注它出自哪个片段。\n"
            "只输出 JSON：\n"
            '{{"answer": "回答正文", "used_sources": ["文件名"], '
            '"confidence": 0.0, "need_human": false}}'
        ),
    ),
    PromptTemplate(
        name="memory_extract",
        owner="shared",
        version="1.1",
        description="从对话里抽取值得长期记住的客户事实",
        variables=("transcript",),
        template=(
            "从下面的对话中抽取「值得长期记住的客户事实」。\n\n"
            "只抽稳定的偏好与约束（主营站点、价格敏感度、不接受的条件、"
            "长期合作要求），不要抽一次性的问题（如「这次想买什么」）。\n\n"
            "对话：\n{transcript}\n\n"
            "只输出 JSON：\n"
            '{{"facts": [{{"key": "事实名", "value": "事实内容", "tags": ["标签"]}}]}}\n'
            "没有值得记住的内容时返回 {{\"facts\": []}}。"
        ),
    ),
)


def build_default_registry() -> PromptRegistry:
    """建一个装好内置 Prompt 的注册表。业务仓可继续往上注册自己的模板。"""
    registry = PromptRegistry()
    registry.register_many(BUILTIN_PROMPTS)
    return registry


if __name__ == "__main__":
    reg = build_default_registry()
    out = reg.render(
        "rag_answer",
        persona="跨境电商售前顾问",
        question="ACOS 过高怎么办",
        context="[1] 广告 ACOS 优化：先看搜索词报告，否掉高花费零转化词",
    )
    # 缺失变量不抛异常，而是留 {变量} 字面量
    partial = reg.render("rag_answer", persona="顾问")
    assert "{question}" in partial and "{context}" in partial
    print("PromptRegistry 自测通过，内置模板：", reg.names())
    print(out.splitlines()[-1])
