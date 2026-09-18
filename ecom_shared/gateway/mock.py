"""Mock Gateway：完全离线、确定性的模型替身。

**这不是"没有 key 时的降级"，是一个显式 provider。**
区别很重要：隐式降级会让人以为 key 配好了其实没生效；provider=mock 是明确声明
"我就是要离线跑"。CI、本地演示、只验证编排逻辑时用它。

它的工作原理不是随机吐字符串，而是**从 prompt 里反推期望的结构**：

1. 显式模板优先 —— 调用方用 `register_template("关键词", {...})` 注册精确返回。
2. 否则扫描 prompt 里的 JSON 示例（绝大多数 Agent 的 prompt 都会带一个输出格式示例），
   把示例里的值替换成确定性占位符，键结构原样保留。
   这样返回的 JSON **结构与真实调用一致**，下游的解析、校验、前端渲染全都能真跑，
   只是内容没有语义价值。
3. 都不匹配则返回纯文本占位。

所以它验证的是"管道通不通"，不是"模型答得好不好" —— README 里对这点写得很直白。
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator

from ..config import Settings
from .llm import LLMGateway, UsageStats
from .router import classify_complexity

_JSON_BLOCK_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

#: 占位文本会带上这个前缀，方便在演示时一眼看出哪段是 mock 产出
MOCK_TAG = "[MOCK]"


def _placeholder(value: Any, key: str = "") -> Any:
    """按值的类型生成确定性占位。保持类型不变，下游的类型校验才不会误报。"""
    if isinstance(value, str):
        return f"{MOCK_TAG} {key or '内容'}（离线占位，填入 API Key 后由真实模型生成）"
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return 0
    if isinstance(value, list):
        # 空数组无法验证"多条输出"的渲染逻辑，给一条结构一致的样本
        if not value:
            return []
        return [_placeholder(value[0], key)]
    if isinstance(value, dict):
        return {k: _placeholder(v, k) for k, v in value.items()}
    if value is None:
        return None
    return value


def _find_schema_example(prompt: str) -> dict | None:
    """从 prompt 里扫出第一个可解析的 JSON 对象，当作输出结构模板。

    只认顶层没有嵌套花括号的片段，避免把整段 prompt 误当 JSON。
    从后往前扫：Agent 的 prompt 通常把"输出格式"放在最后。
    """
    matches = _JSON_BLOCK_RE.findall(prompt)
    for raw in reversed(matches):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and parsed:
            return parsed
    return None


class MockLLMGateway:
    """与 LLMGateway 接口完全一致的离线替身。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.usage = UsageStats()
        self._templates: list[tuple[str, Any]] = []

    # -- 与真实网关一致的接口 ------------------------------------------------
    def resolve_route(
        self, task_text: str, force: str | None = None
    ) -> tuple[str, str, int]:
        if force in ("light", "heavy"):
            return (
                self.settings.model_light if force == "light" else self.settings.model_heavy,
                force,
                0,
            )
        tier, score = classify_complexity(task_text)
        model = (
            self.settings.model_light if tier == "light" else self.settings.model_heavy
        )
        return model, tier, score

    def resolve_model(self, task_text: str, force: str | None = None) -> str:
        return self.resolve_route(task_text, force)[0]

    async def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        temperature: float = 0.3,
        force_tier: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        model, tier, score = self.resolve_route(f"{system}\n{user}", force_tier)
        content = self._render(system, user, json_mode)
        # 用字符数粗估 token，让成本看板在离线模式下也有数据可展示
        self.usage.record(model, len(system + user) // 2, len(content) // 2, tier)
        return content, {
            "model": model,
            "tier": tier,
            "route_score": score,
            "prompt_tokens": len(system + user) // 2,
            "completion_tokens": len(content) // 2,
            "mock": True,
        }

    async def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        force_tier: str | None = None,
    ) -> AsyncIterator[str]:
        """分块吐出，模拟打字机效果。分块粒度按 16 字符，接近真实厂商的 chunk 大小。"""
        content = self._render(system, user, json_mode=True)
        for i in range(0, len(content), 16):
            yield content[i : i + 16]
            await asyncio.sleep(0)  # 让出事件循环，行为与真流式一致

    # -- mock 专有 -----------------------------------------------------------
    def register_template(self, keyword: str, response: Any) -> None:
        """注册精确返回。keyword 命中 system 或 user 任一即可。

        单测和演示脚本用它对特定 Agent 定点造数据，比依赖启发式更可控。
        """
        self._templates.append((keyword, response))

    def _render(self, system: str, user: str, json_mode: bool) -> str:
        blob = f"{system}\n{user}"

        # 1) 显式模板
        for keyword, response in self._templates:
            if keyword in blob:
                if isinstance(response, str):
                    return response
                return json.dumps(response, ensure_ascii=False)

        # 2) 工具结果已回灌 → 该收尾给答案了。
        #    **这一条必须排在"声明了可用工具"之前**：Agent 循环的 system prompt
        #    里始终带着工具清单，如果先判「可用工具」就每轮都返回工具调用，
        #    永远拿不到 action=answer，循环必然撞上迭代上限。
        #    （真模型不会犯这个错 —— 它读得懂上下文；mock 是启发式，顺序即语义。）
        if "工具" in user and "返回" in user:
            # 2a) 但**报错的**工具结果不该直接收尾：真模型看到 ok=false 会重试，
            #     不会拿一条失败记录当依据作答。不区分成功/失败的话，循环的错误预算
            #     永远等不到第二次失败，"反复撞同一个工具报错"这类失败模式就造不出来 ——
            #     而它恰恰是"光靠改 Prompt 修不掉、必须动编排策略"的典型案例。
            #     仅当工具仍可用时才重试；被禁用后必须收尾，否则又会死循环。
            if _looks_like_tool_error(user) and "可用工具" in system:
                return json.dumps(
                    {
                        "action": "tool",
                        "tool": "rag_search",
                        "args": {"query": _extract_question(user), "top_k": 3},
                        "thought": f"{MOCK_TAG} 上一次调用失败，换个参数再试一次",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "action": "answer",
                    "content": f"{MOCK_TAG} 已根据检索到的知识库片段生成解答。",
                },
                ensure_ascii=False,
            )
        # 3) Agent 循环的工具决策：system 里声明了可用工具时，先发起一次检索
        if "可用工具" in system or "可用工具" in user:
            return json.dumps(
                {
                    "action": "tool",
                    "tool": "rag_search",
                    "args": {"query": _extract_question(user), "top_k": 3},
                    "thought": f"{MOCK_TAG} 先检索知识库再回答",
                },
                ensure_ascii=False,
            )

        # 4) 从 prompt 的 JSON 示例反推结构
        example = _find_schema_example(blob)
        if example is not None:
            return json.dumps(_placeholder(example), ensure_ascii=False)

        # 5) 纯文本兜底
        if not json_mode:
            return f"{MOCK_TAG} 已收到：「{_extract_question(user)[:40]}」"
        return json.dumps({"content": f"{MOCK_TAG} 离线占位回答"}, ensure_ascii=False)


def _extract_question(user: str) -> str:
    """从 user prompt 里剥出真正的用户问题，别把整段模板当成问题。"""
    for marker in ("用户问题：", "用户问题:", "【用户输入】", "问题："):
        if marker in user:
            return user.split(marker, 1)[1].strip()[:120]
    return user.strip()[:120]


#: 工具回灌结果里代表失败的标记。循环把 ToolResult 序列化成 JSON 再回灌，
#: 所以这里按 JSON 键值判，而不是去匹配某句中文错误文案 —— 文案会变，键不会。
_TOOL_ERROR_MARKS = ('"ok": false', '"ok":false', '"isError": true', '"isError":true')


def _looks_like_tool_error(text: str) -> bool:
    """回灌的工具观察值是不是一条失败记录。"""
    return any(mark in text for mark in _TOOL_ERROR_MARKS)


def get_gateway(settings: Settings):
    """工厂：按 provider 选真实网关还是离线替身。

    调用方只需要 `gw = get_gateway(settings)`，不需要知道当前是哪种 ——
    这正是"统一接入"的价值，也是把 mock 做成 provider 而不是 if-else 散落各处的理由。
    """
    if settings.is_mock:
        return MockLLMGateway(settings)
    return LLMGateway(settings)
