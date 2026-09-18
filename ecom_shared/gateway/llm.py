"""LLM Gateway：所有业务 Agent 与模型之间的唯一出口。

四个能力都在这一个类里，因为它们是同一个诉求的四条边：

1. **多厂商统一接入** —— 所有 provider 走 OpenAI 兼容协议，业务代码不感知厂商差异。
   换厂商只改 .env 里的 LLM_PROVIDER，代码零改动。
2. **模型路由** —— 按任务复杂度在轻/重模型之间选，简单任务不占用大模型额度。
   这是"成本下降 50%–90%"这个数字的来源，不是靠压 token 数，是靠**选对模型**。
3. **输出容错** —— 模型输出不稳定是常态，三级容错解析保证脏数据不会透传到前端。
4. **用量与成本记账** —— 每次调用的 token 和估算成本都记下来，前端的成本看板直接读它。

第 2、3 点是面试最常被追问的地方。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from ..config import Settings
from ..errors import ModelCallError
from .router import classify_complexity, is_unsupported_json_mode

logger = logging.getLogger(__name__)

# 价格表：单位「元 / 千 token」，只用于**估算**展示，不参与任何计费逻辑。
# 真实账单以厂商后台为准 —— 这里必须写清楚是估算，否则会给人"成本数据可信"的错觉。
PRICE_TABLE: dict[str, dict[str, float]] = {
    "qwen-plus": {"in": 0.0008, "out": 0.002},
    "qwen-max": {"in": 0.02, "out": 0.06},
    "deepseek-chat": {"in": 0.001, "out": 0.002},
    "gpt-4o-mini": {"in": 0.0011, "out": 0.0044},
    "gpt-4o": {"in": 0.018, "out": 0.072},
}


@dataclass
class UsageStats:
    """进程内累计的模型用量与估算成本。

    展示用途：前端成本看板。**不持久化** —— 进程重启即清零。
    要做真实成本核算必须落库，见 README「已知边界」一节。
    """

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)
    #: 按轻/重档位分桶的调用次数，用于展示路由效果
    by_tier: dict[str, int] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def estimated_cost(self) -> float:
        """估算总成本（元）。单价缺失的模型按 0 计，不猜。"""
        total = 0.0
        for model, tokens in self.by_model.items():
            price = PRICE_TABLE.get(model)
            if not price:
                continue
            # 只用输入价近似（无法从 by_model 里拆出输入/输出），
            # 所以这个数字是**保守偏低**的估算，不应写进简历当精确指标
            total += tokens / 1000 * price["in"]
        return round(total, 4)

    def record(self, model: str, prompt: int, completion: int, tier: str = "") -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.by_model[model] = self.by_model.get(model, 0) + prompt + completion
        if tier:
            self.by_tier[tier] = self.by_tier.get(tier, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "by_model": dict(self.by_model),
            "by_tier": dict(self.by_tier),
            "estimated_cost_cny": self.estimated_cost,
        }


class LLMGateway:
    """真实厂商接入。business 层只跟它打交道，不直接接触 SDK。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.usage = UsageStats()
        self._client = AsyncOpenAI(
            api_key=settings.api_key or "ollama",
            base_url=settings.api_base,
            timeout=settings.request_timeout,
            max_retries=1,
        )

    # ------------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------------
    def resolve_route(
        self, task_text: str, force: str | None = None
    ) -> tuple[str, str, int]:
        """一次算完路由结果，返回 (model, tier, score)。

        流式链路要把 tier/score 下发给前端做成本看板。若先 classify_complexity
        再 resolve_model，同一套关键词扫描会算两遍，且两处逻辑一旦漂移，
        就会出现"看板显示走轻量、实际调用重量"这种自相矛盾的展示。
        """
        if force in ("light", "heavy"):
            model = (
                self.settings.model_light
                if force == "light"
                else self.settings.model_heavy
            )
            return model, force, 0
        tier, score = classify_complexity(task_text)
        model = (
            self.settings.model_light if tier == "light" else self.settings.model_heavy
        )
        return model, tier, score

    def resolve_model(self, task_text: str, force: str | None = None) -> str:
        """选择模型：force 优先，否则按复杂度路由。"""
        return self.resolve_route(task_text, force)[0]

    # ------------------------------------------------------------------
    # 非流式
    # ------------------------------------------------------------------
    async def complete(
        self,
        system: str,
        user: str,
        *,
        json_mode: bool = True,
        temperature: float = 0.3,
        force_tier: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """非流式调用，返回 (内容, 元信息)。"""
        model, tier, score = self.resolve_route(f"{system}\n{user}", force_tier)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": False,
        }
        if json_mode:
            # 百炼 / OpenAI 支持强制 JSON 模式；DeepSeek 不支持，
            # 所以容错解析必须留着，不能依赖这个参数。
            kwargs["response_format"] = {"type": "json_object"}

        resp = await self._create(kwargs, model, json_mode, "complete")

        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        self.usage.record(model, prompt_tokens, completion_tokens, tier)
        meta = {
            "model": model,
            "tier": tier,
            "route_score": score,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
        logger.info("llm.complete", extra=meta)
        return content, meta

    # ------------------------------------------------------------------
    # 流式
    # ------------------------------------------------------------------
    async def stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        force_tier: str | None = None,
    ) -> AsyncIterator[str]:
        """流式调用，逐块产出文本。用于 SSE 打字机效果。

        关键点：必须带上 response_format 强制 JSON 模式。
        否则模型在流式下常常吐出 Python 风格的单引号字典（{'key': 'value'}），
        前端 JSON.parse 会直接失败，只能显示一串原始文本 —— 演示时非常致命。
        """
        model, tier, score = self.resolve_route(f"{system}\n{user}", force_tier)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "stream": True,
            "response_format": {"type": "json_object"},
        }
        stream = await self._create(kwargs, model, True, "stream")

        collected: list[str] = []
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                collected.append(delta)
                yield delta
        # 流式拿不到精确 token 数（部分厂商最后一个 chunk 才给），按字符粗估，
        # 保证看板有数据。粗估 = 中文约 1.5 字符/token，这里取 /2 的保守值。
        text = "".join(collected)
        self.usage.record(model, len(system + user) // 2, len(text) // 2, tier)
        logger.info(
            "llm.stream",
            extra={"model": model, "tier": tier, "route_score": score, "chars": len(text)},
        )

    # ------------------------------------------------------------------
    # 内部：统一的调用 + JSON 模式降级
    # ------------------------------------------------------------------
    async def _create(
        self, kwargs: dict[str, Any], model: str, json_mode: bool, op: str
    ):
        try:
            return await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # 统一包装，不让 SDK 异常穿透到控制器
            if not (json_mode and is_unsupported_json_mode(exc)):
                raise ModelCallError(
                    f"模型调用失败（{self.settings.provider_label}）: {exc}"
                ) from exc
            # 厂商不支持强制 JSON 模式：去掉参数重试一次，输出交给三级容错解析兜底
            logger.warning(
                f"llm.{op}.json_mode_unsupported",
                extra={"model": model, "error": str(exc)[:200]},
            )
            kwargs.pop("response_format", None)
            try:
                return await self._client.chat.completions.create(**kwargs)
            except Exception as exc2:
                raise ModelCallError(
                    f"模型调用失败（{self.settings.provider_label}）: {exc2}"
                ) from exc2
