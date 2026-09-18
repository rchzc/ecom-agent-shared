"""向量化：优先用厂商 embedding 接口，不可用时降级到本地模型。

**降级路径是真实语义向量，不是占位哈希。** 这点必须说清楚，否则"降级"听起来像
"随便给个向量让流程别报错"——那种做法会让检索结果变成随机排序，比直接报错更糟。

- DeepSeek 不提供 embedding 接口（supports_embedding=false），此时返回空列表，
  由向量库后端用本地 ONNX 模型（ChromaDB 内置）生成。
- ChromaDB 的本地模型是真实的 embedding 模型，只是维度和效果弱于云端，
  检索仍然是按语义相似度排的，不是假排序。
"""
from __future__ import annotations

import logging
from typing import Sequence

from openai import AsyncOpenAI

from ..config import Settings
from ..errors import ModelCallError

logger = logging.getLogger(__name__)


class Embedder:
    """文本向量化。

    返回空列表是**有意义的返回**（表示"交给后端自己算"），不是失败。
    调用方（向量库后端）据此决定是否传 embeddings 参数。
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None
        if settings.is_mock:
            # 离线模式：不建远端连接
            return
        if settings.supports_embedding and settings.model_embedding:
            self._client = AsyncOpenAI(
                api_key=settings.api_key or "ollama",
                base_url=settings.api_base,
                timeout=settings.request_timeout,
                max_retries=1,
            )
        elif settings.provider == "ollama" and settings.model_embedding:
            self._client = AsyncOpenAI(
                api_key="ollama",
                base_url=settings.api_base,
                timeout=settings.request_timeout,
            )

    @property
    def mode(self) -> str:
        """当前向量化模式，用于启动日志和前端展示 —— 用户有权知道检索用的是哪种向量。"""
        if self._client is not None:
            return f"remote:{self.settings.model_embedding}"
        return "local:backend-default"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本转成向量。返回空列表表示交给向量库后端的内置模型。"""
        if self._client is None:
            logger.info("embed.local_fallback", extra={"count": len(texts)})
            return []

        # 换行会破坏部分厂商 embedding 接口的分句逻辑，统一压成空格
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        try:
            resp = await self._client.embeddings.create(
                model=self.settings.model_embedding,
                input=cleaned,
            )
        except Exception as exc:
            raise ModelCallError(
                f"向量化失败（{self.settings.provider_label}）: {exc}"
            ) from exc

        vectors = [list(item.embedding) for item in resp.data]
        logger.info(
            "embed.remote",
            extra={"count": len(vectors), "dim": len(vectors[0]) if vectors else 0},
        )
        return vectors
