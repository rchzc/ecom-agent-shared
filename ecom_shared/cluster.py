"""共享集群门面：一次组装，业务侧全套可用。

这是业务仓唯一需要 import 的东西：

    from ecom_shared import SharedCluster

    cluster = SharedCluster.build(scope_prefix="presale")
    ctx = await cluster.rag.search_as_context("ACOS 过高")
    answer, meta = await cluster.gateway.complete(system, user)

    # 或者走工具协议（Agent 循环用这个）
    result = await cluster.tools.call("rag_search", {"query": "ACOS 过高"})

**为什么要有这一层**：六个业务项目如果各自 new 一遍 gateway / rag / memory，
就会出现"售前 Agent 用了 RAG，内容 Agent 忘了初始化长记忆"这种不一致，
而且每个项目都要复制一份组装代码。收敛成一个入口后，
组装逻辑只有一处，「共享集群」这个词才有实物对应 —— 否则它只是架构图上的一行字。

组装顺序有依赖：gateway 不依赖别人，rag 依赖 settings，tools 依赖 rag + memory。
`build()` 按这个顺序来，业务侧不用关心。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .config import Settings, load_settings
from .gateway.llm import UsageStats
from .gateway.mock import get_gateway
from .logging_setup import setup_logging
from .mcp.builtin import MetricsProvider, Notifier, ProductProvider, build_shared_registry
from .mcp.registry import ToolRegistry
from .memory.longterm import JsonFileStore, LongTermMemory, MemoryStore
from .memory.session import SessionMemory
from .prompts.registry import PromptRegistry, build_default_registry
from .rag.service import RagService

logger = logging.getLogger(__name__)


@dataclass
class SharedCluster:
    """共享集群的运行时实例。"""

    settings: Settings
    gateway: Any  # LLMGateway | MockLLMGateway —— 接口一致，这里不强制类型
    rag: RagService
    prompts: PromptRegistry
    tools: ToolRegistry
    session_memory: SessionMemory
    longterm_memory: LongTermMemory
    #: 业务侧后续注册的 Prompt / 工具都记在这里，便于 describe() 展示归属
    _extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        settings: Settings | None = None,
        *,
        env_file: str | None = None,
        memory_store: MemoryStore | None = None,
        product_provider: ProductProvider | None = None,
        metrics_provider: MetricsProvider | None = None,
        notifier: Notifier | None = None,
        server_name: str = "ecom-agent-shared",
        setup_log: bool = True,
    ) -> "SharedCluster":
        """组装集群。

        `env_file` 让业务仓指定自己仓库的 .env（各仓库配置独立，
        但共享包沿用同一套变量名，切换时不需要改代码）。
        """
        settings = settings or load_settings(env_file)
        if setup_log:
            setup_logging(settings.log_level)

        gateway = get_gateway(settings)
        rag = RagService(settings)
        prompts = build_default_registry()
        session_memory = SessionMemory(max_turns=settings.session_max_turns)
        longterm_memory = LongTermMemory(store=memory_store) if settings.longterm_enabled else LongTermMemory()

        tools = build_shared_registry(
            settings,
            rag,
            session_memory=session_memory,
            longterm_memory=longterm_memory,
            product_provider=product_provider,
            metrics_provider=metrics_provider,
            notifier=notifier,
            server_name=server_name,
        )

        logger.info(
            "cluster.ready",
            extra={
                "provider": settings.provider,
                "mock": settings.is_mock,
                "vector_backend": settings.vector_backend,
                "embedding_mode": rag.embedder.mode,
                "tools": len(tools.names()),
                "prompts": len(prompts.names()),
            },
        )
        return cls(
            settings=settings,
            gateway=gateway,
            rag=rag,
            prompts=prompts,
            tools=tools,
            session_memory=session_memory,
            longterm_memory=longterm_memory,
        )

    @classmethod
    def build_from_memory_file(
        cls, path: str, settings: Settings | None = None, **kwargs: Any
    ) -> "SharedCluster":
        """把长期记忆落到 JSON 文件，重启不丢。"""
        return cls.build(settings=settings, memory_store=JsonFileStore(path), **kwargs)

    # ------------------------------------------------------------------
    @property
    def usage(self) -> UsageStats:
        return self.gateway.usage

    # --- 常用转调：数据中台几乎每次启动都要建索引，给个直通方法省掉 c.rag.xxx ---
    async def ingest_documents(self, documents, *, rebuild: bool = True):
        return await self.rag.ingest_documents(documents, rebuild=rebuild)

    async def ingest_dir(self, docs_dir: str, *, rebuild: bool = True):
        return await self.rag.ingest_dir(docs_dir, rebuild=rebuild)

    async def search(
        self,
        query: str,
        *,
        domain: str = "*",
        top_k: int | None = None,
        alpha: float | None = None,
    ):
        """检索。`alpha` 单独暴露出来是为了让评测脚本能扫参 ——
        重排权重应该有实测依据，不该是个写死的常量。"""
        return await self.rag.search(query, domain=domain, top_k=top_k, alpha=alpha)

    def describe(self) -> dict[str, Any]:
        """集群状态快照。前端"共享集群"面板直接渲染这个。

        把 mock / 后端 / embedding 模式都暴露出来，是为了避免"看起来在跑真模型
        其实是 mock"这种自欺 —— 顶部状态栏会明明白白写出 `mock: true`。
        """
        return {
            "provider": self.settings.provider,
            "provider_label": self.settings.provider_label,
            "mock": self.settings.is_mock,
            "model_light": self.settings.model_light,
            "model_heavy": self.settings.model_heavy,
            "retrieval": self.rag.stats(),
            "tools": self.tools.describe(),
            "prompts": self.prompts.describe(),
            "usage": self.usage.snapshot(),
            "longterm_memory": {
                "enabled": self.settings.longterm_enabled,
                "half_life_days": self.longterm_memory.half_life_days,
            },
        }

    def register_tool(self, *args: Any, **kwargs: Any) -> None:
        """业务侧注册自己的工具（复用同一份协议与调用留痕）。"""
        self.tools.register(*args, **kwargs)


__all__ = ["SharedCluster"]
