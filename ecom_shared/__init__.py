"""ecom-agent-shared —— MCP & Agent 共享集群。

跨境电商 AI 生态的最底层：所有业务 Agent（售前咨询、内容运营、数据中台、
销售考核）都通过它接入模型、检索知识、读写记忆、调用工具。

    ┌──────────────────────────────────────────────┐
    │  业务应用：售前 Agent / 内容 Agent / 数据中台 / 销售考核   │
    └───────────────────────┬──────────────────────┘
                            │ 只依赖这一个包
    ┌───────────────────────▼──────────────────────┐
    │  ecom-agent-shared（本包）                      │
    │  ├─ gateway  LLM 网关：多厂商 / 路由 / 容错 / 记账  │
    │  ├─ rag      切分 / 向量化 / 三后端 / 混合重排     │
    │  ├─ memory   会话记忆 + 长期事实记忆              │
    │  ├─ prompts  Prompt 注册中心（版本化、可覆写）      │
    │  └─ mcp      工具注册 + JSON-RPC + stdio/HTTP   │
    └──────────────────────────────────────────────┘

快速上手：

    from ecom_shared import SharedCluster

    cluster = SharedCluster.build()                    # 读环境变量
    print(cluster.describe()["provider"])              # 当前接的是哪家模型

    hits = await cluster.rag.search("ACOS 过高怎么办")
    out = await cluster.tools.call("rag_search", {"query": "ACOS 过高怎么办"})

离线跑（不需要任何 API Key）：

    LLM_PROVIDER=mock VECTOR_BACKEND=lexical python demo.py
"""
from .cluster import SharedCluster
from .config import PROVIDER_PRESETS, Settings, load_settings
from .errors import (
    AppError,
    ConfigError,
    ExternalApiError,
    KnowledgeBaseError,
    ModelCallError,
    ModelOutputError,
    NotFoundError,
    RateLimitError,
    ToolError,
    UnauthorizedError,
    ValidationError,
)
from .gateway import (
    LLMGateway,
    MockLLMGateway,
    UsageStats,
    classify_complexity,
    get_gateway,
    parse_json_array_lenient,
    parse_json_lenient,
)
from .logging_setup import get_request_id, set_request_id, setup_logging
from .memory import Fact, LongTermMemory, SessionMemory
from .prompts import PromptRegistry, PromptTemplate
from .rag import Chunk, RagService, RetrievedChunk, rerank
from .mcp import ToolRegistry, build_shared_registry

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # 门面
    "SharedCluster",
    # 配置与错误
    "Settings",
    "load_settings",
    "PROVIDER_PRESETS",
    "AppError",
    "ConfigError",
    "ModelCallError",
    "ModelOutputError",
    "KnowledgeBaseError",
    "ToolError",
    "ValidationError",
    "NotFoundError",
    "UnauthorizedError",
    "ExternalApiError",
    "RateLimitError",
    # 网关
    "LLMGateway",
    "MockLLMGateway",
    "UsageStats",
    "get_gateway",
    "classify_complexity",
    "parse_json_lenient",
    "parse_json_array_lenient",
    # RAG
    "RagService",
    "Chunk",
    "RetrievedChunk",
    "rerank",
    # 记忆
    "SessionMemory",
    "LongTermMemory",
    "Fact",
    # Prompt
    "PromptRegistry",
    "PromptTemplate",
    # 工具
    "ToolRegistry",
    "build_shared_registry",
    # 观测
    "setup_logging",
    "set_request_id",
    "get_request_id",
]
