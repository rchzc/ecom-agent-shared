"""共享集群的内置工具集。

**这 7 个工具就是"所有业务 Agent 复用同一套工具"这句话的实物。**
售前 Agent 要查知识库、内容 Agent 也要查知识库；售前 Agent 要发通知、
销售考核系统也要发通知 —— 如果每个 Agent 各写一份，就会出现
"知识库检索在 A 里按域过滤、在 B 里忘了过滤"这类不一致。

工具分成三类，按依赖注入的方式组装（`build_shared_registry`）：

1. **集群自带**：rag_search / memory_recall / memory_remember / notify / alert
   —— 依赖共享层自己的 RAG、记忆、日志，不需要业务方提供任何东西。
2. **业务数据（注入 provider）**：product_query / metrics_query
   —— 商品数据、经营数据属于业务仓，共享层不认识它们的数据结构。
   共享层只定义**工具契约**（参数与返回结构），运行期由业务仓注入实现。
   这样共享包不会被某个业务的数据模型绑死。
3. **外部集成（注入 notifier）**：飞书 / n8n webhook 由业务仓注入，
   未注入时退化为日志输出并明确标记 `delivered: False`
   —— 不假装通知发出去了。

依赖注入而不是直接 import 业务模块，是"共享集群能独立成仓库"的技术前提：
方向是业务依赖共享，不是共享依赖业务。
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from ..config import Settings
from ..memory.longterm import LongTermMemory
from ..memory.session import SessionMemory
from ..rag.service import RagService
from .registry import ToolRegistry

logger = logging.getLogger(__name__)

ProductProvider = Callable[[str], dict[str, Any]]
MetricsProvider = Callable[[str, int], dict[str, Any]]
Notifier = Callable[[str, str], dict[str, Any]] | Callable[[str, str], Awaitable[dict[str, Any]]]


def build_shared_registry(
    settings: Settings,
    rag: RagService,
    *,
    session_memory: SessionMemory | None = None,
    longterm_memory: LongTermMemory | None = None,
    product_provider: ProductProvider | None = None,
    metrics_provider: MetricsProvider | None = None,
    notifier: Notifier | None = None,
    server_name: str = "ecom-agent-shared",
) -> ToolRegistry:
    """组装共享工具注册表。业务仓调用这一个函数就能拿到全套工具。"""
    registry = ToolRegistry(server_name=server_name)

    # ------------------------------------------------------------------
    # 1. 知识检索 —— 全生态用得最多的一个工具
    # ------------------------------------------------------------------
    async def rag_search(query: str, domain: str = "*", top_k: int = 4) -> dict[str, Any]:
        chunks = await rag.search_as_context(query, domain=domain, top_k=int(top_k))
        return {
            "count": len(chunks),
            "domain": domain,
            "contexts": chunks,
        }

    registry.register(
        "rag_search",
        "检索跨境电商运营知识库，返回相关片段与来源文件名。适合回答『怎么做/为什么』类问题。",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索用的自然语言问题或关键词"},
                "domain": {
                    "type": "string",
                    "description": "限定业务域（如 selection / ads / logistics），* 表示不限",
                    "default": "*",
                },
                "top_k": {"type": "integer", "description": "返回片段数", "default": 4},
            },
            "required": ["query"],
        },
        rag_search,
        owner="shared",
    )

    # ------------------------------------------------------------------
    # 2. 记忆：读
    # ------------------------------------------------------------------
    if longterm_memory is not None:

        def memory_recall(query: str, scope: str = "default", top_k: int = 5) -> dict[str, Any]:
            facts = longterm_memory.recall(query, scope=scope, top_k=int(top_k))
            return {
                "count": len(facts),
                "facts": [
                    {"key": f.key, "value": f.value, "age_days": round(f.age_days(), 1)}
                    for f in facts
                ],
            }

        registry.register(
            "memory_recall",
            "检索关于当前客户/店铺的长期记忆事实（主营站点、价格敏感度、长期约束等）。",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "想回忆的内容"},
                    "scope": {"type": "string", "description": "记忆命名空间，通常是客户或店铺 ID", "default": "default"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
            memory_recall,
            owner="shared",
        )

        def memory_remember(
            key: str, value: str, scope: str = "default", tags: list[str] | None = None
        ) -> dict[str, Any]:
            longterm_memory.remember(key, value, scope=scope, tags=tags or [])
            return {"saved": True, "key": key, "scope": scope}

        registry.register(
            "memory_remember",
            "把一条值得长期保留的客户事实写入长期记忆（同 key 覆盖）。",
            {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "事实名，如『主营站点』"},
                    "value": {"type": "string", "description": "事实内容"},
                    "scope": {"type": "string", "default": "default"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["key", "value"],
            },
            memory_remember,
            owner="shared",
            # 写记忆是有副作用的操作，Agent 循环可在无人确认时不自动调用
            side_effect=True,
        )

    # ------------------------------------------------------------------
    # 3. 会话记忆：读（供多轮指代解析）
    # ------------------------------------------------------------------
    if session_memory is not None:

        def session_context(session_id: str, max_turns: int = 5) -> dict[str, Any]:
            turns = session_memory.history(session_id)[-max_turns * 2 :]
            return {
                "session_id": session_id,
                "turns": [{"role": t.role, "content": t.content} for t in turns],
                "last_user_query": session_memory.last_user_query(session_id),
            }

        registry.register(
            "session_context",
            "取回最近若干轮对话，用于解析『它』『那个』这类指代。",
            {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "max_turns": {"type": "integer", "default": 5},
                },
                "required": ["session_id"],
            },
            session_context,
            owner="shared",
        )

    # ------------------------------------------------------------------
    # 4. 业务数据（注入 provider）
    # ------------------------------------------------------------------
    def _no_product(name: str) -> dict[str, Any]:
        return {
            "found": False,
            "name": name,
            "reason": "未注入 product_provider，共享集群不认识业务商品数据。"
                      "请在业务仓调用 build_shared_registry 时传入实现。",
        }

    def product_query(name: str) -> dict[str, Any]:
        provider = product_provider or _no_product
        return provider(name)

    registry.register(
        "product_query",
        "按商品名/ASIN/SKU 查询商品资料（价格、卖点、库存、竞品）。数据由业务仓注入。",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "商品名、ASIN 或 SKU"},
            },
            "required": ["name"],
        },
        product_query,
        owner="shared",
    )

    def _no_metrics(seller_id: str, days: int) -> dict[str, Any]:
        return {
            "available": False,
            "reason": "未注入 metrics_provider。经营数据由业务仓（数据中台）提供。",
        }

    def metrics_query(seller_id: str, days: int = 7) -> dict[str, Any]:
        provider = metrics_provider or _no_metrics
        return provider(seller_id, int(days))

    registry.register(
        "metrics_query",
        "查询店铺经营指标（销售额、订单量、广告 ACOS、退货率）。数据由业务仓注入。",
        {
            "type": "object",
            "properties": {
                "seller_id": {"type": "string"},
                "days": {"type": "integer", "description": "统计最近多少天", "default": 7},
            },
            "required": ["seller_id"],
        },
        metrics_query,
        owner="shared",
    )

    # ------------------------------------------------------------------
    # 5. 通知与预警（副作用）
    # ------------------------------------------------------------------
    async def notify(message: str, channel: str = "feishu") -> dict[str, Any]:
        """发送通知。未注入 notifier 时**明确标记未投递**，不假装成功。"""
        if notifier is None:
            logger.warning("notify.no_backend", extra={"channel": channel})
            return {
                "delivered": False,
                "channel": channel,
                "reason": "未注入 notifier（业务仓接飞书 / n8n webhook）",
            }
        result = notifier(message, channel)
        if hasattr(result, "__await__"):
            result = await result  # type: ignore[assignment]
        return {"delivered": True, "channel": channel, **(result or {})}

    registry.register(
        "notify",
        "把消息推送到飞书群或 n8n webhook。",
        {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "channel": {"type": "string", "default": "feishu"},
            },
            "required": ["message"],
        },
        notify,
        owner="shared",
        side_effect=True,
    )

    def alert(message: str, level: str = "warn") -> dict[str, Any]:
        logger.warning("alert.raised", extra={"level": level, "message": message})
        return {"raised": True, "level": level, "message": message}

    registry.register(
        "alert",
        "触发智能预警，用于指标越界时主动提醒运营。",
        {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "level": {"type": "string", "enum": ["info", "warn", "critical"], "default": "warn"},
            },
            "required": ["message"],
        },
        alert,
        owner="shared",
        side_effect=True,
    )

    return registry
