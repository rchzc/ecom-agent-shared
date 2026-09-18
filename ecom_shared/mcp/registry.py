"""工具注册中心：对齐 MCP 协议的工具定义与调用语义。

**Function Calling 和 MCP 到底差在哪（面试必问，这里把答案固化进设计）：**

- Function Calling 是**厂商私有**的：OpenAI 的函数格式、Anthropic 的工具格式、
  百炼的格式互不兼容。你在 OpenAI 上写的工具有一天换到别家就得重写一遍。
- MCP（Model Context Protocol）是**协议标准**：工具的定义（tools/list）与调用
  （tools/call）走同一套 JSON-RPC 消息。工具实现一次，任何支持 MCP 的客户端
  都能接 —— Claude Desktop、IDE 插件、自研 Agent 都能用同一批工具。
- 一句话概括：**Function Calling 是"某个模型的工具"，MCP 是"所有模型的工具"。**

这个模块实现的是 MCP 的**工具语义**（tools/list + tools/call + JSON-RPC 分帧），
传输层（stdio / HTTP）由 `server.py` 负责。这样拆的好处是：
工具定义与传输解耦，换传输不影响任何工具实现，也方便单测（直接调 handle 即可，
不用起进程）。

**关键设计：工具出错不抛异常，返回 isError 结果。**
原因在 Agent 循环里：一次工具失败不是"链路断了"，而是"这次尝试没成功" ——
循环应该把失败原因当成观察结果喂回模型，让它换个策略（换关键词、换工具）。
如果抛异常穿透上去，整个对话就中断了。这也是 MCP 协议本身的约定。
"""
from __future__ import annotations

import inspect
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

ToolFunc = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(frozen=True)
class ToolSpec:
    """一个工具的定义。字段命名对齐 MCP 的 Tool 对象。"""

    name: str
    description: str
    #: JSON Schema。手写而不是从函数签名自动推导 —— 签名推不出"这个参数的取值范围
    #: 是哪几个枚举值"，而模型恰恰最需要这个信息才能填对参数。
    input_schema: dict[str, Any]
    func: ToolFunc
    #: 归属模块，前端按此分组展示工具清单
    owner: str = "shared"
    #: 声明式标注：这个工具会读外部数据还是会产生副作用（写库、发通知）。
    #: Agent 循环可以据此决定"有副作用的工具在无人确认时不自动调用"。
    side_effect: bool = False

    def to_mcp(self) -> dict[str, Any]:
        """转成 MCP tools/list 里的 tool 条目。"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class ToolCallRecord:
    """一次工具调用的留痕。用于前端展示"这个回答用了哪些工具"，
    也是排查"为什么模型选了 B 工具而不是 A"的唯一依据。"""

    name: str
    arguments: dict[str, Any]
    ok: bool
    elapsed_ms: int
    error: str = ""
    result_preview: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "ok": self.ok,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "result_preview": self.result_preview,
        }


class ToolRegistry:
    """MCP 工具注册表 + JSON-RPC 处理器。"""

    PROTOCOL_VERSION = "2024-11-05"

    def __init__(self, server_name: str = "ecom-agent-shared") -> None:
        self.server_name = server_name
        self._tools: dict[str, ToolSpec] = {}
        #: 调用历史（有界），供前端与排障使用
        self._history: list[ToolCallRecord] = []
        self._history_limit = 200

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        func: ToolFunc,
        *,
        owner: str = "shared",
        side_effect: bool = False,
        override: bool = False,
    ) -> None:
        if name in self._tools and not override:
            raise ValueError(
                f"工具 {name!r} 已注册（owner={self._tools[name].owner}）。"
                f"如需覆盖请显式传 override=True。"
            )
        if not name or not name.replace("_", "").isalnum():
            # MCP 客户端按名字调用，名字里有奇怪字符会在协议层就出问题
            raise ValueError(f"工具名 {name!r} 不合法（只允许字母、数字、下划线）")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_schema=input_schema,
            func=func,
            owner=owner,
            side_effect=side_effect,
        )
        logger.info("mcp.tool_registered", extra={"tool": name, "owner": owner})

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def has(self, name: str) -> bool:
        return name in self._tools

    def get(self, name: str) -> ToolSpec:
        if name not in self._tools:
            raise KeyError(
                f"未注册的工具 {name!r}，已注册：{sorted(self._tools)}"
            )
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [self._tools[n] for n in self.names()]

    def list_tools(self) -> list[dict[str, Any]]:
        """MCP `tools/list` 的结果体。"""
        return [spec.to_mcp() for spec in self.specs()]

    def describe(self) -> list[dict[str, Any]]:
        """给前端用的工具清单（含 owner / side_effect，这些是 MCP 协议之外的元信息）。"""
        return [
            {
                **spec.to_mcp(),
                "owner": spec.owner,
                "side_effect": spec.side_effect,
                "parameters": list(
                    (spec.input_schema.get("properties") or {}).keys()
                ),
            }
            for spec in self.specs()
        ]

    def history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._history[-limit:]][::-1]

    # ------------------------------------------------------------------
    # 调用
    # ------------------------------------------------------------------
    async def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """执行工具，**永不抛异常**，失败时返回 ok=False 的结构化结果。

        参数校验只做"必填项是否存在"，不做类型强校验：
        类型不对时交给工具自己报错，报出的错误信息通常比通用校验器更具体
        （"top_k 必须是正整数，收到 'abc'" 比 "type mismatch at $.top_k" 有用）。
        """
        arguments = arguments or {}
        if name not in self._tools:
            return {
                "ok": False,
                "error": f"unknown tool: {name}",
                "available": self.names(),
            }

        spec = self._tools[name]
        missing = [
            key
            for key in (spec.input_schema.get("required") or [])
            if key not in arguments
        ]
        if missing:
            return {
                "ok": False,
                "error": f"缺少必填参数：{', '.join(missing)}",
                "required": spec.input_schema.get("required") or [],
            }

        started = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(spec.func):
                result = await spec.func(**arguments)
            else:
                result = spec.func(**arguments)
            if inspect.isawaitable(result):
                result = await result
            ok, error = True, ""
        except TypeError as exc:
            # 参数名写错是最常见的失败原因（模型偶尔会臆造参数名），
            # 单独把它拎出来，错误信息里带上正确的参数名，方便模型下一轮自我纠正
            ok, error = False, f"参数不匹配：{exc}"
            result = {
                "ok": False,
                "error": error,
                "accepted_arguments": list(
                    (spec.input_schema.get("properties") or {}).keys()
                ),
            }
        except Exception as exc:
            ok, error = False, str(exc)
            result = {"ok": False, "error": error}

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        record = ToolCallRecord(
            name=name,
            arguments=arguments,
            ok=ok,
            elapsed_ms=elapsed_ms,
            error=error,
            result_preview=json.dumps(result, ensure_ascii=False, default=str)[:200],
        )
        self._history.append(record)
        if len(self._history) > self._history_limit:
            del self._history[: len(self._history) - self._history_limit]

        logger.info(
            "mcp.tool_call",
            extra={"tool": name, "ok": ok, "elapsed_ms": elapsed_ms, "error": error[:200]},
        )
        if not ok:
            return {"ok": False, "error": error, "tool": name}
        return {"ok": True, "tool": name, "elapsed_ms": elapsed_ms, "result": result}

    # ------------------------------------------------------------------
    # JSON-RPC 分帧（MCP 传输层的公共部分，stdio 与 HTTP 都复用它）
    # ------------------------------------------------------------------
    async def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        """处理一条 JSON-RPC 请求。返回 JSON-RPC 响应体。

        支持的方法：initialize / tools/list / tools/call / ping。
        未实现的方法明确返回 -32601（Method not found），而不是假装成功 ——
        客户端能据此知道是自己版本不匹配，而不是工具坏了。
        """
        rpc_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        def ok(result: Any) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": rpc_id, "result": result}

        def err(code: int, message: str) -> dict[str, Any]:
            return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}

        if request.get("jsonrpc") != "2.0":
            return err(-32600, "invalid request: jsonrpc must be '2.0'")

        if method == "initialize":
            return ok(
                {
                    "protocolVersion": self.PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.server_name, "version": "1.0.0"},
                }
            )
        if method == "ping":
            return ok({})
        if method == "tools/list":
            return ok({"tools": self.list_tools()})
        if method == "tools/call":
            name = params.get("name")
            if not name:
                return err(-32602, "invalid params: 'name' is required")
            outcome = await self.call(name, params.get("arguments") or {})
            if not outcome.get("ok"):
                # 按 MCP 约定：工具执行失败仍然是**成功的 RPC 响应**，
                # 只是 content 里带 isError=true。这样客户端能把失败信息交给模型，
                # 而不是让整个 RPC 调用挂掉。
                return ok(
                    {
                        "content": [
                            {"type": "text", "text": outcome.get("error", "tool failed")}
                        ],
                        "isError": True,
                    }
                )
            return ok(
                {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                outcome["result"], ensure_ascii=False, default=str
                            ),
                        }
                    ],
                    "isError": False,
                }
            )
        return err(-32601, f"method not found: {method}")


if __name__ == "__main__":
    import asyncio

    reg = ToolRegistry()
    reg.register(
        "echo",
        "回显参数",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: {"echo": text},
    )

    async def main() -> None:
        print(await reg.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}))
        print(await reg.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}))
        print(await reg.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                "params": {"name": "echo", "arguments": {"text": "hi"}}}))
        # 未注册方法 → -32601
        bad = await reg.handle({"jsonrpc": "2.0", "id": 4, "method": "nope"})
        assert bad["error"]["code"] == -32601
        # 工具失败 → RPC 成功 + isError
        fail = await reg.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                 "params": {"name": "echo", "arguments": {}}})
        assert fail["result"]["isError"] is True
        print("ToolRegistry 自测通过")

    asyncio.run(main())
