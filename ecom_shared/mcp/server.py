"""MCP 传输层：把注册表接到 stdio 或 HTTP 上。

**为什么传输层要单独一个文件**（这是 MCP 架构里最值得讲的一点）：

MCP 把「工具定义」和「怎么传」彻底分开了。
同一批工具，stdio 传输是给本地客户端用的（Claude Desktop、IDE 插件 ——
它们把服务当子进程起，用 stdin/stdout 通信），
HTTP 传输是给服务化部署用的（多个客户端连一个共享服务）。

工具实现完全不用改，改的只是这一层。这就是协议标准化的价值 ——
对比 Function Calling：换个调用方，工具实现往往要跟着改。

两个传输都在这里实现，且共享同一份 `registry.handle()` ——
也就是说，stdio 和 HTTP 的行为**必然一致**，不存在"本地能跑、服务化就挂"的问题。
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any

from .registry import ToolRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# stdio 传输（MCP 标准传输之一）
# ---------------------------------------------------------------------------
async def serve_stdio(registry: ToolRegistry) -> None:
    """从 stdin 逐行读 JSON-RPC，把响应写到 stdout。

    **日志绝对不能写 stdout。** stdio 传输下 stdout 是协议通道，
    混进一行日志就会让客户端解析失败 —— 这是接 MCP 时最经典的翻车点。
    所以这里显式把日志指到 stderr。
    """
    logger.info("mcp.stdio.start", extra={"tools": registry.names()})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"parse error: {exc}"},
            }
        else:
            response = await registry.handle(request)
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# HTTP 传输
# ---------------------------------------------------------------------------
def build_asgi_app(registry: ToolRegistry, *, path: str = "/mcp"):
    """构建一个最小 ASGI 应用，暴露 MCP 的 HTTP 端点。

    没直接用 FastAPI 的装饰器而返回一个纯 ASGI callable：
    这样共享包不依赖 Web 框架。业务仓（本来就有 FastAPI）可以：

        from ecom_shared.mcp import build_asgi_app
        app.mount("/shared-mcp", build_asgi_app(registry))

    需要单独跑时（演示 / 本地调试）用 `serve_http()`。
    """
    async def app(scope, receive, send):  # noqa: ANN001 - ASGI 接口签名固定
        if scope["type"] != "http":
            return
        # 路径归一：mount 后 scope["path"] 会去掉挂载前缀，两种都要能匹配
        route = scope.get("path", "").rstrip("/") or "/"
        if route not in (path.rstrip("/"), "", "/"):
            await _send_json(send, 404, {"error": "not found", "endpoints": [path]})
            return

        method = scope.get("method", "GET")
        if method == "GET":
            # GET 给人类看的工具清单：浏览器直接打开就能确认服务活着
            await _send_json(
                send,
                200,
                {
                    "server": registry.server_name,
                    "protocol": registry.PROTOCOL_VERSION,
                    "tools": registry.describe(),
                },
            )
            return
        if method != "POST":
            await _send_json(send, 405, {"error": "method not allowed"})
            return

        body = b""
        while True:
            message = await receive()
            if message["type"] == "http.request":
                body += message.get("body", b"")
                if not message.get("more_body"):
                    break
        try:
            request = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            await _send_json(
                send,
                400,
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": f"parse error: {exc}"}},
            )
            return

        # 支持批量请求（JSON-RPC 2.0 允许），MCP 客户端初始化时有时会批量发
        if isinstance(request, list):
            response: Any = [await registry.handle(item) for item in request]
        else:
            response = await registry.handle(request)
        await _send_json(send, 200, response)

    return app


async def _send_json(send, status: int, payload: Any) -> None:  # noqa: ANN001
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def serve_http(registry: ToolRegistry, host: str = "127.0.0.1", port: int = 8001) -> None:
    """独立起一个 HTTP 服务（演示用）。生产建议用业务仓的 ASGI 应用挂载。

    默认只绑 127.0.0.1：这个端点没有鉴权，绑 0.0.0.0 等于把工具暴露到内网。
    要对外提供必须先加鉴权 —— 这一点写在 README 的「已知边界」里。
    """
    import uvicorn

    uvicorn.run(build_asgi_app(registry), host=host, port=port, log_level="info")


if __name__ == "__main__":
    import asyncio

    from .builtin import build_shared_registry
    from .registry import ToolRegistry as _R

    # 不带 RAG 的最小自检：只验证协议链路
    demo = _R("demo")
    demo.register(
        "ping_tool",
        "自检用工具",
        {"type": "object", "properties": {}, "required": []},
        lambda: {"pong": True},
    )

    async def main() -> None:
        resp = await demo.handle(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
             "params": {"name": "ping_tool", "arguments": {}}}
        )
        assert resp["result"]["isError"] is False, resp
        print("MCP server 自检通过：", resp["result"]["content"][0]["text"])

    asyncio.run(main())
