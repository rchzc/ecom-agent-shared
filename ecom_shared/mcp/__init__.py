"""MCP 工具层：工具注册、JSON-RPC 分帧、stdio / HTTP 传输。"""
from .builtin import build_shared_registry
from .registry import ToolCallRecord, ToolRegistry, ToolSpec
from .server import build_asgi_app, serve_http, serve_stdio

__all__ = [
    "ToolRegistry",
    "ToolSpec",
    "ToolCallRecord",
    "build_shared_registry",
    "serve_stdio",
    "serve_http",
    "build_asgi_app",
]
