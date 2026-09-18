"""结构化日志：每行一条 JSON，带请求 ID 贯穿全链路。

为什么不用 print / 默认 logging 格式：
- 默认格式是给人看的，但线上排查需要**按字段过滤**（"把这个 request_id 的所有日志捞出来"）。
  文本格式下只能靠 grep 字符串，字段顺序一变就失效。
- JSON 行格式可以直接被 Loki / ELK / CloudWatch 结构化解析，不需要写正则。

请求 ID 用 contextvar 而不是函数传参：
- 传参意味着每个函数签名都要多一个 request_id，业务代码被迫感知可观测性细节；
- contextvar 在 async 场景下按协程隔离，天然正确。
"""
from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

# 当前请求 ID。默认 "-" 表示不在请求上下文中（启动阶段、后台任务）
_request_id: ContextVar[str] = ContextVar("request_id", default="-")

# 这些是 LogRecord 的内置属性，序列化时要排除，否则每行日志都带一堆噪声
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


def new_request_id() -> str:
    """生成一个短请求 ID（8 位足够，太长反而不好念、不好贴进聊天框）。"""
    return uuid.uuid4().hex[:8]


def set_request_id(rid: str | None = None) -> str:
    rid = rid or new_request_id()
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


class JsonFormatter(logging.Formatter):
    """把 LogRecord 序列化成单行 JSON。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        # extra={...} 传进来的业务字段直接平铺到顶层，前端/日志系统无需再解一层嵌套
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # ensure_ascii=False：中文直接可读，不转成 \uXXXX 串（查询时肉眼可搜）
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO", stream=None) -> None:
    """安装全局日志配置。重复调用是幂等的（避免 uvicorn reload 时句柄翻倍）。

    **默认写 stderr，不是 stdout。** 这条不是风格问题，是协议问题：
    MCP 的 stdio 传输里 stdout 是**协议通道**，混进一行日志客户端就解析失败。
    本模块的 docstring 与 `mcp/server.py` 都写了这条禁忌，但默认值指向 stdout
    时，只要有人忘了覆盖 stream，禁忌就会被无声违反 —— 把正确行为做成默认值，
    比在每个入口重复提醒更可靠。

    stderr 对普通 CLI 也更合适：stdout 留给数据，日志走 stderr，
    `python -m xxx > out.json` 才不会被日志污染。
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # uvicorn 自带 handler 会输出非 JSON 的访问日志，这里让它复用根 handler
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True
    # 第三方库在 INFO 级别过于聒噪，压到 WARNING
    for name in ("httpx", "httpcore", "chromadb", "openai", "urllib3"):
        logging.getLogger(name).setLevel("WARNING")
