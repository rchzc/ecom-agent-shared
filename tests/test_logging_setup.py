"""日志与可观测性单测。

重点是**日志写哪儿**：这不是风格问题，是协议问题。
MCP 的 stdio 传输下 stdout 是协议通道，日志混进去客户端就解析失败。
"""
from __future__ import annotations

import logging
import sys

from ecom_shared.logging_setup import (
    JsonFormatter,
    get_request_id,
    new_request_id,
    set_request_id,
    setup_logging,
)


def test_logs_go_to_stderr_not_stdout():
    """默认落 stderr —— stdio 传输的协议通道必须保持干净。"""
    setup_logging("INFO")
    handlers = logging.getLogger().handlers
    assert handlers
    assert all(h.stream is sys.stderr for h in handlers), "日志写到了 stdout，会污染 MCP stdio 通道"


def test_stream_can_be_overridden():
    setup_logging("INFO", stream=sys.stdout)
    assert logging.getLogger().handlers[0].stream is sys.stdout
    setup_logging("INFO")  # 还原，避免影响后续用例


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    setup_logging("INFO")
    setup_logging("INFO")
    assert len(logging.getLogger().handlers) == 1


def test_level_is_applied():
    setup_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    setup_logging("INFO")


def test_json_formatter_emits_single_line_json():
    import json

    record = logging.LogRecord(
        name="t", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello", args=(), exc_info=None,
    )
    record.extra_field = "v"
    line = JsonFormatter().format(record)
    assert "\n" not in line
    payload = json.loads(line)
    assert payload["msg"] == "hello"
    assert payload["extra_field"] == "v"
    assert payload["request_id"]


def test_request_id_roundtrip_and_autogeneration():
    rid = set_request_id()
    assert get_request_id() == rid
    assert set_request_id("abc12345") == "abc12345"
    assert get_request_id() == "abc12345"
    assert len(new_request_id()) == 8


def test_noisy_libraries_are_downgraded():
    setup_logging("INFO")
    for name in ("httpx", "httpcore", "openai"):
        assert logging.getLogger(name).level == logging.WARNING
