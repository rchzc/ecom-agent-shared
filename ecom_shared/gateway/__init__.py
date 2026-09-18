"""模型接入层：路由、真实网关、离线替身。"""
from .llm import PRICE_TABLE, LLMGateway, UsageStats
from .mock import MOCK_TAG, MockLLMGateway, get_gateway
from .router import (
    classify_complexity,
    is_unsupported_json_mode,
    parse_json_array_lenient,
    parse_json_lenient,
)

__all__ = [
    "LLMGateway",
    "MockLLMGateway",
    "UsageStats",
    "PRICE_TABLE",
    "MOCK_TAG",
    "get_gateway",
    "classify_complexity",
    "is_unsupported_json_mode",
    "parse_json_lenient",
    "parse_json_array_lenient",
]
