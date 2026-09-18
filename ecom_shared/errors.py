"""类型化错误体系（共享集群的错误契约）。

上层业务模块只 raise 这些错误，由各自的框架适配层（FastAPI 处理器 / CLI 兜底）
统一转成 HTTP 响应或退出码。这样做的价值：

1. 不同错误对应不同处置：配置缺失 503（重试无用）、模型输出异常 502（上游脏数据）、
   参数错误 422（客户端问题）。调用方据此决定重试、降级还是直接报错。
2. 错误对外只暴露 code + message，绝不带堆栈 —— 堆栈会泄露内部路径与依赖版本。
3. 控制器里不写 try/except，避免"每个接口各自处理一遍异常"的重复代码。

放在共享包里而不是各业务仓自己定义，是因为所有业务 Agent 都调同一个 LLM 网关，
错误语义必须一致；否则「同一个上游故障」在 A 仓是 502、在 B 仓是 500，前端没法统一处理。
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """所有业务异常的基类。子类只需覆写 status_code / code。"""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, detail: Any = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": {
                "code": self.code,
                "message": self.message,
            }
        }
        if self.detail is not None:
            payload["error"]["detail"] = self.detail
        return payload


class ConfigError(AppError):
    """配置缺失或不合法 —— 服务端问题，重试无用。"""

    status_code = 503
    code = "config_error"


class ModelOutputError(AppError):
    """模型返回了无法解析的内容 —— 上游异常。"""

    status_code = 502
    code = "model_output_error"


class ModelCallError(AppError):
    """模型调用失败（超时、鉴权失败、限流）。"""

    status_code = 502
    code = "model_call_error"


class KnowledgeBaseError(AppError):
    """知识库不可用（未建库、检索失败、维度不匹配）。"""

    status_code = 503
    code = "knowledge_base_error"


class ToolError(AppError):
    """MCP 工具执行失败。

    与 ModelCallError 分开：工具失败往往是参数不对或外部系统挂了，
    调用方（Agent 循环）会把它当成一次"观察结果"喂回模型让它换个策略，
    而模型调用失败是整条链路断了，重试才有意义。
    """

    status_code = 502
    code = "tool_error"


class ValidationError(AppError):
    """输入不合法 —— 客户端问题。"""

    status_code = 422
    code = "validation_error"


class NotFoundError(AppError):
    """资源不存在。"""

    status_code = 404
    code = "not_found"


class UnauthorizedError(AppError):
    """凭据缺失或无效 —— 客户端问题，重试无用，需换凭据。"""

    status_code = 401
    code = "unauthorized"


class ExternalApiError(AppError):
    """外部系统调用失败（飞书 / Shopify / Amazon 等，非 LLM）。"""

    status_code = 502
    code = "external_api_error"


class RateLimitError(AppError):
    """触发限流 —— 客户端需退避后重试。"""

    status_code = 429
    code = "rate_limited"
