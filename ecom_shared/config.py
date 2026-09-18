"""共享集群的集中配置。

设计原则（三层都用同一套）：
1. 所有配置集中在这里，业务代码不直接读 os.getenv —— 配置散落的项目，改一个厂商
   要全局搜索 20 个文件。
2. 启动时一次性校验，缺密钥直接抛 ConfigError，不做静默降级。静默降级会把
   「其实没连通」伪装成「跑通了」，是 Demo 项目最常见的坑。
3. 多厂商 preset 表：切换 provider 只改一个环境变量，业务代码零改动。
4. mock 是一个**显式 provider**，不是"没配 key 时的隐式兜底"。
   这个区别很重要：隐式兜底会让人以为自己配好了 key 其实没生效，
   而 provider=mock 是明确声明"我现在就是要离线跑"。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .errors import ConfigError

# 包目录 = <repo>/ecom_shared，上一级就是仓库根
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_PKG_DIR)


# ---------------------------------------------------------------------------
# 多厂商 preset：统一走 OpenAI 兼容协议
# ---------------------------------------------------------------------------
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "dashscope": {
        "label": "阿里云百炼",
        "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "light": "qwen-plus",
        "heavy": "qwen-max",
        "embedding": "text-embedding-v3",
        "supports_embedding": "true",
    },
    "deepseek": {
        "label": "DeepSeek",
        "api_base": "https://api.deepseek.com/v1",
        "light": "deepseek-chat",
        "heavy": "deepseek-chat",
        "embedding": "",
        "supports_embedding": "false",
    },
    "openai": {
        "label": "OpenAI",
        "api_base": "https://api.openai.com/v1",
        "light": "gpt-4o-mini",
        "heavy": "gpt-4o",
        "embedding": "text-embedding-3-small",
        "supports_embedding": "true",
    },
    "ollama": {
        "label": "Ollama 本地",
        "api_base": "http://localhost:11434/v1",
        "light": "qwen2.5:3b",
        "heavy": "qwen2.5:7b",
        "embedding": "nomic-embed-text",
        "supports_embedding": "true",
    },
    "mock": {
        # 离线 provider：不需要 key、不联网，返回确定性内容。
        # 用于 CI、本地演示、以及"只想验证编排逻辑"的场景。
        "label": "Mock（离线）",
        "api_base": "",
        "light": "mock-light",
        "heavy": "mock-heavy",
        "embedding": "",
        "supports_embedding": "false",
    },
}

# 轻量任务关键词：命中越多越倾向轻量模型
LIGHT_HINTS = ("分类", "判断", "提取", "是否", "翻译", "总结", "归类", "标签", "格式化")
# 重量任务关键词：命中越多越倾向重量模型
HEAVY_HINTS = ("分析", "诊断", "生成", "策略", "优化", "撰写", "方案", "评估", "对比", "归因")
# 文本超过该长度直接判定为复杂任务
COMPLEX_LENGTH_THRESHOLD = 200

# chroma = 生产默认（HNSW + 持久化）；numpy = 零依赖暴力检索（需外部提供向量）；
# lexical = 纯词法 BM25（不需要向量，保证离线/CI 一定有可用的检索后端）
VECTOR_BACKENDS = ("chroma", "numpy", "lexical")


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_int(key: str, default: int) -> int:
    raw = _env(key, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"配置项 {key}={raw!r} 不是合法整数") from exc


@dataclass(frozen=True)
class Settings:
    """运行时配置快照（不可变，避免运行期被意外修改）。

    只包含**共享层关心的东西**：模型接入 + 检索 + 记忆 + 日志。
    业务层自己的配置（飞书凭证、批量上限、调度时间等）由业务仓在
    自己那层扩展，不要塞进共享包 —— 否则共享包会被业务细节污染，
    第二个业务仓要用时就得带上一堆用不到的配置项。
    """

    provider: str
    api_key: str
    api_base: str
    model_light: str
    model_heavy: str
    model_embedding: str
    supports_embedding: bool
    request_timeout: int
    # --- 检索 ---
    top_k: int
    chunk_size: int
    chunk_overlap: int
    rerank_alpha: float
    vector_dir: str
    vector_backend: str
    collection: str
    # --- 记忆 / Prompt ---
    session_max_turns: int
    longterm_enabled: bool
    # --- 观测 ---
    log_level: str

    @property
    def provider_label(self) -> str:
        return PROVIDER_PRESETS.get(self.provider, {}).get("label", self.provider)

    @property
    def is_mock(self) -> bool:
        """是否离线模式。业务层据此决定要不要真去建索引、发通知。"""
        return self.provider == "mock"


def load_settings(env_file: str | None = None, *, load_env_file: bool = True) -> Settings:
    """读取并校验配置。任何一项不合法都直接抛 ConfigError。

    env_file 为 None 时按 python-dotenv 默认规则从当前工作目录向上找 .env；
    显式传入路径则以该文件为准（业务仓通常传自己仓库根目录的 .env）。

    load_env_file=False 表示"只按当前进程的环境变量解析，不要再去读 .env 文件"。
    什么时候需要它：读文件是有副作用的 I/O（会往 os.environ 里塞值），而校验逻辑
    应该是纯粹的函数。调用方若已经在模块导入阶段 load_dotenv 过一次（业务仓的常见
    做法，因为大多数模块要的配置早于 load_settings 被调用），或者需要确定性地按
    测试注入的环境变量取值，就传 False。
    不传这个开关会有一个隐蔽后果：测试里 monkeypatch.delenv("LLM_API_KEY") 删掉的
    变量，会被这里重新读 .env 时填回来，于是"缺 Key 应当报错"的用例永远不报错 ——
    测试看着在跑，其实什么都没验证。
    """
    if load_env_file:
        load_dotenv(env_file) if env_file else load_dotenv()

    provider = _env("LLM_PROVIDER", "dashscope").lower()
    if provider not in PROVIDER_PRESETS:
        raise ConfigError(
            f"未知的 LLM_PROVIDER={provider!r}，可选：{', '.join(PROVIDER_PRESETS)}"
        )
    preset = PROVIDER_PRESETS[provider]

    api_key = _env("LLM_API_KEY") or _env("API_KEY") or _env("DASHSCOPE_API_KEY")
    # ollama 走本地服务、mock 根本不出网，这两者不需要 key
    if provider not in ("ollama", "mock") and not api_key:
        raise ConfigError(
            f"缺少 LLM_API_KEY。当前 provider={provider}（{preset['label']}），"
            f"请填入对应厂商的 API Key，或改用 LLM_PROVIDER=mock 离线运行。"
        )

    chunk_size = _env_int("CHUNK_SIZE", 600)
    chunk_overlap = _env_int("CHUNK_OVERLAP", 50)
    top_k = _env_int("TOP_K", 4)
    timeout = _env_int("REQUEST_TIMEOUT", 120)
    session_max_turns = _env_int("SESSION_MAX_TURNS", 10)

    if chunk_overlap >= chunk_size:
        raise ConfigError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
    if chunk_size < 1:
        raise ConfigError("CHUNK_SIZE 必须 >= 1")
    if top_k < 1:
        raise ConfigError("TOP_K 必须 >= 1")
    if timeout < 1:
        raise ConfigError("REQUEST_TIMEOUT 必须 >= 1")
    if session_max_turns < 1:
        raise ConfigError("SESSION_MAX_TURNS 必须 >= 1")

    try:
        alpha = float(_env("RERANK_ALPHA", "0.7"))
    except ValueError as exc:
        raise ConfigError(f"RERANK_ALPHA={_env('RERANK_ALPHA')!r} 不是合法小数") from exc
    if not 0.0 <= alpha <= 1.0:
        raise ConfigError("RERANK_ALPHA 必须在 0~1 之间")

    backend = _env("VECTOR_BACKEND", "chroma").lower()
    if backend not in VECTOR_BACKENDS:
        raise ConfigError(
            f"VECTOR_BACKEND={backend!r} 不支持，可选：{', '.join(VECTOR_BACKENDS)}"
        )

    return Settings(
        provider=provider,
        api_key=api_key,
        api_base=_env("LLM_API_BASE") or preset["api_base"],
        model_light=_env("MODEL_LIGHT") or preset["light"],
        model_heavy=_env("MODEL_HEAVY") or preset["heavy"],
        model_embedding=_env("MODEL_EMBEDDING") or preset["embedding"],
        supports_embedding=preset["supports_embedding"] == "true",
        request_timeout=timeout,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rerank_alpha=alpha,
        vector_dir=os.getenv("VECTOR_DIR") or os.path.join(REPO_ROOT, "vector_store"),
        vector_backend=backend,
        collection=_env("VECTOR_COLLECTION", "ecom_kb"),
        session_max_turns=session_max_turns,
        longterm_enabled=_env("LONGTERM_MEMORY", "true").lower() != "false",
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )
