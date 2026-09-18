"""配置加载与校验的测试。

配置这一层出问题的方式很特别：**它不会崩，它会让服务带着错的值继续跑**。
少一个 Key 就在运行时才 401、chunk_overlap 大于 chunk_size 就在检索时莫名召回变差。
所以这里的用例只问一件事：该拒绝的配置，是否真的在**启动时**就拒绝了。

这也是本项目对外的承诺之一 —— 缺失即快速失败，报 503 加一句可读的原因，
而不是静默降级成"能起来但结果不对"。
"""
from __future__ import annotations

import os

import pytest

from ecom_shared.config import PROVIDER_PRESETS, load_settings
from ecom_shared.errors import ConfigError


# --------------------------------------------------------------- 正常路径

def test_mock_provider_needs_no_api_key():
    """离线跑得起来是硬要求：没有真 Key 的贡献者也要能 pytest。"""
    settings = load_settings(load_env_file=False)
    assert settings.provider == "mock"
    assert settings.is_mock is True


def test_provider_presets_cover_the_offline_ones():
    """离线两个 provider 必须在 preset 表里，否则 mock 会因为"未知 provider"被拒。"""
    assert "mock" in PROVIDER_PRESETS
    assert "ollama" in PROVIDER_PRESETS


# --------------------------------------------------------------- 该拒绝的配置

def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    monkeypatch.setenv("LLM_API_KEY", "k")
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


def test_missing_api_key_raises_for_real_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    for key in ("LLM_API_KEY", "API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


def test_missing_api_key_is_fine_for_ollama(monkeypatch):
    """ollama 走本机服务、mock 根本不出网 —— 这两个不该被"缺 Key"挡住。"""
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert load_settings(load_env_file=False).provider == "ollama"


def test_non_numeric_config_raises(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("TOP_K", "not-a-number")
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    """重叠必须严格小于块长。

    重叠加过头等于每个切片都被上一段污染、向量被稀释 —— 检索质量悄悄变差，
    但服务照样起得来。属于"必须在启动时拦住"的典型。
    """
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


@pytest.mark.parametrize("key", ["TOP_K", "CHUNK_SIZE", "REQUEST_TIMEOUT", "SESSION_MAX_TURNS"])
def test_zero_or_negative_values_are_rejected(monkeypatch, key):
    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv(key, "0")
    with pytest.raises(ConfigError):
        load_settings(load_env_file=False)


# --------------------------------------------------------------- load_env_file 开关

def test_load_env_file_false_ignores_the_dotenv_file(tmp_path, monkeypatch):
    """`load_env_file=False` 必须真的不读 .env。

    这是回归测试，它守的是一个"看着在跑、其实什么都没验证"的场景：
    `load_settings()` 如果在解析前又去读一遍 .env，那么调用方（以及测试）
    刚设置/刚删掉的环境变量会被文件里的值盖掉。曾经因为这个，
    `monkeypatch.delenv("LLM_API_KEY")` 之后仍然读到了 .env 里的 Key，
    于是"缺 Key 应当报错"的用例永远不报错 —— 却一直显示绿色。
    """
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    for key in ("LLM_API_KEY", "API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(ConfigError):
        load_settings(env_file=str(env_file), load_env_file=False)

    # 而且不能有"顺手把文件读进进程环境"的副作用
    assert os.environ.get("LLM_API_KEY") is None


def test_load_env_file_true_reads_the_dotenv_file(tmp_path, monkeypatch):
    """同一个文件，开关打开时应当被读到 —— 证明上面的用例失败是因为"没读"，
    而不是因为文件本身无效或路径写错了。"""
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_API_KEY=from-file\n", encoding="utf-8")

    monkeypatch.setenv("LLM_PROVIDER", "dashscope")
    for key in ("LLM_API_KEY", "API_KEY", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings(env_file=str(env_file))
    assert settings.api_key == "from-file"
