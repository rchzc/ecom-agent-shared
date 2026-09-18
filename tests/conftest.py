"""测试公共 fixture。

**所有测试必须离线可跑** —— provider=mock、vector_backend=lexical。
没有真实 API Key 的贡献者 clone 下来就能 `pytest`，这是硬要求：
需要网络的测试会变成"本地能过、CI 挂掉"，最后没人跑。

真实厂商的连通性验证放在 `scripts/check_provider.py`，不进单测。
"""
from __future__ import annotations

import os

import pytest

from ecom_shared.config import load_settings


@pytest.fixture(scope="session", autouse=True)
def _offline_env():
    """把离线配置钉死在整个测试会话上。

    autouse + session：任何一个测试忘了设，都不会偷偷连真模型。
    用 monkeypatch 做不到 session 级，所以在会话级别直接改 os.environ 并在
    结束时还原。
    """
    saved = {k: os.environ.get(k) for k in ("LLM_PROVIDER", "VECTOR_BACKEND", "LOG_LEVEL", "LLM_API_KEY")}
    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["VECTOR_BACKEND"] = "lexical"
    os.environ["LOG_LEVEL"] = "CRITICAL"  # 测试输出里不要混日志
    os.environ.pop("LLM_API_KEY", None)
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def cluster(tmp_path, settings):
    """每个用例一个干净的集群：向量目录指向 tmp_path，避免用例互相污染。"""
    from dataclasses import replace

    from ecom_shared import SharedCluster

    return SharedCluster.build(
        settings=replace(settings, vector_dir=str(tmp_path / "vs")),
        setup_log=False,
    )
