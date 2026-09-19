import os
import socket
from unittest.mock import patch

import pytest

os.environ["PYTHON_DOTENV_DISABLED"] = "1"


@pytest.fixture(autouse=True)
def block_external_network(monkeypatch):
    """Unit and local-storage tests must never call a live provider."""
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Network is disabled in the offline test suite")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    yield
    assert not attempts, "A dependency attempted a network call during an offline test"


@pytest.fixture(autouse=True)
def mock_env_vars():
    """为所有测试提供模拟的环境变量，避免依赖真实 .env 文件。"""
    with patch.dict("os.environ", {
        "ZHIPU_API_KEY": "test-key-for-unit-tests",
        "ZHIPU_BASE_URL": "https://test.example.com/v4",
        "ZHIPU_CHAT_MODEL": "test-model",
        "ZHIPU_EMBEDDING_MODEL": "test-embedding",
        "ANONYMIZED_TELEMETRY": "False",
        "LANGSMITH_TRACING": "false",
        "LANGCHAIN_TRACING_V2": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
    }):
        # 清除 lru_cache 以确保每个测试拿到新的实例
        from src.config import get_chat_llm, get_embeddings, get_settings
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_chat_llm.cache_clear()
        yield
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_chat_llm.cache_clear()
