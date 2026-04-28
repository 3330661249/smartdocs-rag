import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_env_vars():
    """为所有测试提供模拟的环境变量，避免依赖真实 .env 文件。"""
    with patch.dict("os.environ", {
        "ZHIPU_API_KEY": "test-key-for-unit-tests",
        "ZHIPU_BASE_URL": "https://test.example.com/v4",
        "ZHIPU_CHAT_MODEL": "test-model",
        "ZHIPU_EMBEDDING_MODEL": "test-embedding",
    }):
        # 清除 lru_cache 以确保每个测试拿到新的实例
        from src.config import get_settings, get_embeddings, get_chat_llm
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_chat_llm.cache_clear()
        yield
        get_settings.cache_clear()
        get_embeddings.cache_clear()
        get_chat_llm.cache_clear()
