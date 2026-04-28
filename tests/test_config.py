import pytest
from src.config import get_settings, Settings


def test_get_settings_returns_settings():
    settings = get_settings()
    assert isinstance(settings, Settings)
    assert settings.zhipu_api_key == "test-key-for-unit-tests"
    assert settings.zhipu_base_url == "https://test.example.com/v4"
    assert settings.zhipu_chat_model == "test-model"
    assert settings.zhipu_embedding_model == "test-embedding"


def test_get_settings_cached():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_get_settings_missing_api_key():
    import os
    original = os.environ.pop("ZHIPU_API_KEY", None)
    try:
        from src.config import get_settings
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
            get_settings()
    finally:
        if original is not None:
            os.environ["ZHIPU_API_KEY"] = original
        get_settings.cache_clear()


def test_get_settings_missing_base_url():
    import os
    original = os.environ.pop("ZHIPU_BASE_URL", None)
    try:
        from src.config import get_settings
        get_settings.cache_clear()
        with pytest.raises(ValueError, match="ZHIPU_BASE_URL"):
            get_settings()
    finally:
        if original is not None:
            os.environ["ZHIPU_BASE_URL"] = original
        get_settings.cache_clear()


def test_get_embeddings_singleton():
    from src.config import get_embeddings
    e1 = get_embeddings()
    e2 = get_embeddings()
    assert e1 is e2


def test_get_chat_llm_singleton():
    from src.config import get_chat_llm
    l1 = get_chat_llm()
    l2 = get_chat_llm()
    assert l1 is l2
