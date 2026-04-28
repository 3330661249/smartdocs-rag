import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_ROOT = BASE_DIR / "vectorstores"
DEFAULT_EMBEDDING_MODEL = "Embedding-3"
DEFAULT_CHAT_MODEL = "glm-4.7"
MAX_QUERY_LENGTH = 2000
MAX_FILE_SIZE_MB = 50


@dataclass(frozen=True)
class Settings:
    zhipu_api_key: str
    zhipu_base_url: str
    zhipu_chat_model: str = DEFAULT_CHAT_MODEL
    zhipu_embedding_model: str = DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    base_url = os.getenv("ZHIPU_BASE_URL", "").strip()
    chat_model = os.getenv("ZHIPU_CHAT_MODEL", DEFAULT_CHAT_MODEL).strip() or DEFAULT_CHAT_MODEL
    embedding_model = os.getenv("ZHIPU_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL

    if not api_key:
        raise ValueError("未检测到 ZHIPU_API_KEY，请检查 .env 文件配置。")
    if not base_url:
        raise ValueError("未检测到 ZHIPU_BASE_URL，请检查 .env 文件配置。")

    return Settings(
        zhipu_api_key=api_key,
        zhipu_base_url=base_url,
        zhipu_chat_model=chat_model,
        zhipu_embedding_model=embedding_model,
    )


@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    return OpenAIEmbeddings(
        model=settings.zhipu_embedding_model,
        api_key=settings.zhipu_api_key,
        base_url=settings.zhipu_base_url,
    )


@lru_cache(maxsize=1)
def get_chat_llm():
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    return ChatOpenAI(
        model=settings.zhipu_chat_model,
        api_key=settings.zhipu_api_key,
        base_url=settings.zhipu_base_url,
        temperature=0,
    )
