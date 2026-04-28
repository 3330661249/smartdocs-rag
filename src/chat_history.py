import json
from pathlib import Path

from src.logging_utils import get_logger
from src.vectorstore import get_vectorstore_path, normalize_kb_name

logger = get_logger(__name__)

CHAT_HISTORY_FILE = "chat_history.json"


def _history_path(kb_name: str) -> Path:
    normalized = normalize_kb_name(kb_name)
    return get_vectorstore_path(normalized) / CHAT_HISTORY_FILE


def load_chat_history(kb_name: str) -> list[dict]:
    history_path = _history_path(kb_name)
    if not history_path.exists():
        return []

    try:
        content = json.loads(history_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("会话历史文件损坏，已忽略: kb_name=%s", kb_name)
        return []

    if not isinstance(content, list):
        logger.warning("会话历史格式无效，已忽略: kb_name=%s", kb_name)
        return []

    return [item for item in content if isinstance(item, dict)]


def save_chat_history(kb_name: str, history: list[dict]) -> None:
    history_path = _history_path(kb_name)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("会话历史已保存: kb_name=%s, turns=%d", kb_name, len(history))


def clear_chat_history(kb_name: str) -> None:
    history_path = _history_path(kb_name)
    if history_path.exists():
        history_path.unlink()
        logger.info("会话历史已清空: kb_name=%s", kb_name)
