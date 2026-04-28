import json

from src.chat_history import clear_chat_history, load_chat_history, save_chat_history


def test_save_and_load_chat_history(tmp_path, monkeypatch):
    import src.chat_history as history_module

    monkeypatch.setattr(history_module, "get_vectorstore_path", lambda kb_name: tmp_path / kb_name)
    history = [
        {"question": "Q1", "answer": "A1", "citations": []},
        {"question": "Q2", "answer": "A2", "citations": [{"source": "a.txt"}]},
    ]

    save_chat_history("demo_kb", history)
    loaded = load_chat_history("demo_kb")
    assert loaded == history


def test_load_chat_history_returns_empty_for_invalid_json(tmp_path, monkeypatch):
    import src.chat_history as history_module

    monkeypatch.setattr(history_module, "get_vectorstore_path", lambda kb_name: tmp_path / kb_name)
    history_path = (tmp_path / "demo_kb" / "chat_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("{bad json", encoding="utf-8")

    assert load_chat_history("demo_kb") == []


def test_clear_chat_history(tmp_path, monkeypatch):
    import src.chat_history as history_module

    monkeypatch.setattr(history_module, "get_vectorstore_path", lambda kb_name: tmp_path / kb_name)
    save_chat_history("demo_kb", [{"question": "Q1", "answer": "A1"}])
    clear_chat_history("demo_kb")

    history_path = tmp_path / "demo_kb" / "chat_history.json"
    assert not history_path.exists()
