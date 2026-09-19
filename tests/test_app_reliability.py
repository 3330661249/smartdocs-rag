from pathlib import Path

import pytest
from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

from src import qa_chain, vectorstore


@pytest.mark.parametrize("draft,expected_history", [
    ("没有证据的模型断言", 0),
    ("来自文档的回答 [1]", 1),
])
def test_app_only_persists_answers_with_valid_citation_ids(tmp_path, monkeypatch, draft, expected_history):
    monkeypatch.setattr(vectorstore, "VECTORSTORE_ROOT", tmp_path / "stores")
    monkeypatch.setattr(vectorstore, "list_vectorstores", lambda: [])
    monkeypatch.setattr(vectorstore, "get_vectorstore_metadata", lambda _: {"sources": ["guide.txt"]})
    doc = Document(page_content="可追溯内容", metadata={"source": "guide.txt", "chunk_id": 0})
    monkeypatch.setattr(vectorstore, "search_similar_chunks", lambda *args, **kwargs: [{"doc": doc, "score": 0.9}])
    monkeypatch.setattr(qa_chain, "stream_answer", lambda *args, **kwargs: iter([draft]))

    # A fresh environment may spend more than the default 3 seconds loading Streamlit's rendering dependencies.
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.session_state["vectorstore"] = object()
    app.session_state["current_kb_name"] = "ui_test"
    app.session_state["chat_history"] = []
    app.run()
    app.chat_input[0].set_value("文档说了什么？").run()

    assert not app.exception
    assert len(app.session_state["chat_history"]) == expected_history
    visible_markdown = "\n".join(item.value for item in app.markdown)
    if expected_history:
        assert draft in visible_markdown
        assert (tmp_path / "stores" / "ui_test" / "chat_history.json").exists()
    else:
        assert draft not in visible_markdown
        assert "未通过校验" in visible_markdown
        assert not (tmp_path / "stores" / "ui_test" / "chat_history.json").exists()


def test_interrupted_stream_clears_partial_answer_and_hides_provider_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(vectorstore, "VECTORSTORE_ROOT", tmp_path / "stores")
    monkeypatch.setattr(vectorstore, "list_vectorstores", lambda: [])
    monkeypatch.setattr(vectorstore, "get_vectorstore_metadata", lambda _: {"sources": ["guide.txt"]})
    doc = Document(page_content="证据", metadata={"source": "guide.txt", "chunk_id": 0})
    monkeypatch.setattr(vectorstore, "search_similar_chunks", lambda *args, **kwargs: [{"doc": doc, "score": 0.9}])

    def interrupted(*args, **kwargs):
        yield "未完成的模型断言"
        raise RuntimeError("provider-private-test")

    monkeypatch.setattr(qa_chain, "stream_answer", interrupted)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15)
    app.session_state["vectorstore"] = object()
    app.session_state["current_kb_name"] = "ui_test"
    app.session_state["chat_history"] = []
    app.run()
    app.chat_input[0].set_value("问题").run()
    assert not app.exception
    visible = "\n".join(item.value for item in [*app.markdown, *app.error])
    assert "未完成的模型断言" not in visible
    assert "provider-private-test" not in visible + capsys.readouterr().err
    assert "QA_FAILED" in visible
    assert app.session_state["chat_history"] == []
