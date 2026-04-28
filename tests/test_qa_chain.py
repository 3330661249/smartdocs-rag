import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.qa_chain import (
    _build_citations,
    _build_history,
    _parse_json_response,
    build_stream_result,
    generate_answer,
    stream_answer,
)


def test_build_citations():
    docs = [
        Document(page_content="内容一" * 20, metadata={"source": "a.txt", "chunk_id": 0, "start_index": 0}),
        Document(page_content="内容二" * 20, metadata={"source": "b.txt", "chunk_id": 1, "start_index": 100, "page": 3}),
    ]
    citations = _build_citations(docs)
    assert len(citations) == 2
    assert citations[0]["index"] == 1
    assert citations[0]["source"] == "a.txt"
    assert citations[1]["page"] == 3
    assert len(citations[0]["preview"]) <= 120


def test_parse_json_response_clean():
    result = _parse_json_response('{"answer": "测试", "enough_context": true, "used_citations": [1]}')
    assert result["answer"] == "测试"
    assert result["enough_context"] is True


def test_parse_json_response_with_code_fence():
    raw = '```json\n{"answer": " fenced ", "enough_context": false, "used_citations": []}\n```'
    result = _parse_json_response(raw)
    assert result["answer"] == " fenced "


def test_build_history_empty():
    assert _build_history([]) == "无历史对话"


def test_build_history_recent_turns():
    history = [
        {"question": "Q1", "answer": "A1"},
        {"question": "Q2", "answer": "A2"},
        {"question": "Q3", "answer": "A3"},
        {"question": "Q4", "answer": "A4"},
    ]
    built = _build_history(history, limit=3)
    assert "Q1" not in built
    assert "Q2" in built
    assert "A4" in built


def test_generate_answer_empty_query():
    with pytest.raises(ValueError, match="不能为空"):
        generate_answer("", [])


def test_generate_answer_whitespace_query():
    with pytest.raises(ValueError, match="不能为空"):
        generate_answer("   ", [])


def test_generate_answer_too_long():
    long_query = "x" * 2001
    with pytest.raises(ValueError, match="超过最大限制"):
        generate_answer(long_query, [])


def test_generate_answer_with_mock_llm(monkeypatch):
    mock_response = MagicMock()
    mock_response.content = '{"answer": "RAG 是检索增强生成", "enough_context": true, "used_citations": [1]}'

    mock_chain = MagicMock()
    mock_chain.invoke.return_value = mock_response

    import src.qa_chain as qa_module
    original_prompt = qa_module.QA_PROMPT

    def mock_or(self, other):
        return mock_chain

    monkeypatch.setattr(type(original_prompt), "__or__", mock_or)

    docs = [
        Document(page_content="RAG 全称是 Retrieval-Augmented Generation", metadata={
            "source": "test.txt", "chunk_id": 0, "start_index": 0,
        }),
    ]
    result = generate_answer("RAG 是什么？", docs)
    assert "RAG" in result["answer"]
    assert result["enough_context"] is True


def test_build_stream_result_filters_citations():
    docs = [
        Document(page_content="第一段", metadata={"source": "a.txt", "chunk_id": 0, "start_index": 0}),
        Document(page_content="第二段", metadata={"source": "b.txt", "chunk_id": 1, "start_index": 20}),
    ]
    result = build_stream_result("答案引用了第一段 [1]", docs)
    assert result["enough_context"] is True
    assert len(result["citations"]) == 1
    assert result["citations"][0]["source"] == "a.txt"


def test_build_stream_result_insufficient_context():
    docs = [
        Document(page_content="第一段", metadata={"source": "a.txt", "chunk_id": 0, "start_index": 0}),
    ]
    result = build_stream_result("根据当前上下文信息不足，暂时无法确认。", docs)
    assert result["enough_context"] is False


def test_stream_answer_with_mock_llm(monkeypatch):
    mock_chain = MagicMock()
    mock_chain.stream.return_value = [
        MagicMock(content="RAG 是"),
        MagicMock(content="检索增强生成 [1]"),
    ]

    import src.qa_chain as qa_module
    original_prompt = qa_module.STREAM_QA_PROMPT

    def mock_or(self, other):
        return mock_chain

    monkeypatch.setattr(type(original_prompt), "__or__", mock_or)

    docs = [
        Document(page_content="RAG 全称是 Retrieval-Augmented Generation", metadata={
            "source": "test.txt", "chunk_id": 0, "start_index": 0,
        }),
    ]
    output = "".join(stream_answer("RAG 是什么？", docs))
    assert output == "RAG 是检索增强生成 [1]"
