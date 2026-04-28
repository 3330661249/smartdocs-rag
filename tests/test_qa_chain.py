import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from src.qa_chain import _build_citations, _parse_json_response, generate_answer


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
