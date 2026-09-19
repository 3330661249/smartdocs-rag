"""Regression tests that keep the real prompt and application boundaries intact."""

import json

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from evals import run_eval
from src import qa_chain, vectorstore


@pytest.fixture
def evidence():
    return [Document(page_content="RAG 是检索增强生成。", metadata={"source": "guide.txt", "chunk_id": 0})]


def fake_response(monkeypatch, content):
    # Replace only the paid provider; template formatting and chain invocation stay real.
    monkeypatch.setattr(qa_chain, "get_chat_llm", lambda: RunnableLambda(lambda _: AIMessage(content=content)))


def test_generate_answer_formats_real_prompt_before_fake_provider(monkeypatch, evidence):
    fake_response(monkeypatch, json.dumps({
        "answer": "RAG 是检索增强生成 [1]", "enough_context": True, "used_citations": [1],
    }))
    result = qa_chain.generate_answer("RAG 是什么？", evidence)
    assert result["status"] == "answered"
    assert result["citations"][0]["source"] == "guide.txt"


@pytest.mark.parametrize("answer,status", [
    ("没有给出证据的断言", "invalid_citations"),
    ("不存在的引用 [99]", "invalid_citations"),
    ("合法与非法混用 [1] [99]", "invalid_citations"),
    ("", "invalid_response"),
    ("根据当前上下文信息不足，无法确认。", "insufficient_context"),
])
def test_stream_failures_never_receive_success_or_fabricated_citations(evidence, answer, status):
    result = qa_chain.build_stream_result(answer, evidence)
    assert result["status"] == status
    assert result["enough_context"] is False
    assert result["citations"] == []


@pytest.mark.parametrize("content", [
    "不是 JSON [1]",
    '[]',
    '{"answer": 123, "enough_context": true, "used_citations": [1]}',
    '{"answer": "答案 [1]", "enough_context": "false", "used_citations": [1]}',
    '{"answer": "答案 [1]", "enough_context": true, "used_citations": [true]}',
])
def test_malformed_provider_output_fails_closed(monkeypatch, evidence, content):
    fake_response(monkeypatch, content)
    result = qa_chain.generate_answer("问题", evidence)
    assert result["status"] == "invalid_response"
    assert result["enough_context"] is False
    assert result["citations"] == []


@pytest.mark.parametrize("answer,used", [("无引用的答案", []), ("答案 [99]", [99]), ("答案 [1]", [])])
def test_structured_citation_claims_must_match_existing_inline_citations(monkeypatch, evidence, answer, used):
    fake_response(monkeypatch, json.dumps({"answer": answer, "enough_context": True, "used_citations": used}))
    result = qa_chain.generate_answer("问题", evidence)
    assert result["status"] == "invalid_citations"
    assert result["enough_context"] is False
    assert result["citations"] == []


def test_no_evidence_refuses_without_calling_model(monkeypatch):
    def unexpected_provider():
        raise AssertionError("Empty evidence must not call a paid provider")

    monkeypatch.setattr(qa_chain, "get_chat_llm", unexpected_provider)
    assert qa_chain.generate_answer("问题", [])["status"] == "insufficient_context"


def test_selected_source_is_searched_before_candidate_truncation():
    other = Document(page_content="其他来源", metadata={"source": "other.txt"})
    target = Document(page_content="目标答案", metadata={"source": "target.txt"})

    class RankedStore:
        def similarity_search_with_relevance_scores(self, query, k, filter=None):
            ranked = [(other, 0.99)] * 5 + [(target, 0.9)]
            if filter:
                sources = filter["source"]
                allowed = sources["$in"] if isinstance(sources, dict) else [sources]
                ranked = [(doc, score) for doc, score in ranked if doc.metadata["source"] in allowed]
            return ranked[:k]

    results = vectorstore.search_similar_chunks(RankedStore(), "问题", k=1, allowed_sources=["target.txt"])
    assert [item["doc"].page_content for item in results] == ["目标答案"]


def test_local_chroma_persists_reloads_filters_and_deletes(tmp_path, monkeypatch):
    class OfflineEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.0]

    # Chroma's native dependencies can create session sidecars in the process cwd.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vectorstore, "VECTORSTORE_ROOT", tmp_path / "stores")
    monkeypatch.setattr(vectorstore, "get_embeddings", lambda: OfflineEmbeddings())
    docs = [Document(page_content=f"其他片段 {i}", metadata={"source": "other.txt", "chunk_id": i}) for i in range(8)]
    docs.append(Document(page_content="目标答案", metadata={"source": "target.txt", "chunk_id": 0}))
    vectorstore.build_vectorstore(docs, kb_name="offline_test")
    loaded = vectorstore.load_vectorstore("offline_test")
    assert len(loaded.get()["ids"]) == 9
    for sources in [["target.txt"], ["target.txt", "missing.txt"]]:
        result = vectorstore.search_similar_chunks(loaded, "问题", k=1, allowed_sources=sources)
        assert [item["doc"].page_content for item in result] == ["目标答案"]
    assert vectorstore.search_similar_chunks(loaded, "问题", allowed_sources=["missing.txt"]) == []
    vectorstore.delete_vectorstore("offline_test")
    assert not vectorstore.vectorstore_exists("offline_test")


@pytest.mark.parametrize("failure_stage", [None, "build", "search"])
def test_eval_uses_owned_temporary_library_and_cleans_it_on_failure(tmp_path, monkeypatch, failure_stage):
    stores = tmp_path / "stores"
    stores.mkdir()
    existing = stores / "eval_demo_kb"
    existing.mkdir()
    (existing / "keep.txt").write_text("user-owned", encoding="utf-8")
    monkeypatch.setattr(vectorstore, "VECTORSTORE_ROOT", stores)
    data = tmp_path / "data"
    data.mkdir()
    (data / "guide.txt").write_text("RAG 是检索增强生成。", encoding="utf-8")
    samples = tmp_path / "samples.json"
    samples.write_text(json.dumps([{
        "question": "RAG?", "expected_keywords": ["RAG"], "expected_source": "guide.txt",
    }]), encoding="utf-8")
    monkeypatch.setattr(run_eval, "DATA_DIR", data)
    monkeypatch.setattr(run_eval, "EVAL_FILE", samples)
    built_names = []

    def local_build(documents, kb_name, overwrite=False):
        built_names.append(kb_name)
        root = vectorstore.get_vectorstore_path(kb_name)
        if root.exists():
            raise ValueError("must not collide with an existing library")
        root.mkdir()
        if failure_stage == "build":
            raise RuntimeError("simulated embedding failure")
        return object()

    def local_search(*args, **kwargs):
        if failure_stage == "search":
            raise RuntimeError("simulated retrieval failure")
        return [{"doc": Document(page_content="RAG", metadata={"source": "guide.txt"})}]

    monkeypatch.setattr(run_eval, "build_vectorstore", local_build)
    monkeypatch.setattr(run_eval, "search_similar_chunks", local_search)
    monkeypatch.setattr(run_eval, "generate_answer", lambda *args: {
        "answer": "RAG [1]", "enough_context": True, "status": "answered",
        "citations": [{"source": "guide.txt"}],
    })
    for _ in range(2):
        with pytest.raises(RuntimeError if failure_stage else SystemExit) as exc:
            run_eval.main()
        if not failure_stage:
            assert exc.value.code == 0
        assert sorted(path.name for path in stores.iterdir()) == ["eval_demo_kb"]
    assert len(set(built_names)) == 2
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "user-owned"
