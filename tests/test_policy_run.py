"""The optional live runner must be bounded, traceable, and cleanup-safe."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document

from evals import policy_eval, policy_run


def test_live_run_records_selected_cases_and_cleans_store(tmp_path, monkeypatch):
    built = []
    deleted = []
    monkeypatch.setattr(policy_run, "get_settings", lambda: SimpleNamespace(
        zhipu_chat_model="example-chat", zhipu_embedding_model="example-embed"))
    monkeypatch.setattr(policy_run, "build_vectorstore", lambda docs, kb_name: built.append(kb_name) or object())
    monkeypatch.setattr(policy_run, "delete_vectorstore", deleted.append)
    monkeypatch.setattr(policy_run, "vectorstore_exists", lambda name: name in built)
    monkeypatch.setattr(policy_run, "search_similar_chunks", lambda *args, **kwargs: [
        {"doc": Document(page_content="铁路标准", metadata={"source": "travel-2026.md", "chunk_id": 0}), "score": 0.9},
    ])
    monkeypatch.setattr(policy_run, "generate_answer", lambda *args, **kwargs: {
        "answer": "按二等座报销 [1]", "status": "answered",
        "citations": [{"source": "travel-2026.md", "index": 1}],
    })
    path = tmp_path / "runs.jsonl"

    completed = policy_run.run_live(policy_run.load_cases()[:2], path)

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert completed == 2
    assert [row["id"] for row in rows] == ["D01", "D02"]
    assert all(row["retrieved_sources"] == ["travel-2026.md"] for row in rows)
    assert all(row["cited_sources"] == ["travel-2026.md"] for row in rows)
    assert all(row["model"] == "example-chat" and row["embedding_model"] == "example-embed" for row in rows)
    assert all(row["reviewer_support"] is None and row["duration_ms"] >= 0 for row in rows)
    assert len(built) == 1 and deleted == built
    scored = policy_eval.score_runs(policy_eval.load_cases(), rows, expected_suite_hash=policy_eval.suite_hash())
    assert scored["coverage"] == {"evaluated": 2, "total": 30}
    assert scored["suite_match"] is True
    assert scored["semantic_support"]["reviewed"] == 0


def test_execution_failure_records_fixed_status_and_stops_without_secret(tmp_path, monkeypatch):
    built = []
    deleted = []
    monkeypatch.setattr(policy_run, "get_settings", lambda: SimpleNamespace(
        zhipu_chat_model="example-chat", zhipu_embedding_model="example-embed"))
    monkeypatch.setattr(policy_run, "build_vectorstore", lambda docs, kb_name: built.append(kb_name) or object())
    monkeypatch.setattr(policy_run, "delete_vectorstore", deleted.append)
    monkeypatch.setattr(policy_run, "vectorstore_exists", lambda name: name in built)
    monkeypatch.setattr(policy_run, "search_similar_chunks", lambda *args, **kwargs: [
        {"doc": Document(page_content="铁路标准", metadata={"source": "travel-2026.md"}), "score": 0.9},
    ])

    def fail_provider(*args, **kwargs):
        raise RuntimeError("SECRET-DETAIL in vendor response")

    monkeypatch.setattr(policy_run, "generate_answer", fail_provider)
    path = tmp_path / "runs.jsonl"

    completed = policy_run.run_live(policy_run.load_cases()[:2], path)

    assert completed == 1
    output = path.read_text(encoding="utf-8")
    assert "SECRET-DETAIL" not in output
    assert json.loads(output)["status"] == "run_error"
    assert deleted == built


def test_cli_requires_explicit_live_mode_and_never_overwrites_runs(tmp_path, monkeypatch):
    path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(policy_run, "run_live", lambda *args, **kwargs: pytest.fail("live runner must not start"))

    with pytest.raises(SystemExit):
        policy_run.main(["--output", str(path)])
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(SystemExit):
        policy_run.main(["--live", "--output", str(path)])
    assert path.read_text(encoding="utf-8") == "existing"


def test_cli_requires_explicit_scope_for_full_suite(tmp_path, monkeypatch):
    selected = []
    monkeypatch.setattr(policy_run, "run_live", lambda cases, output: selected.extend(c["id"] for c in cases) or len(cases))

    policy_run.main(["--live", "--output", str(tmp_path / "runs.jsonl")])

    assert selected == ["D01", "D02", "D03"]
