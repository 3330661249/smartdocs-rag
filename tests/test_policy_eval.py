"""An evaluation suite must expose missing evidence instead of inventing success."""

import json

import pytest

from evals import policy_eval


def test_synthetic_suite_has_30_grounded_and_distinct_cases():
    cases = policy_eval.load_cases()

    assert len(cases) == 30
    assert len({case["id"] for case in cases}) == 30
    assert {case["category"] for case in cases} == {"direct", "cross_source", "missing", "ambiguous"}
    assert sum(case["expected_behavior"] == "answer" for case in cases) == 18
    assert sum(case["expected_behavior"] == "abstain" for case in cases) == 12
    assert all(isinstance(case.get("reference_answer"), str) and case["reference_answer"].strip() for case in cases)


def test_score_keeps_retrieval_citation_and_human_support_separate():
    cases = [
        {"id": "answer", "category": "direct", "expected_behavior": "answer", "expected_sources": ["guide.md"]},
        {"id": "missing", "category": "missing", "expected_behavior": "abstain", "expected_sources": []},
    ]
    runs = [
        {"id": "answer", "retrieved_sources": ["guide.md"], "cited_sources": ["guide.md"],
         "status": "answered", "reviewer_support": None},
        {"id": "missing", "retrieved_sources": ["guide.md"], "cited_sources": [],
         "status": "insufficient_context", "reviewer_support": None},
    ]

    report = policy_eval.score_runs(cases, runs)

    assert report["coverage"] == {"evaluated": 2, "total": 2}
    assert report["answerable"]["retrieval_complete"] == {"passed": 1, "total": 1}
    assert report["answerable"]["citation_complete"] == {"passed": 1, "total": 1}
    assert report["answerable"]["answered"] == {"passed": 1, "total": 1}
    assert report["abstain"]["correct"] == {"passed": 1, "total": 1}
    assert report["semantic_support"] == {"supported": 0, "reviewed": 0, "unreviewed": 1}


def test_score_flags_bad_citation_and_wrongly_answered_ambiguous_case():
    cases = [
        {"id": "answer", "category": "direct", "expected_behavior": "answer", "expected_sources": ["guide.md"]},
        {"id": "ambiguous", "category": "ambiguous", "expected_behavior": "abstain", "expected_sources": []},
    ]
    runs = [
        {"id": "answer", "retrieved_sources": ["guide.md"], "cited_sources": ["other.md"],
         "status": "answered", "reviewer_support": "unsupported"},
        {"id": "ambiguous", "retrieved_sources": ["guide.md"], "cited_sources": ["guide.md"],
         "status": "answered", "reviewer_support": None},
    ]

    report = policy_eval.score_runs(cases, runs)

    assert report["answerable"]["retrieval_complete"] == {"passed": 1, "total": 1}
    assert report["answerable"]["citation_complete"] == {"passed": 0, "total": 1}
    assert report["abstain"]["correct"] == {"passed": 0, "total": 1}
    assert report["semantic_support"] == {"supported": 0, "reviewed": 1, "unreviewed": 0}


def test_partial_run_is_not_reported_as_full_suite_success():
    cases = [
        {"id": "one", "category": "direct", "expected_behavior": "answer", "expected_sources": ["guide.md"]},
        {"id": "two", "category": "direct", "expected_behavior": "answer", "expected_sources": ["guide.md"]},
    ]
    runs = [{"id": "one", "retrieved_sources": ["guide.md"], "cited_sources": ["guide.md"],
             "status": "answered", "reviewer_support": "supported"}]

    report = policy_eval.score_runs(cases, runs)

    assert report["coverage"] == {"evaluated": 1, "total": 2}
    assert report["complete"] is False
    assert report["missing_case_ids"] == ["two"]


def test_run_error_is_a_failed_case_without_a_human_score():
    cases = [{"id": "one", "category": "direct", "expected_behavior": "answer", "expected_sources": ["guide.md"]}]
    runs = [{"id": "one", "retrieved_sources": ["guide.md"], "cited_sources": [],
             "status": "run_error", "reviewer_support": None}]

    report = policy_eval.score_runs(cases, runs)

    assert report["answerable"]["answered"] == {"passed": 0, "total": 1}
    assert report["semantic_support"] == {"supported": 0, "reviewed": 0, "unreviewed": 0}
    assert report["failed_case_ids"] == ["one"]


def test_abstention_with_a_citation_is_not_counted_as_safe_refusal():
    cases = [{"id": "one", "category": "missing", "expected_behavior": "abstain", "expected_sources": []}]
    runs = [{"id": "one", "retrieved_sources": ["guide.md"], "cited_sources": ["guide.md"],
             "status": "insufficient_context", "reviewer_support": None}]

    report = policy_eval.score_runs(cases, runs)

    assert report["abstain"]["correct"] == {"passed": 0, "total": 1}
    assert report["failed_case_ids"] == ["one"]


def test_scoring_rejects_run_from_a_different_case_version():
    cases = [{"id": "one", "category": "missing", "expected_behavior": "abstain", "expected_sources": []}]
    runs = [{"id": "one", "retrieved_sources": [], "cited_sources": [], "status": "insufficient_context",
             "suite_sha256": "old-version", "reviewer_support": None}]

    with pytest.raises(ValueError, match="资料版本"):
        policy_eval.score_runs(cases, runs, expected_suite_hash="new-version")


def test_invalid_case_source_type_has_a_clear_error(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "id": "one", "category": "direct", "question": "测试？", "expected_behavior": "answer",
        "expected_sources": [{"bad": "source"}], "evidence": [], "reference_answer": "无法判断。",
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="预期来源"):
        policy_eval.load_cases(path)


def test_case_without_reference_answer_is_rejected(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([{
        "id": "one", "category": "missing", "question": "资料有吗？", "expected_behavior": "abstain",
        "expected_sources": [], "evidence": [],
    }]), encoding="utf-8")

    with pytest.raises(ValueError, match="参考"):
        policy_eval.load_cases(path)


@pytest.mark.parametrize("field,value", [("id", ["one"]), ("status", ["answered"])])
def test_malformed_run_identity_or_status_gives_validation_error(field, value):
    cases = [{"id": "one", "category": "missing", "expected_behavior": "abstain", "expected_sources": []}]
    row = {"id": "one", "retrieved_sources": [], "cited_sources": [],
           "status": "insufficient_context", "reviewer_support": None}
    row[field] = value

    with pytest.raises(ValueError):
        policy_eval.score_runs(cases, [row])


@pytest.mark.parametrize("runs", [
    [{"id": "unknown", "retrieved_sources": [], "cited_sources": [], "status": "insufficient_context"}],
    [{"id": "one", "retrieved_sources": [], "cited_sources": [], "status": "insufficient_context"}] * 2,
])
def test_unknown_or_duplicate_case_ids_are_rejected(runs):
    cases = [{"id": "one", "category": "missing", "expected_behavior": "abstain", "expected_sources": []}]

    with pytest.raises(ValueError):
        policy_eval.score_runs(cases, runs)


def test_cli_rejects_missing_run_file_without_claiming_a_score(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        policy_eval.main(["--runs", str(tmp_path / "missing.jsonl")])

    assert exc.value.code != 0
    assert "不存在" in capsys.readouterr().err


def test_jsonl_reader_rejects_non_object_rows(tmp_path):
    runs_path = tmp_path / "runs.jsonl"
    runs_path.write_text(json.dumps([]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON 对象"):
        policy_eval.load_runs(runs_path)
