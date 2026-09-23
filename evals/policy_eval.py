"""Validate synthetic policy cases and score recorded runs without model calls.

The scorer checks source identity and explicit refusal. Whether an answer is
actually supported by a passage remains a separate human review field.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

CASE_DIR = Path(__file__).resolve().parent / "policy_case"
CASE_FILE = CASE_DIR / "cases.json"
CATEGORIES = {"direct", "cross_source", "missing", "ambiguous"}
BEHAVIORS = {"answer", "abstain"}
STATUSES = {"answered", "insufficient_context", "invalid_response", "invalid_citations", "run_error"}
REVIEWS = {None, "supported", "unsupported"}


def suite_hash() -> str:
    digest = hashlib.sha256()
    for path in [CASE_FILE, *sorted(CASE_DIR.glob("*.md"))]:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def load_cases(path: Path = CASE_FILE) -> list[dict]:
    """Read cases and check that quoted evidence exists in the synthetic corpus."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("评测集必须是非空数组。")

    seen = set()
    for case in raw:
        if not isinstance(case, dict):
            raise ValueError("每条评测样例必须是 JSON 对象。")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"评测样例 ID 缺失或重复：{case_id}")
        seen.add(case_id)
        category = case.get("category")
        behavior = case.get("expected_behavior")
        if (not isinstance(category, str) or category not in CATEGORIES
                or not isinstance(behavior, str) or behavior not in BEHAVIORS):
            raise ValueError(f"{case_id} 的类别或预期行为无效。")
        if not isinstance(case.get("question"), str) or not case["question"].strip():
            raise ValueError(f"{case_id} 缺少问题。")
        if not isinstance(case.get("reference_answer"), str) or not case["reference_answer"].strip():
            raise ValueError(f"{case_id} 缺少人工复核参考答案。")
        sources = case.get("expected_sources")
        if (not isinstance(sources, list) or any(not isinstance(s, str) for s in sources)
                or len(set(sources)) != len(sources)):
            raise ValueError(f"{case_id} 的预期来源无效。")
        if case["expected_behavior"] == "answer" and not sources:
            raise ValueError(f"{case_id} 缺少预期来源。")
        if case["expected_behavior"] == "abstain" and sources:
            raise ValueError(f"{case_id} 应拒答，不应填写预期来源。")

        evidence = case.get("evidence", [])
        if not isinstance(evidence, list) or (sources and not evidence):
            raise ValueError(f"{case_id} 缺少原文证据。")
        evidence_sources = set()
        for item in evidence:
            if (not isinstance(item, dict) or not isinstance(item.get("source"), str)
                    or not isinstance(item.get("quote"), str) or not item["quote"].strip()):
                raise ValueError(f"{case_id} 的证据结构无效。")
            source = item["source"]
            if source not in sources or Path(source).name != source:
                raise ValueError(f"{case_id} 引用了未登记的来源。")
            source_path = CASE_DIR / source
            if not source_path.is_file() or item["quote"] not in source_path.read_text(encoding="utf-8"):
                raise ValueError(f"{case_id} 的原文证据与资料不一致。")
            evidence_sources.add(source)
        if evidence_sources != set(sources):
            raise ValueError(f"{case_id} 的来源没有逐一给出原文证据。")
    return raw


def load_runs(path: Path) -> list[dict]:
    """Read one recorded run per line; never invoke the app or a provider."""
    runs = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"第 {line_number} 行不是有效 JSON。") from exc
        if not isinstance(row, dict):
            raise ValueError(f"第 {line_number} 行必须是 JSON 对象。")
        runs.append(row)
    if not runs:
        raise ValueError("运行记录为空。")
    return runs


def _metric(passed: int, total: int) -> dict:
    return {"passed": passed, "total": total}


def score_runs(cases: list[dict], runs: list[dict], expected_suite_hash: str | None = None) -> dict:
    """Compute observed behavior; leave semantic support unscored until reviewed."""
    case_by_id = {case["id"]: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("评测样例 ID 重复。")

    seen = set()
    answerable_total = abstain_total = 0
    retrieval_pass = citation_pass = answered_pass = abstain_pass = 0
    supported = reviewed = unreviewed = 0
    failures = []
    fingerprints = [row.get("suite_sha256") for row in runs if isinstance(row, dict)]
    if any(not isinstance(value, (str, type(None))) for value in fingerprints) or len(set(fingerprints)) != 1:
        raise ValueError("运行记录的资料版本不一致。")
    fingerprint = fingerprints[0]
    if fingerprint is not None and expected_suite_hash is not None and fingerprint != expected_suite_hash:
        raise ValueError("运行记录的资料版本与当前评测集不一致。")

    for row in runs:
        if not isinstance(row, dict):
            raise ValueError("运行记录必须是 JSON 对象。")
        case_id = row.get("id")
        if not isinstance(case_id, str) or case_id not in case_by_id or case_id in seen:
            raise ValueError(f"运行记录 ID 未知或重复：{case_id}")
        seen.add(case_id)
        retrieved = row.get("retrieved_sources")
        cited = row.get("cited_sources")
        if not all(isinstance(values, list) and all(isinstance(value, str) for value in values)
                   for values in (retrieved, cited)):
            raise ValueError(f"{case_id} 的检索/引用来源必须是字符串数组。")
        review = row.get("reviewer_support")
        status = row.get("status")
        if (not isinstance(status, str) or status not in STATUSES or not isinstance(review, (str, type(None)))
                or review not in REVIEWS):
            raise ValueError(f"{case_id} 的状态或人工复核标注无效。")

        case = case_by_id[case_id]
        if case["expected_behavior"] == "answer":
            answerable_total += 1
            expected = set(case["expected_sources"])
            retrieved_set = set(retrieved)
            cited_set = set(cited)
            has_retrieval = expected <= retrieved_set
            has_citation = expected <= cited_set and cited_set <= retrieved_set
            is_answered = row["status"] == "answered"
            retrieval_pass += has_retrieval
            citation_pass += has_citation
            answered_pass += is_answered
            if is_answered:
                if row["reviewer_support"] is None:
                    unreviewed += 1
                else:
                    reviewed += 1
                    supported += row["reviewer_support"] == "supported"
            if not (has_retrieval and has_citation and is_answered):
                failures.append(case_id)
        else:
            abstain_total += 1
            correct = row["status"] == "insufficient_context" and not cited
            abstain_pass += correct
            if not correct:
                failures.append(case_id)

    return {
        "coverage": {"evaluated": len(seen), "total": len(cases)},
        "suite_match": True if fingerprint is not None and expected_suite_hash is not None else None,
        "complete": len(seen) == len(cases),
        "missing_case_ids": sorted(case_by_id.keys() - seen),
        "answerable": {
            "retrieval_complete": _metric(retrieval_pass, answerable_total),
            "citation_complete": _metric(citation_pass, answerable_total),
            "answered": _metric(answered_pass, answerable_total),
        },
        "abstain": {"correct": _metric(abstain_pass, abstain_total)},
        "semantic_support": {"supported": supported, "reviewed": reviewed, "unreviewed": unreviewed},
        "failed_case_ids": failures,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="校验合成制度评测集，或对已有运行记录评分；不会调用模型。")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true", help="只校验样例与证据原文")
    group.add_argument("--runs", type=Path, help="每行一条 JSON 运行记录")
    args = parser.parse_args(argv)

    try:
        cases = load_cases()
        if args.validate_only:
            result = {"cases": len(cases), "categories": dict(Counter(c["category"] for c in cases)),
                      "model_score": None}
        else:
            if not args.runs.is_file():
                parser.error(f"运行记录不存在：{args.runs}")
            result = score_runs(cases, load_runs(args.runs), expected_suite_hash=suite_hash())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
