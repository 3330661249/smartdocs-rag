"""Optional, explicitly invoked live run over the synthetic policy suite."""

import argparse
import json
import sys
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from evals.policy_eval import CASE_DIR, load_cases, suite_hash
from src.config import get_settings
from src.loader import load_document
from src.qa_chain import generate_answer
from src.splitter import split_document
from src.vectorstore import build_vectorstore, delete_vectorstore, search_similar_chunks, vectorstore_exists


def _build_policy_store(kb_name: str):
    documents = []
    for path in sorted(CASE_DIR.glob("*.md")):
        uploaded = BytesIO(path.read_bytes())
        uploaded.name = path.name
        documents.extend(split_document(load_document(uploaded), chunk_size=500, overlap=80))
    return build_vectorstore(documents, kb_name=kb_name)


def run_live(cases: list[dict], output: Path) -> int:
    """Write an append-safe trace of selected cases and remove only our temporary KB."""
    settings = get_settings()
    kb_name = f"policy_eval_{uuid4().hex}"
    fingerprint = suite_hash()
    completed = 0
    try:
        with output.open("x", encoding="utf-8") as handle:
            store = _build_policy_store(kb_name)
            for case in cases:
                start = perf_counter()
                retrieved_sources = []
                cited_sources = []
                status = "run_error"
                answer = ""
                try:
                    results = search_similar_chunks(store, case["question"], k=3, score_threshold=0.1)
                    docs = [item["doc"] for item in results]
                    retrieved_sources = list(dict.fromkeys(doc.metadata.get("source", "未知") for doc in docs))
                    result = generate_answer(case["question"], docs)
                    status = result["status"]
                    answer = result["answer"]
                    cited_sources = list(dict.fromkeys(item["source"] for item in result["citations"]))
                except Exception:
                    # Exception text may contain request details or credentials.
                    status = "run_error"

                record = {
                    "id": case["id"],
                    "question": case["question"],
                    "status": status,
                    "answer": answer,
                    "retrieved_sources": retrieved_sources,
                    "cited_sources": cited_sources,
                    "reviewer_support": None,
                    "model": settings.zhipu_chat_model,
                    "embedding_model": settings.zhipu_embedding_model,
                    "suite_sha256": fingerprint,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "duration_ms": round((perf_counter() - start) * 1000, 2),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                if status == "run_error":
                    break
    finally:
        if vectorstore_exists(kb_name):
            delete_vectorstore(kb_name)
    return completed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="显式调用已配置的外部模型，在合成制度资料上记录问答轨迹。")
    parser.add_argument("--live", action="store_true", help="确认本次会调用模型与向量化服务")
    parser.add_argument("--output", required=True, type=Path, help="新的 JSONL 输出路径；不覆盖已有文件")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--limit", type=int, default=3, help="最多运行多少条；默认 3 条")
    scope.add_argument("--all", action="store_true", help="运行全部 30 条")
    args = parser.parse_args(argv)

    if not args.live:
        parser.error("必须显式指定 --live 才能调用外部服务。")
    if args.output.exists():
        parser.error(f"输出文件已存在，不会覆盖：{args.output}")
    if not args.output.parent.is_dir():
        parser.error(f"输出目录不存在：{args.output.parent}")

    try:
        cases = load_cases()
        if not args.all and not 1 <= args.limit <= len(cases):
            parser.error(f"--limit 必须在 1 到 {len(cases)} 之间。")
        selected = cases if args.all else cases[:args.limit]
        completed = run_live(selected, args.output)
    except (OSError, ValueError, json.JSONDecodeError):
        parser.error("运行未完成：请检查本地配置、资料或输出路径。")

    print(f"已记录 {completed}/{len(selected)} 条合成案例。运行记录：{args.output}", file=sys.stderr)
    if completed < len(selected):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
