import json
import sys
from io import BytesIO
from pathlib import Path

from src.loader import load_document
from src.logging_utils import get_logger
from src.qa_chain import generate_answer
from src.splitter import split_document
from src.vectorstore import build_vectorstore, delete_vectorstore, search_similar_chunks

logger = get_logger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_FILE = BASE_DIR / "evals" / "sample_qa.json"


def _uploaded_file_from_path(path: Path):
    uploaded = BytesIO(path.read_bytes())
    uploaded.name = path.name
    return uploaded


def build_demo_vectorstore():
    documents = []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        document = load_document(_uploaded_file_from_path(path))
        documents.extend(split_document(document, chunk_size=500, overlap=80))
    return build_vectorstore(documents, kb_name="eval_demo_kb")


def main():
    vectorstore = build_demo_vectorstore()
    samples = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

    results = []
    for sample in samples:
        search_results = search_similar_chunks(vectorstore, sample["question"], k=3, score_threshold=0.1)
        docs = [item["doc"] for item in search_results]
        answer_result = generate_answer(sample["question"], docs) if docs else {
            "answer": "",
            "enough_context": False,
            "citations": [],
        }

        answer = answer_result["answer"]
        keyword_hit = any(keyword in answer for keyword in sample["expected_keywords"])
        source_hit = any(
            citation["source"] == sample["expected_source"]
            for citation in answer_result["citations"]
        )

        results.append(
            {
                "question": sample["question"],
                "keyword_hit": keyword_hit,
                "source_hit": source_hit,
                "answer": answer,
            }
        )

    success_count = sum(1 for item in results if item["keyword_hit"] and item["source_hit"])

    logger.info("评估样例数：%d", len(results))
    logger.info("同时命中关键词和引用来源的样例数：%d / %d", success_count, len(results))
    for item in results:
        logger.info("问题：%s | 关键词命中：%s | 来源命中：%s", item["question"], item["keyword_hit"], item["source_hit"])

    try:
        delete_vectorstore("eval_demo_kb")
        logger.info("已清理评估向量库")
    except Exception:
        pass

    if success_count < len(results):
        logger.warning("评估未全部通过: %d/%d 成功", success_count, len(results))
        sys.exit(1)
    else:
        logger.info("评估全部通过: %d/%d 成功", success_count, len(results))
        sys.exit(0)


if __name__ == "__main__":
    main()
