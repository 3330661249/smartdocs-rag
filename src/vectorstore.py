import json
import re
import shutil
from datetime import datetime
from pathlib import Path

from langchain_community.vectorstores import Chroma

from src.config import VECTORSTORE_ROOT, get_embeddings
from src.logging_utils import get_logger

logger = get_logger(__name__)

METADATA_FILE = "kb_meta.json"


def normalize_kb_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z一-鿿_-]+", "_", name).strip("_")
    return cleaned or "knowledge_base"


def normalize_collection_name(name: str) -> str:
    ascii_only = re.sub(r"[^0-9A-Za-z._-]+", "_", name)
    ascii_only = re.sub(r"_+", "_", ascii_only).strip("._-")
    if not ascii_only:
        ascii_only = "kb"
    if len(ascii_only) < 3:
        ascii_only = f"{ascii_only}_kb"
    return ascii_only[:63].rstrip("._-") or "kb_001"


def get_vectorstore_path(kb_name: str) -> Path:
    VECTORSTORE_ROOT.mkdir(exist_ok=True)
    return VECTORSTORE_ROOT / normalize_kb_name(kb_name)


def _metadata_path(kb_name: str) -> Path:
    return get_vectorstore_path(kb_name) / METADATA_FILE


def _version_directory(kb_name: str, collection_name: str) -> Path:
    return get_vectorstore_path(kb_name) / "versions" / collection_name


def vectorstore_exists(kb_name: str) -> bool:
    return get_vectorstore_path(kb_name).exists()


def _build_metadata(kb_name: str, documents: list) -> dict:
    normalized_name = normalize_kb_name(kb_name)
    collection_base = normalize_collection_name(normalized_name)
    collection_name = f"{collection_base}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    persist_directory = _version_directory(kb_name, collection_name)
    return {
        "kb_name": normalized_name,
        "collection_name": collection_name,
        "persist_directory": str(persist_directory),
        "document_count": len({doc.metadata.get("source", "未知") for doc in documents}),
        "chunk_count": len(documents),
        "sources": sorted({doc.metadata.get("source", "未知") for doc in documents}),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _write_vectorstore_metadata(kb_name: str, meta: dict):
    metadata_path = _metadata_path(kb_name)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_vectorstore_metadata(kb_name: str) -> dict | None:
    meta_path = _metadata_path(kb_name)
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def list_vectorstores() -> list[str]:
    if not VECTORSTORE_ROOT.exists():
        return []

    return sorted(
        item.name
        for item in VECTORSTORE_ROOT.iterdir()
        if item.is_dir()
    )


def build_vectorstore(documents, kb_name: str, overwrite: bool = False):
    if not documents:
        raise ValueError("documents 为空，无法构建向量库。")

    kb_root = get_vectorstore_path(kb_name)
    if kb_root.exists() and not overwrite:
        raise ValueError(f"知识库 {normalize_kb_name(kb_name)} 已存在，请确认是否覆盖。")

    meta = _build_metadata(kb_name, documents)
    persist_directory = Path(meta["persist_directory"])
    persist_directory.mkdir(parents=True, exist_ok=True)

    logger.info("构建向量库: kb_name=%s, chunks=%d, overwrite=%s", normalize_kb_name(kb_name), len(documents), overwrite)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        persist_directory=str(persist_directory),
        collection_name=meta["collection_name"],
    )
    _write_vectorstore_metadata(kb_name, meta)
    logger.info("向量库构建完成: %s", meta["collection_name"])
    return vectorstore


def load_vectorstore(kb_name: str):
    kb_root = get_vectorstore_path(kb_name)
    if not kb_root.exists():
        raise ValueError(f"知识库 {kb_name} 不存在。")

    meta = get_vectorstore_metadata(kb_name)
    if meta and meta.get("persist_directory"):
        persist_directory = meta["persist_directory"]
        collection_name = meta["collection_name"]
    else:
        persist_directory = str(kb_root)
        collection_name = normalize_kb_name(kb_name)

    logger.info("加载向量库: kb_name=%s, collection=%s", kb_name, collection_name)
    return Chroma(
        collection_name=collection_name,
        embedding_function=get_embeddings(),
        persist_directory=persist_directory,
    )


def delete_vectorstore(kb_name: str):
    persist_directory = get_vectorstore_path(kb_name)
    if not persist_directory.exists():
        raise ValueError(f"知识库 {kb_name} 不存在。")
    logger.info("删除向量库: kb_name=%s", kb_name)
    shutil.rmtree(persist_directory)


def search_similar_chunks(vectorstore, query, k=3, score_threshold=0.3, allowed_sources: list[str] | None = None):
    results = []
    fetch_k = max(k * 5, k)

    def source_allowed(doc) -> bool:
        if not allowed_sources:
            return True
        return doc.metadata.get("source") in allowed_sources

    if hasattr(vectorstore, "similarity_search_with_relevance_scores"):
        pairs = vectorstore.similarity_search_with_relevance_scores(query, k=fetch_k)
        for doc, score in pairs:
            if score >= score_threshold and source_allowed(doc):
                results.append({"doc": doc, "score": score})
                if len(results) >= k:
                    break
    else:
        docs = vectorstore.similarity_search(query, k=fetch_k)
        for doc in docs:
            if source_allowed(doc):
                results.append({"doc": doc, "score": None})
            if len(results) >= k:
                break

    logger.info("检索: query=%r, k=%d, threshold=%.2f, results=%d", query[:50], k, score_threshold, len(results))
    return results[:k]
