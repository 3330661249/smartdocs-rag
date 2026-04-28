from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.logging_utils import get_logger

logger = get_logger(__name__)


def _page_for_offset(pages: list[dict], start_index: int) -> int | None:
    if not pages:
        return None

    offset = 0
    for page in pages:
        page_text = page["text"]
        next_offset = offset + len(page_text) + 2
        if start_index < next_offset:
            return page["page"]
        offset = next_offset

    return pages[-1]["page"]


def split_document(document: dict, chunk_size: int = 500, overlap: int = 80) -> list[Document]:
    text = document["text"]
    source = document["source"]
    pages = document.get("pages", [])

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    raw_docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source, "file_type": document.get("file_type", "text")}],
    )

    split_docs = []
    for index, raw_doc in enumerate(raw_docs):
        start_index = raw_doc.metadata.get("start_index", 0)
        metadata = {
            "source": source,
            "file_type": document.get("file_type", "text"),
            "chunk_id": index,
            "start_index": start_index,
        }

        page_number = _page_for_offset(pages, start_index)
        if page_number is not None:
            metadata["page"] = page_number

        split_docs.append(
            Document(
                page_content=raw_doc.page_content,
                metadata=metadata,
            )
        )

    logger.info("切分文档: source=%s, chunk_size=%d, overlap=%d, chunks=%d", source, chunk_size, overlap, len(split_docs))
    return split_docs
