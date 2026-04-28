from pathlib import Path

from src.config import MAX_FILE_SIZE_MB
from src.logging_utils import get_logger

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
logger = get_logger(__name__)


def _decode_text_file(uploaded_file) -> str:
    uploaded_file.seek(0)
    return uploaded_file.read().decode("utf-8")


def _extract_pdf_text(uploaded_file) -> tuple[str, list[dict]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("读取 PDF 需要安装 pypdf，请先执行 pip install pypdf。") from exc

    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    logger.info("Start extracting PDF text: %s", getattr(uploaded_file, "name", "unknown"))

    pages = []
    full_text_parts = []

    for index, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        pages.append({"page": index, "text": text})
        full_text_parts.append(text)

    if not full_text_parts:
        raise ValueError("PDF 未提取到可用文本，可能是扫描件或图片型 PDF。")

    logger.info("Extracted %s text pages from PDF: %s", len(pages), getattr(uploaded_file, "name", "unknown"))
    return "\n\n".join(full_text_parts), pages


def load_document(uploaded_file) -> dict:
    file_name = uploaded_file.name
    suffix = Path(file_name).suffix.lower()
    logger.info("Load document request: %s (%s)", file_name, suffix or "unknown")

    uploaded_file.seek(0, 2)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"文件 {file_name} 大小为 {file_size / 1024 / 1024:.1f}MB，超过 {MAX_FILE_SIZE_MB}MB 限制。")

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持 {suffix or '该'} 文件类型。")

    if suffix in {".txt", ".md"}:
        text = _decode_text_file(uploaded_file)
        logger.info("Loaded text document: %s, chars=%s", file_name, len(text))
        return {
            "text": text,
            "source": file_name,
            "file_type": suffix.lstrip("."),
            "pages": [],
        }

    text, pages = _extract_pdf_text(uploaded_file)
    logger.info("Loaded PDF document: %s, chars=%s, pages=%s", file_name, len(text), len(pages))
    return {
        "text": text,
        "source": file_name,
        "file_type": "pdf",
        "pages": pages,
    }
