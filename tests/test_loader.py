from io import BytesIO

import pytest

from src.loader import load_document


def _make_file(content: bytes, name: str) -> BytesIO:
    f = BytesIO(content)
    f.name = name
    return f


def test_load_txt_file():
    f = _make_file("你好世界".encode("utf-8"), "test.txt")
    result = load_document(f)
    assert result["text"] == "你好世界"
    assert result["source"] == "test.txt"
    assert result["file_type"] == "txt"
    assert result["pages"] == []


def test_load_md_file():
    f = _make_file("# 标题\n内容".encode("utf-8"), "readme.md")
    result = load_document(f)
    assert "标题" in result["text"]
    assert result["file_type"] == "md"


def test_load_unsupported_extension():
    f = _make_file(b"data", "photo.jpg")
    with pytest.raises(ValueError, match="暂不支持"):
        load_document(f)


def test_load_empty_file_name():
    f = _make_file(b"content", "")
    with pytest.raises(ValueError):
        load_document(f)


def test_file_size_limit():
    from src.config import MAX_FILE_SIZE_MB
    big_content = b"x" * (MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
    f = _make_file(big_content, "huge.txt")
    with pytest.raises(ValueError, match="超过"):
        load_document(f)
