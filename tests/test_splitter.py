from src.splitter import split_document, _page_for_offset


def test_split_basic_text():
    doc = {"text": "这是第一段内容。" * 100, "source": "test.txt", "file_type": "txt", "pages": []}
    chunks = split_document(doc, chunk_size=200, overlap=30)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "test.txt"
        assert "chunk_id" in chunk.metadata
        assert "start_index" in chunk.metadata


def test_split_preserves_metadata():
    doc = {"text": "短文本", "source": "hello.md", "file_type": "md", "pages": []}
    chunks = split_document(doc, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "hello.md"
    assert chunks[0].metadata["file_type"] == "md"
    assert chunks[0].metadata["chunk_id"] == 0


def test_page_for_offset_with_pages():
    pages = [
        {"page": 1, "text": "A" * 100},
        {"page": 2, "text": "B" * 100},
    ]
    assert _page_for_offset(pages, 50) == 1
    assert _page_for_offset(pages, 150) == 2


def test_page_for_offset_empty():
    assert _page_for_offset([], 0) is None


def test_page_for_offset_beyond_last():
    pages = [{"page": 1, "text": "short"}]
    assert _page_for_offset(pages, 9999) == 1


def test_split_with_page_mapping():
    doc = {
        "text": "第一页内容。" * 50 + "\n\n" + "第二页内容。" * 50,
        "source": "doc.pdf",
        "file_type": "pdf",
        "pages": [
            {"page": 1, "text": "第一页内容。" * 50},
            {"page": 2, "text": "第二页内容。" * 50},
        ],
    }
    chunks = split_document(doc, chunk_size=300, overlap=50)
    assert all("page" in c.metadata for c in chunks)
