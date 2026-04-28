import pytest
from src.vectorstore import normalize_kb_name, normalize_collection_name, vectorstore_exists, list_vectorstores


def test_normalize_kb_name_chinese():
    assert normalize_kb_name("我的知识库") == "我的知识库"


def test_normalize_kb_name_special_chars():
    result = normalize_kb_name("my kb @# v2!")
    assert "@" not in result
    assert "#" not in result


def test_normalize_kb_name_empty():
    assert normalize_kb_name("") == "knowledge_base"


def test_normalize_kb_name_underscores():
    assert normalize_kb_name("my___kb") == "my___kb"


def test_normalize_collection_name_ascii():
    result = normalize_collection_name("test_kb")
    assert result == "test_kb"


def test_normalize_collection_name_chinese():
    result = normalize_collection_name("我的库")
    assert len(result) >= 3
    assert all(c.isascii() or c == "_" for c in result)


def test_normalize_collection_name_too_short():
    result = normalize_collection_name("a")
    assert len(result) >= 3


def test_normalize_collection_name_too_long():
    result = normalize_collection_name("a" * 100)
    assert len(result) <= 63


def test_vectorstore_exists_nonexistent():
    assert vectorstore_exists("nonexistent_kb_12345") is False


def test_list_vectorstores_no_dir(tmp_path, monkeypatch):
    import src.vectorstore as vs
    monkeypatch.setattr(vs, "VECTORSTORE_ROOT", tmp_path / "nonexistent")
    assert list_vectorstores() == []


def test_list_vectorstores_with_dirs(tmp_path, monkeypatch):
    import src.vectorstore as vs
    kb_dir = tmp_path / "vectorstores"
    kb_dir.mkdir()
    (kb_dir / "kb_one").mkdir()
    (kb_dir / "kb_two").mkdir()
    monkeypatch.setattr(vs, "VECTORSTORE_ROOT", kb_dir)
    result = list_vectorstores()
    assert result == ["kb_one", "kb_two"]
