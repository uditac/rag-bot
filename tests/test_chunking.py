import pytest
from chunking import chunk_text, load_documents


def test_empty_text_gives_no_chunks():
    assert chunk_text("   ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("hello world", size=100, overlap=10) == ["hello world"]


def test_chunks_respect_size_and_overlap():
    text = "a" * 250
    chunks = chunk_text(text, size=100, overlap=20)
    assert all(len(c) <= 100 for c in chunks)
    assert len(chunks) == 3
    assert "".join(chunks).count("a") >= 250  # nothing lost


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        chunk_text("text", size=10, overlap=10)


def test_load_documents_reads_txt_and_md(tmp_path):
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "c.pdf").write_text("ignored", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert [name for name, _ in docs] == ["a.txt", "b.md"]


def test_load_documents_missing_folder():
    with pytest.raises(FileNotFoundError):
        load_documents("does-not-exist")
