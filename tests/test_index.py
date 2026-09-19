import os
from io import StringIO
from pathlib import Path

import pytest

from jev_rerank.index import Index, read_passages


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    index = Index(tmp_path / "cache" / "index.sqlite3", root, chunk_chars=256)
    yield root, index
    index.close()


def test_streaming_windows_cover_all_text_and_keep_empty_files():
    text = "αβγδε" * 900
    chunks = list(read_passages(StringIO(text), 256, 25))
    assert all(len(chunk) <= 256 for chunk in chunks)
    assert chunks[0] + "".join(chunk[25:] for chunk in chunks[1:]) == text
    assert list(read_passages(StringIO(""), 256, 25)) == [""]


def test_filename_and_late_passage_retrieval(corpus):
    root, index = corpus
    (root / "annual_report_2025.txt").write_text("unrelated body")
    (root / "long.txt").write_text("boring content " * 200 + " needlequartz at the end")
    (root / "other.txt").write_text("a distracting document")
    stats = index.sync(lambda _: None)
    filename = list(index.candidates("annual_report_2025.txt", 1, stats.documents))
    assert filename[0].path == "annual_report_2025.txt"
    late = list(index.candidates("needlequartz", 1, stats.documents))
    assert late[0].path == "long.txt"
    assert late[0].ordinal > 0
    assert "needlequartz" in late[0].text


def test_incremental_refresh_add_change_delete_and_rename(corpus, monkeypatch):
    root, index = corpus
    source = root / "original.txt"
    source.write_text("alpha")
    (root / "deleted.txt").write_text("betabetabeta")
    first = index.sync(lambda _: None)
    assert first.updated == 2

    def unexpected_open(*args, **kwargs):
        pytest.fail("Unchanged files should not be reread")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", unexpected_open)
        assert index.sync(lambda _: None).updated == 0

    source.write_text("changed content")
    source.rename(root / "renamed.txt")
    (root / "deleted.txt").unlink()
    (root / "added.txt").write_text("new content")
    third = index.sync(lambda _: None)
    assert (third.updated, third.removed, third.documents) == (2, 2, 2)
    assert {p.path for p in index.candidates("anything", 0, 2)} == {
        "renamed.txt",
        "added.txt",
    }
    assert (
        index.db.execute(
            "SELECT count(*) FROM search WHERE search MATCH 'betabetabeta'"
        ).fetchone()[0]
        == 0
    )
    assert index.sync(lambda _: None, rebuild=True).updated == 2


def test_invalid_files_roll_back_all_chunks_and_hidden_symlinks_are_ignored(corpus):
    root, index = corpus
    (root / "valid.txt").write_text("")
    (root / "binary.txt").write_bytes(b"x" * 1000 + b"\x00")
    (root / "invalid.txt").write_bytes(b"\xff")
    (root / ".hidden.txt").write_text("hidden")
    (root / "link.txt").symlink_to(root / "valid.txt")
    warnings = []
    stats = index.sync(warnings.append)
    assert (stats.documents, stats.passages, stats.skipped) == (1, 1, 2)
    assert len(warnings) == 2


def test_shortlist_diversifies_files_and_handles_fts_syntax(corpus):
    root, index = corpus
    (root / "many.txt").write_text("needle " * 1000)
    (root / "second.txt").write_text("needle")
    (root / "third.txt").write_text("other")
    stats = index.sync(lambda _: None)
    result = list(index.candidates('"needle" OR : () * -', 2, stats.documents))
    assert {p.path for p in result} == {"many.txt", "second.txt"}
    fallback = list(index.candidates("😀", 2, stats.documents))
    assert len({p.path for p in fallback}) == 2
    assert [p.path for p in fallback] == ["many.txt", "second.txt"]


def test_small_corpus_and_exhaustive_mode_include_all_passages(corpus):
    root, index = corpus
    (root / "long.txt").write_text("a" * 2000)
    stats = index.sync(lambda _: None)
    assert len(list(index.candidates("missing", 10, 1))) == stats.passages
    assert len(list(index.candidates("missing", 0, 1))) == stats.passages


def test_becoming_invalid_removes_old_search_results(corpus):
    root, index = corpus
    source = root / "file.txt"
    source.write_text("previously valid")
    index.sync(lambda _: None)
    source.write_bytes(b"\x00")
    stats = index.sync(lambda _: None)
    assert (stats.documents, stats.skipped, stats.removed) == (0, 1, 1)


def test_failed_scan_preserves_previous_index(corpus, monkeypatch):
    root, index = corpus
    (root / "original.txt").write_text("old content")
    index.sync(lambda _: None)

    def failing_scan(*args):
        raise OSError("Permission denied")

    monkeypatch.setattr("jev_rerank.index.text_files", failing_scan)
    with pytest.raises(OSError):
        index.sync(lambda _: None)
    assert len(list(index.candidates("", 0, 1))) == 1


def test_cache_in_corpus_is_not_itself_indexed(tmp_path):
    (tmp_path / "file.txt").write_text("valid text")
    index = Index(tmp_path / "index.sqlite3", tmp_path)
    try:
        stats = index.sync(lambda message: pytest.fail(message))
        assert stats.documents == 1
    finally:
        index.close()


def test_unreadable_nested_directory_aborts_refresh(corpus, monkeypatch):
    root, index = corpus
    nested = root / "nested"
    nested.mkdir()
    (nested / "document.txt").write_text("content")
    index.sync(lambda _: None)
    scandir = os.scandir

    def blocked_scan(path):
        if Path(path) == nested:
            raise PermissionError("Cannot read nested directory")
        return scandir(path)

    with monkeypatch.context() as patch:
        patch.setattr(os, "scandir", blocked_scan)
        with pytest.raises(OSError, match="nested"):
            index.sync(lambda _: None)
    assert len(list(index.candidates("", 0, 1))) == 1
    assert index.sync(lambda _: None).documents == 1
