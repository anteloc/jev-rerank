"""Incremental SQLite FTS5 index; neither files nor the corpus are loaded whole."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class Passage:
    path: str
    text: str
    ordinal: int
    digest: str


@dataclass
class IndexStats:
    documents: int = 0
    passages: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0


def read_passages(stream: TextIO, size: int, overlap: int) -> Iterator[str]:
    """Read bounded, overlapping windows, preserving every character."""
    buffer = stream.read(size)
    if not buffer:
        yield ""  # Empty files can still match by name.
        return
    while True:
        if "\x00" in buffer:
            raise ValueError("binary content (NUL bytes)")
        yield buffer
        more = stream.read(size - overlap)
        if not more:
            return
        buffer = (buffer[-overlap:] if overlap else "") + more


def text_files(root: Path) -> Iterator[Path]:
    """Stream directory entries; skip hidden entries and all symlinks."""
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name.startswith(".") or entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    yield from text_files(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    yield Path(entry.path)
    except OSError as exc:
        # Do not silently turn an inaccessible subtree into deleted documents.
        raise OSError(f"Cannot scan directory {str(root)!r}: {exc}") from exc


def match_expression(query: str) -> str:
    # Quote tokens so arbitrary user input cannot become FTS query syntax.
    # Bound expression size, while the complete query still goes to TypeSafe.
    tokens = list(dict.fromkeys(re.findall(r"[^\W_]+", query.casefold())))[:64]
    return " OR ".join(f'"{token}"' for token in tokens)


class Index:
    def __init__(self, database: Path, root: Path, chunk_chars: int = 4000):
        self.root = root.resolve()
        self.database = database.resolve()
        self.chunk_chars = chunk_chars
        self.overlap = min(400, chunk_chars // 10)
        database.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(database, timeout=60)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY, path TEXT UNIQUE NOT NULL,
                mtime_ns INTEGER NOT NULL, ctime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                path TEXT NOT NULL, text TEXT NOT NULL,
                ordinal INTEGER NOT NULL, digest TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS passages_document
                ON passages(document_id, ordinal);
            CREATE VIRTUAL TABLE IF NOT EXISTS search USING fts5(
                path, text, content='passages', content_rowid='id',
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TRIGGER IF NOT EXISTS passages_ai AFTER INSERT ON passages BEGIN
                INSERT INTO search(rowid, path, text)
                    VALUES (new.id, new.path, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS passages_ad AFTER DELETE ON passages BEGIN
                INSERT INTO search(search, rowid, path, text)
                    VALUES ('delete', old.id, old.path, old.text);
            END;
            CREATE TABLE IF NOT EXISTS scores (
                key TEXT PRIMARY KEY, score REAL NOT NULL
            );
        """)
        expected = {
            "schema": SCHEMA_VERSION,
            "root": str(self.root),
            "chunk_chars": str(chunk_chars),
        }
        actual = dict(self.db.execute("SELECT key, value FROM metadata"))
        if actual and actual != expected:
            self.db.close()
            raise ValueError("Index settings changed; use another --cache-dir.")
        with self.db:
            self.db.executemany(
                "INSERT OR IGNORE INTO metadata VALUES (?, ?)", expected.items()
            )

    def close(self) -> None:
        self.db.close()

    def sync(self, warn: Callable[[str], None], *, rebuild: bool = False) -> IndexStats:
        stats = IndexStats()
        try:
            # Serialize refreshes; commit the new corpus atomically.
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("CREATE TEMP TABLE seen (id INTEGER PRIMARY KEY)")
            own_files = {
                Path(str(self.database) + suffix)
                for suffix in ("", "-wal", "-shm", "-journal")
            }
            for path in text_files(self.root):
                if path in own_files:
                    continue
                relative = path.relative_to(self.root).as_posix()
                old = self.db.execute(
                    "SELECT * FROM documents WHERE path=?", (relative,)
                ).fetchone()
                self.db.execute("SAVEPOINT document")
                try:
                    stat = path.stat()
                    stamp = (stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)
                    if (
                        not rebuild
                        and old
                        and stamp == (old["mtime_ns"], old["ctime_ns"], old["size"])
                    ):
                        doc_id = old["id"]
                    else:
                        if old:
                            self.db.execute(
                                "DELETE FROM documents WHERE id=?", (old["id"],)
                            )
                        doc_id = self.db.execute(
                            "INSERT INTO documents(path, mtime_ns, ctime_ns, size) "
                            "VALUES (?, ?, ?, ?)",
                            (relative, *stamp),
                        ).lastrowid
                        with path.open(encoding="utf-8-sig") as stream:
                            for ordinal, text in enumerate(
                                read_passages(stream, self.chunk_chars, self.overlap)
                            ):
                                digest = hashlib.sha256(text.encode()).hexdigest()
                                self.db.execute(
                                    "INSERT INTO passages(document_id, path, text, "
                                    "ordinal, digest) VALUES (?, ?, ?, ?, ?)",
                                    (doc_id, relative, text, ordinal, digest),
                                )
                        after = path.stat()
                        if stamp != (
                            after.st_mtime_ns,
                            after.st_ctime_ns,
                            after.st_size,
                        ):
                            raise ValueError("file changed while it was being read")
                        stats.updated += 1
                    self.db.execute("INSERT INTO seen VALUES (?)", (doc_id,))
                except (OSError, UnicodeError, ValueError) as exc:
                    self.db.execute("ROLLBACK TO document")
                    stats.skipped += 1
                    warn(f"Skipping {relative!r}: {exc}")
                finally:
                    self.db.execute("RELEASE document")
            stats.removed = self.db.execute(
                "DELETE FROM documents WHERE id NOT IN (SELECT id FROM seen)"
            ).rowcount
            self.db.execute("DROP TABLE seen")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        stats.documents = self.db.execute("SELECT count(*) FROM documents").fetchone()[
            0
        ]
        stats.passages = self.db.execute("SELECT count(*) FROM passages").fetchone()[0]
        return stats

    @staticmethod
    def _passage(row: sqlite3.Row) -> Passage:
        return Passage(row["path"], row["text"], row["ordinal"], row["digest"])

    def candidates(self, query: str, limit: int, documents: int) -> Iterator[Passage]:
        """All passages in exhaustive mode; one best passage per shortlisted file."""
        if limit == 0 or documents <= limit:
            cursor = self.db.execute("SELECT * FROM passages ORDER BY id")
            for row in cursor:
                yield self._passage(row)
            return

        selected: set[int] = set()
        expression = match_expression(query)
        if expression:
            # Streaming avoids an in-memory sort of the corpus. FTS5's rank
            # column uses its optimized BM25 ordering; filename weight is 5x.
            cursor = self.db.execute(
                "SELECT p.* FROM search JOIN passages p ON p.id=search.rowid "
                "WHERE search MATCH ? AND rank MATCH 'bm25(5.0, 1.0)' "
                "ORDER BY rank",
                (expression,),
            )
            for row in cursor:
                if row["document_id"] in selected:
                    continue
                selected.add(row["document_id"])
                yield self._passage(row)
                if len(selected) >= limit:
                    return
        # Fill sparse/no-keyword shortlists deterministically. This is not a
        # semantic guarantee; callers can request exhaustive scoring with 0.
        cursor = self.db.execute(
            "SELECT p.* FROM documents d JOIN passages p ON p.document_id=d.id "
            "AND p.ordinal=0 ORDER BY d.path"
        )
        for row in cursor:
            if row["document_id"] not in selected:
                selected.add(row["document_id"])
                yield self._passage(row)
                if len(selected) >= limit:
                    return

    def cached_score(self, key: str) -> float | None:
        row = self.db.execute("SELECT score FROM scores WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def save_score(self, key: str, score: float) -> None:
        self.db.execute("INSERT OR REPLACE INTO scores VALUES (?, ?)", (key, score))

    def flush_scores(self) -> None:
        self.db.commit()
