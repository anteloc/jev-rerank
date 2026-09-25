import json
import sqlite3
from contextlib import closing

import pytest

from jev_rerank.index import Index
from jev_rerank.sqlite_source import SQLiteSource


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "records.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE songs (
                id INTEGER PRIMARY KEY, title VARCHAR(200), lyrics CLOB
            );
            CREATE TABLE articles (id INTEGER PRIMARY KEY, body TEXT);
            INSERT INTO songs VALUES (1, 'Ocean', 'Sailing across the sea');
            INSERT INTO songs VALUES (2, 'Forest', NULL);
            INSERT INTO articles VALUES (1, 'Trees and woodlands');
        """)
    return path


def sync(index, database, fields, **kwargs):
    warnings = []
    with SQLiteSource(database, fields) as source:
        stats = index.sync_records(source.records(), warnings.append, **kwargs)
    return stats, warnings


@pytest.fixture
def index(database, tmp_path):
    index = Index(tmp_path / "cache.sqlite3", database, chunk_chars=256)
    yield index
    index.close()


def test_union_preserves_each_field_and_row_and_deduplicates_selectors(database):
    with SQLiteSource(
        database, ["songs.title", "songs.lyrics", "articles.body", "SONGS.TITLE"]
    ) as source:
        assert source.table_fields == ["articles.body", "songs.lyrics", "songs.title"]
        records = list(source.records())
    assert len(records) == 5
    assert len({record.path for record in records}) == 5
    assert {record.value for record in records} == {
        "Ocean",
        "Sailing across the sea",
        "Forest",
        "Trees and woodlands",
        None,
    }
    assert all(record.source["key"]["id"] in (1, 2) for record in records)


@pytest.mark.parametrize(
    "declared",
    ["TEXT", "CLOB", "VARCHAR(100)", "NVARCHAR(20)", "CHAR(10)", "CHARACTER(50)"],
)
def test_text_types_are_accepted(tmp_path, declared):
    path = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(f"CREATE TABLE data (value {declared})")
        db.execute("INSERT INTO data VALUES ('hello')")
    with SQLiteSource(path, ["data.value"]) as source:
        assert next(source.records()).value == "hello"


@pytest.mark.parametrize(
    "declared", ["INTEGER", "BLOB", "REAL", "NUMERIC", "", "CHARINT", "STRING"]
)
def test_non_text_declared_types_are_rejected(tmp_path, declared):
    path = tmp_path / "db.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute(f"CREATE TABLE data (value {declared})")
    with pytest.raises(ValueError, match="declared text type"):
        SQLiteSource(path, ["data.value"])


@pytest.mark.parametrize(
    "field, message",
    [
        ("missing.body", "Table"),
        ("songs.missing", "Field"),
        ("songs.title; DROP TABLE songs;--", "Field"),
        ("songs", "expected"),
        ("songs.", "expected"),
        ("main.songs.title", "expected"),
    ],
)
def test_invalid_selections_fail_without_modifying_source(database, field, message):
    original = database.read_bytes()
    with pytest.raises(ValueError, match=message):
        SQLiteSource(database, [field])
    assert database.read_bytes() == original


def test_source_is_read_only_and_missing_database_is_not_created(database, tmp_path):
    with SQLiteSource(database, ["songs.title"]) as source:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            source.db.execute("DELETE FROM songs")
        assert len(list(source.records())) == 2
    absent = tmp_path / "absent.db"
    with pytest.raises(ValueError, match="existing"):
        SQLiteSource(absent, ["songs.title"])
    assert not absent.exists()


def test_incremental_index_refresh_handles_changes_deletions_and_nulls(database, index):
    fields = ["songs.title", "songs.lyrics", "articles.body"]
    first, _ = sync(index, database, fields)
    assert (first.documents, first.updated, first.skipped) == (4, 4, 1)
    second, _ = sync(index, database, fields)
    assert second.updated == 0
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("UPDATE songs SET title='newtitle', lyrics=NULL WHERE id=1")
        db.execute("DELETE FROM songs WHERE id=2")
        db.execute("INSERT INTO articles VALUES (2, 'new article')")
    third, _ = sync(index, database, fields)
    assert (third.documents, third.updated, third.removed) == (3, 2, 2)
    passages = list(index.candidates("newtitle", 1, third.documents))
    assert passages[0].text == "newtitle"
    assert passages[0].source["table"] == "songs"
    assert passages[0].source["field"] == "title"
    assert passages[0].source["key"] == {"id": 1}
    assert (
        index.db.execute(
            "SELECT count(*) FROM search WHERE search MATCH 'Sailing'"
        ).fetchone()[0]
        == 0
    )
    rebuilt, _ = sync(index, database, fields, rebuild=True)
    assert rebuilt.updated == 3


def test_non_text_values_in_text_column_are_skipped_and_empty_text_is_kept(
    database, index
):
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("INSERT INTO articles VALUES (2, ?)", (b"binary",))
        db.execute("INSERT INTO articles VALUES (3, '')")
        db.execute("INSERT INTO articles VALUES (4, NULL)")
    stats, warnings = sync(index, database, ["articles.body"])
    assert (stats.documents, stats.skipped) == (2, 2)
    assert len(warnings) == 1


def test_record_long_text_is_chunked_and_late_matches_are_retrieved(database, index):
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("UPDATE articles SET body=?", ("filler " * 1000 + "needlequartz",))
    stats, _ = sync(index, database, ["songs.lyrics", "articles.body"])
    result = next(index.candidates("needlequartz", 1, stats.documents))
    assert result.source["table"] == "articles"
    assert result.ordinal > 0
    assert "needlequartz" in result.text


def test_composite_key_without_rowid_and_quoted_identifiers(tmp_path):
    path = tmp_path / "source?#.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE "odd table" (
                a TEXT, b INTEGER, "some""field" TEXT,
                PRIMARY KEY(a, b)
            ) WITHOUT ROWID;
            INSERT INTO "odd table" VALUES ('record', 7, 'find me');
        """)
    with SQLiteSource(path, ['odd table.some"field']) as source:
        record = next(source.records())
    assert record.value == "find me"
    assert record.source["key"] == {"a": "record", "b": 7}


def test_rowid_fallback_with_nullable_primary_key_and_shadowed_rowid(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE data (id TEXT PRIMARY KEY, rowid TEXT, body TEXT);
            INSERT INTO data VALUES (NULL, 'shadow', 'same');
            INSERT INTO data VALUES (NULL, 'shadow', 'same');
        """)
    with SQLiteSource(path, ["data.body"]) as source:
        records = list(source.records())
    assert records[0].path != records[1].path
    assert records[0].source["key"] == {"_rowid_": 1}
    assert records[1].source["key"] == {"_rowid_": 2}


def test_blob_primary_key_is_json_serializable(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE data (id BLOB PRIMARY KEY, body TEXT)")
        db.execute("INSERT INTO data VALUES (?, 'hello')", (b"\x00\xff",))
    with SQLiteSource(path, ["data.body"]) as source:
        record = next(source.records())
    assert record.source["key"] == {"id": {"blob_hex": "00ff"}}
    json.dumps(record.source)


def test_view_without_natural_key_uses_row_number_as_synthetic_key(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE parts (part TEXT PRIMARY KEY, description TEXT);
            INSERT INTO parts VALUES ('3001', 'Brick 2x4');
            INSERT INTO parts VALUES ('3002', 'Brick 2x3');
            CREATE VIEW parts_jev (full_description) AS
                SELECT part || '|' || description AS full_description FROM parts;
        """)
    with SQLiteSource(path, ["parts_jev.full_description"]) as source:
        records = list(source.records())
    assert {r.value for r in records} == {"3001|Brick 2x4", "3002|Brick 2x3"}
    assert {r.source["key"]["_row_number_"] for r in records} == {1, 2}
    assert len({r.path for r in records}) == 2
    json.dumps([r.source for r in records])


def test_table_with_no_accessible_identity_is_rejected(tmp_path):
    path = tmp_path / "source.sqlite"
    with closing(sqlite3.connect(path)) as db, db:
        db.executescript("""
            CREATE TABLE data (rowid TEXT, "_rowid_" TEXT, oid TEXT, body TEXT);
        """)
    with pytest.raises(ValueError, match="needs a primary key or accessible rowid"):
        SQLiteSource(path, ["data.body"])


def test_scan_failure_rolls_back_index(database, index):
    sync(index, database, ["songs.title"])
    previous = list(index.candidates("", 0, 2))

    def interrupted():
        with SQLiteSource(database, ["articles.body"]) as source:
            yield from source.records()
        raise sqlite3.OperationalError("Source read failed")

    with pytest.raises(sqlite3.OperationalError):
        index.sync_records(interrupted(), lambda _: None)
    assert list(index.candidates("", 0, 2)) == previous


def test_wal_changes_are_seen_without_relying_on_main_database_mtime(database, index):
    with closing(sqlite3.connect(database)) as writer:
        writer.execute("PRAGMA journal_mode=WAL")
        sync(index, database, ["songs.title"])
        with writer:
            writer.execute("UPDATE songs SET title='updated in WAL' WHERE id=1")
        updated, _ = sync(index, database, ["songs.title"])
        assert updated.updated == 1
        assert "updated in WAL" in {p.text for p in index.candidates("", 0, 2)}
