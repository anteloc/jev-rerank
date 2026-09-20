import json
import sqlite3
from contextlib import closing

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient

from jev_rerank.cli import default_candidates, main
from jev_rerank.rerank import DEFAULT_MODEL


@pytest.mark.parametrize("directory_option", ["--dir", "--docs"])
def test_cli_json_end_to_end(tmp_path, monkeypatch, capsys, directory_option):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "target.txt").write_text("relevant content")
    (docs / "other.txt").write_text("unrelated content")
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        score = 0.95 if body["state"]["candidate"]["filename"] == "target.txt" else 0.1
        return httpx2.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "answers": {"relevance": {"type": "noul", "noul": score}},
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        "jev_rerank.cli.AsyncTypeSafeClient",
        lambda **kwargs: AsyncTypeSafeClient(
            transport=httpx2.MockTransport(handler), **kwargs
        ),
    )
    args = [
        directory_option,
        str(docs),
        "--query",
        "target",
        "--top",
        "1",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--json",
    ]
    assert main(args) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["results"][0]["path"] == "target.txt"
    assert result["query"] == "target"
    assert "criteria" not in result
    assert result["stats"]["api_calls"] == 2
    assert "Indexed 2 files" in output.err
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["stats"]["cache_hits"] == 2
    assert second["stats"]["updated"] == 0
    assert len(requests) == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--top", "0"],
        ["--top", "-1"],
        ["--query", " "],
        ["--candidates", "1", "--top", "2"],
        ["--concurrency", "0"],
        ["--timeout", "nan"],
        ["--chunk-chars", "10"],
    ],
)
def test_invalid_arguments(tmp_path, monkeypatch, extra):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        main(["--docs", str(tmp_path), "--query", "test", "--top", "1", *extra])
    assert exc.value.code == 2


def test_missing_key_is_actionable(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["--docs", str(tmp_path), "--query", "test", "--top", "1"])
    assert exc.value.code == 2
    assert "Set TYPESAFE_API_KEY" in capsys.readouterr().err


def test_empty_directory(tmp_path, monkeypatch, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert (
        main(
            [
                "--docs",
                str(docs),
                "--query",
                "x",
                "--top",
                "1",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["results"] == []
    assert output["stats"]["api_calls"] == 0


def test_database_cli_ranks_union_and_reuses_cache(tmp_path, monkeypatch, capsys):
    database = tmp_path / "records.db"
    with closing(sqlite3.connect(database)) as db, db:
        db.executescript("""
            CREATE TABLE songs(id INTEGER PRIMARY KEY, title VARCHAR, lyrics CLOB);
            CREATE TABLE notes(id TEXT PRIMARY KEY, body TEXT);
            INSERT INTO songs VALUES (1, 'Ocean', 'Sail across the sea');
            INSERT INTO notes VALUES ('note-a', 'Oceans contain saltwater');
        """)
    original = database.read_bytes()
    requests = []
    scores = {("songs", "title"): 0.5, ("songs", "lyrics"): 0.9, ("notes", "body"): 0.7}

    def handler(request):
        body = json.loads(request.content)
        candidate = body["state"]["candidate"]
        requests.append(candidate)
        assert "filename" not in candidate
        assert candidate["type"] == "sqlite"
        assert candidate["database"] == str(database)
        assert candidate["key"] in ({"id": 1}, {"id": "note-a"})
        score = scores[candidate["table"], candidate["field"]]
        return httpx2.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "answers": {"relevance": {"type": "noul", "noul": score}},
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        "jev_rerank.cli.AsyncTypeSafeClient",
        lambda **kwargs: AsyncTypeSafeClient(
            transport=httpx2.MockTransport(handler), **kwargs
        ),
    )
    base = [
        "--db",
        str(database),
        "--query",
        "ocean",
        "--top",
        "3",
        "--cache-dir",
        str(tmp_path / "cache"),
        "--json",
    ]
    selections = [
        "--table-field",
        "songs.title",
        "--table-field",
        "songs.lyrics",
        "--table-field",
        "notes.body",
    ]
    assert main(base + selections) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["db"] == str(database)
    assert "docs" not in output
    assert [r["source"]["field"] for r in output["results"]] == [
        "lyrics",
        "body",
        "title",
    ]
    assert output["stats"]["api_calls"] == 3
    assert main(base + selections + ["--table-field", "SONGS.TITLE"]) == 0
    cached = json.loads(capsys.readouterr().out)
    assert cached["stats"]["cache_hits"] == 3
    assert cached["stats"]["updated"] == 0
    assert len(requests) == 3
    assert database.read_bytes() == original


@pytest.mark.parametrize(
    "options, message",
    [
        (["--db", "{db}"], "requires at least one --table-field"),
        (["--dir", "{dir}", "--db", "{db}"], "not allowed with argument"),
        (["--dir", "{dir}", "--table-field", "t.body"], "--table-field requires --db"),
        (
            ["--db", "{db}", "--table-field", "malformed"],
            "expected TABLE_NAME.FIELD_NAME",
        ),
    ],
)
def test_invalid_source_options(tmp_path, monkeypatch, capsys, options, message):
    database = tmp_path / "data.db"
    with closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE t (body TEXT)")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    args = [option.format(db=database, dir=tmp_path) for option in options]
    with pytest.raises(SystemExit) as exc:
        main(args + ["--query", "test", "--top", "1"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_invalid_database_field_exits_without_partial_results(
    tmp_path, monkeypatch, capsys
):
    database = tmp_path / "data.db"
    with closing(sqlite3.connect(database)) as db:
        db.execute("CREATE TABLE t (body INTEGER)")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    assert (
        main(
            [
                "--db",
                str(database),
                "--table-field",
                "t.body",
                "--query",
                "x",
                "--top",
                "1",
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 1
    )
    output = capsys.readouterr()
    assert output.out == ""
    assert "declared text type" in output.err


@pytest.mark.parametrize("source_kind", ["file", "sqlite"])
@pytest.mark.parametrize(
    "value", ["", "Intro café\n" + 'Line\t"quoted" 🦊\n' * 40 + "MATCH"]
)
def test_show_full_value_in_both_output_formats_from_cached_scores(
    tmp_path, monkeypatch, capsys, source_kind, value
):
    if source_kind == "file":
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "target.txt").write_text(value, encoding="utf-8")
        source_args = ["--dir", str(docs)]
    else:
        database = tmp_path / "data.db"
        with closing(sqlite3.connect(database)) as db, db:
            db.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, body CLOB)")
            db.execute("INSERT INTO records VALUES (1, ?)", (value,))
        source_args = ["--db", str(database), "--table-field", "records.body"]

    requests = []

    def handler(request):
        candidate = json.loads(request.content)["state"]["candidate"]
        requests.append(candidate)
        score = 0.9 if "MATCH" in candidate["passage"] else 0.1
        return httpx2.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "answers": {"relevance": {"type": "noul", "noul": score}},
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        "jev_rerank.cli.AsyncTypeSafeClient",
        lambda **kwargs: AsyncTypeSafeClient(
            transport=httpx2.MockTransport(handler), **kwargs
        ),
    )
    args = source_args + [
        "--query",
        "MATCH",
        "--top",
        "1",
        "--chunk-chars",
        "256",
        "--cache-dir",
        str(tmp_path / "cache"),
    ]
    assert main(args + ["--json"]) == 0
    original = json.loads(capsys.readouterr().out)
    assert "text" not in original["results"][0]
    if value:
        assert original["results"][0]["passage"] > 0
    request_count = len(requests)

    assert main(args + ["--show", "--json"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["results"][0] == {**original["results"][0], "text": value}
    assert shown["stats"]["api_calls"] == 0
    assert shown["stats"]["cache_hits"] == request_count

    assert main(args + ["--show"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    columns = lines[0].split("\t")
    assert len(columns) == 4
    assert json.loads(columns[3]) == value

    assert main(args) == 0
    assert capsys.readouterr().out.strip() == "\t".join(columns[:3])
    assert len(requests) == request_count


# The same intent written both ways; each must reach TypeSafe identically.
PIPE_QUERY = (
    "a song about\nmissing love\n"
    "| yes: the song talks about missing romantic love\n"
    "| no: the song talks about something non-romantic"
)
JSON_QUERY = json.dumps(
    {
        "query": "a song about\nmissing love",
        "yes": "the song talks about missing romantic love",
        "no": "the song talks about something non-romantic",
    }
)


@pytest.mark.parametrize("query", [PIPE_QUERY, JSON_QUERY])
def test_query_criteria_reach_the_api_and_the_json_output(
    tmp_path, monkeypatch, capsys, query
):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "ballad.txt").write_text("I miss you every night")
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(
            200,
            json={
                "model": DEFAULT_MODEL,
                "answers": {"relevance": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 20, "output_tokens": 1},
            },
        )

    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(
        "jev_rerank.cli.AsyncTypeSafeClient",
        lambda **kwargs: AsyncTypeSafeClient(
            transport=httpx2.MockTransport(handler), **kwargs
        ),
    )
    assert (
        main(
            [
                "--dir",
                str(docs),
                "--query",
                query,
                "--top",
                "1",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--json",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["query"] == "a song about\nmissing love"
    assert output["criteria"] == {
        "yes": "the song talks about missing romantic love",
        "no": "the song talks about something non-romantic",
    }
    # The criteria replace the Noul's defaults; the query alone is the state.
    assert bodies[0]["state"]["query"] == "a song about\nmissing love"
    assert bodies[0]["questions"]["relevance"]["criteria"] == {
        "true": "the song talks about missing romantic love",
        "false": "the song talks about something non-romantic",
    }


@pytest.mark.parametrize(
    "query, message",
    [
        ("a song | yes: romantic only", "both a 'yes' and a 'no'"),
        ('{"query": "a song", "nope": "x"}', "Unknown --query field"),
        ("a song | roll", "must start with 'yes:' or 'no:'"),
    ],
)
def test_invalid_query_is_rejected_with_an_actionable_message(
    tmp_path, monkeypatch, capsys, query, message
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    with pytest.raises(SystemExit) as exc:
        main(["--dir", str(tmp_path), "--query", query, "--top", "1"])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_default_candidates_scores_medium_corpora_exhaustively():
    """The floor must clear a few hundred documents before BM25 shortlists."""
    assert default_candidates(1) == 500
    assert default_candidates(10) == 500
    assert default_candidates(80) == 800
