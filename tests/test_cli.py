import json

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient

from jev_rerank.cli import main
from jev_rerank.rerank import DEFAULT_MODEL


def test_cli_json_end_to_end(tmp_path, monkeypatch, capsys):
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
        "--docs",
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
