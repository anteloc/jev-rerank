import asyncio
import hashlib
import json
import random

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

from jev_rerank.index import Index, Passage
from jev_rerank.query import Query
from jev_rerank.rerank import (
    DEFAULT_MODEL,
    DEFAULT_NO,
    DEFAULT_YES,
    RerankError,
    Result,
    TopK,
    rerank,
)


@pytest.fixture
def index(tmp_path):
    cache = Index(tmp_path / "index.sqlite3", tmp_path)
    yield cache
    cache.close()


def passage(path, text="test", ordinal=0):
    return Passage(path, text, ordinal, hashlib.sha256(text.encode()).hexdigest())


def payload(score):
    return {
        "model": DEFAULT_MODEL,
        "answers": {"relevance": {"type": "noul", "noul": score}},
        "usage": {"input_tokens": 25, "output_tokens": 2},
    }


def client_for(handler, **kwargs):
    return AsyncTypeSafeClient(
        api_key="test-key",
        transport=httpx2.MockTransport(handler),
        retry=RetryPolicy(max_retries=kwargs.pop("retries", 0), backoff_initial=0),
        **kwargs,
    )


async def rank(index, client, passages, **kwargs):
    options = dict(
        query=Query("find the document"),
        top=3,
        concurrency=3,
        model=DEFAULT_MODEL,
        endpoint="https://api.typesafe.ai",
        client=client,
        index=index,
    )
    options.update(kwargs)
    return await rerank(passages, **options)


async def test_real_sdk_contract_filename_state_and_cache_invalidation(index):
    requests = []

    def handler(request):
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        requests.append(body)
        assert body["questions"]["relevance"]["type"] == "noul"
        candidate = body["state"]["candidate"]
        assert candidate["filename"] == "needle.txt"
        assert candidate["path"] == "folder/needle.txt"
        return httpx2.Response(200, json=payload(0.9))

    passages = [passage("folder/needle.txt")]
    async with client_for(handler) as client:
        results, stats = await rank(index, client, passages)
        assert results[0].score == 0.9
        assert (stats.api_calls, stats.input_tokens) == (1, 25)
        _, stats = await rank(index, client, passages)
        assert (stats.cache_hits, stats.api_calls) == (1, 0)
        await rank(index, client, passages, query=Query("different query"))
        await rank(index, client, passages, model="another-model")
        await rank(index, client, passages, endpoint="https://another-endpoint.test")
        await rank(index, client, [passage("folder/needle.txt", "new text")])
        await rank(index, client, passages, use_cache=False)
        assert len(requests) == 6


async def test_concurrency_is_bounded_input_is_lazy_and_files_are_unique(index):
    active = peak = produced = finished = 0

    def passages():
        nonlocal produced
        for i in range(80):
            produced += 1
            assert produced - finished <= 4
            yield passage(f"{i // 2:03}.txt", str(i), i % 2)

    async def handler(request):
        nonlocal active, peak, finished
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.001)
        i = int(json.loads(request.content)["state"]["candidate"]["passage"])
        active -= 1
        finished += 1
        return httpx2.Response(200, json=payload(i / 100))

    async with client_for(handler) as client:
        results, stats = await rank(index, client, passages(), concurrency=4, top=5)
    assert peak == 4
    assert stats.scored == 80
    assert [r.path for r in results] == [f"{i:03}.txt" for i in range(39, 34, -1)]
    assert all(r.passage == 1 for r in results)


def test_top_k_matches_full_sort_with_duplicates_and_ties():
    rng = random.Random(37)
    best = {}
    top = TopK(7)
    for i in range(5000):
        result = Result(f"{rng.randrange(100):03}.txt", rng.randrange(10) / 10, i)
        previous = best.get(result.path)
        if previous is None or previous < result:
            best[result.path] = result
        top.add(result)
        assert len(top.heap) <= 14
    assert top.results() == sorted(best.values(), key=lambda r: (-r.score, r.path))[:7]


async def test_sdk_retries_rate_limit(index):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx2.Response(
                429, headers={"Retry-After": "0"}, json={"error": "busy"}
            )
        return httpx2.Response(200, json=payload(0.8))

    async with client_for(handler, retries=1) as client:
        results, stats = await rank(index, client, [passage("file.txt")])
    assert attempts == 2
    assert results[0].score == 0.8
    assert stats.api_calls == 1


async def test_failure_cancels_inflight_work_and_preserves_completed_cache(index):
    cancelled = asyncio.Event()

    async def handler(request):
        name = json.loads(request.content)["state"]["candidate"]["filename"]
        if name == "good.txt":
            return httpx2.Response(200, json=payload(0.8))
        if name == "bad.txt":
            await asyncio.sleep(0.01)
            return httpx2.Response(401, json={"error": "invalid key"})
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with client_for(handler) as client:
        with pytest.raises(RerankError, match="bad.txt"):
            await rank(
                index,
                client,
                [passage(name) for name in ("good.txt", "bad.txt", "slow.txt")],
            )
        assert cancelled.is_set()
        _, stats = await rank(index, client, [passage("good.txt")])
        assert stats.cache_hits == 1


async def test_oversized_state_fails_before_request(index):
    def handler(request):
        pytest.fail("Oversized state must not be sent")

    async with client_for(handler) as client:
        with pytest.raises(RerankError, match="smaller --chunk-chars"):
            await rank(index, client, [passage("file.txt", "文" * 15000)])


async def test_query_criteria_replace_the_defaults_and_key_the_cache(index):
    questions = []

    def handler(request):
        questions.append(json.loads(request.content)["questions"]["relevance"])
        return httpx2.Response(200, json=payload(0.7))

    passages = [passage("file.txt")]
    romantic = Query("missing love", "romantic loss", "love for books")
    async with client_for(handler) as client:
        await rank(index, client, passages)
        await rank(index, client, passages, query=romantic)
        # Identical criteria reuse the cached judgment; any change re-asks.
        _, stats = await rank(index, client, passages, query=romantic)
        assert stats.cache_hits == 1
        equal = Query("missing love", "romantic loss", "love for books")
        _, stats = await rank(index, client, passages, query=equal)
        assert stats.cache_hits == 1
        different = Query("missing love", "romantic loss", "love for hats")
        await rank(index, client, passages, query=different)
    assert len(questions) == 3
    assert questions[0]["criteria"] == {"true": DEFAULT_YES, "false": DEFAULT_NO}
    assert questions[1]["criteria"] == {
        "true": "romantic loss",
        "false": "love for books",
    }
    assert questions[2]["criteria"] == {
        "true": "romantic loss",
        "false": "love for hats",
    }
    # Only the criteria change; the instructions stay the general ranking question.
    assert len({question["instructions"] for question in questions}) == 1


async def test_one_question_serves_both_files_and_database_records(index):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=payload(0.5))

    source = {"type": "sqlite", "table": "songs", "field": "lyrics", "key": {"id": 1}}
    record = Passage("sqlite://songs/lyrics?key=1", "words", 0, "digest", source)
    async with client_for(handler) as client:
        await rank(index, client, [passage("folder/a.txt"), record])
    file_body, record_body = sorted(
        bodies, key=lambda body: "table" in body["state"]["candidate"]
    )
    assert file_body["questions"] == record_body["questions"]
    assert file_body["state"]["candidate"]["filename"] == "a.txt"
    assert "filename" not in record_body["state"]["candidate"]
    assert record_body["state"]["candidate"]["table"] == "songs"
