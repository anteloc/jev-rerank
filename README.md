# jev-rerank

A Python CLI for finding the text files that best match a natural-language query,
using [TypeSafe's reranking recipe](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md).
Both filenames and document contents are searchable.

## Install and run

Requires Python 3.11+ and SQLite with FTS5 (included in most Python distributions).

```sh
uv sync
export TYPESAFE_API_KEY='your-key'
uv run jev-rerank --docs ./lyrics --query 'Dua Lipa Hallucinate' --top 5
```

Or install the CLI into your environment:

```sh
python -m pip install .
jev-rerank --docs ./documents --query 'how to recover a lost account' --top 10
```

`python -m jev_rerank` works as well. Obtain an API key from the
[TypeSafe console](https://console.typesafe.ai/).

Output is tab-separated `rank`, `score`, and path relative to `--docs`. Progress
and warnings go to stderr, so stdout can be redirected or piped. Use `--json` for
results plus corpus counts, cache hits, API usage, and elapsed time. JSON results
also include the zero-based index of the winning passage.

```sh
uv run jev-rerank --docs ./documents --query '2025 annual report' --top 10 --json
```

## How it scales

1. Recursively scan non-hidden regular files. UTF-8 and UTF-8-with-BOM text is
   accepted regardless of extension. Binary, invalid UTF-8, and unreadable files
   are skipped with warnings; symlinks and hidden entries are ignored. Empty files
   are indexed so they can match by filename.
2. Stream files into overlapping passages (4,000 characters by default). Store the
   complete text in a persistent SQLite FTS5 index on disk. Later runs check file
   size, modification time, and change time, and reread only changed files. Deleted
   files are removed from the index. There is no silent truncation of long files.
3. For large collections, BM25 selects up to `max(100, 10 * top)` distinct files,
   weighting their relative paths five times as strongly as passage text. Score
   the best matching passage from each shortlisted file. If lexical matches do
   not fill the shortlist, fill remaining slots by alphabetical path.
4. If the collection fits within the shortlist budget, or `--candidates 0` is set,
   score **every passage of every file**. Keep each file's highest passage score,
   so a file occupies only one result slot.
5. A fixed pool of async workers shares one TypeSafe client. The SDK reuses HTTP
   connections and retries transient failures, including rate limits. The tool
   creates only `--concurrency` tasks, even for millions of passages, and retains
   results in a bounded top-k heap. Cached scores avoid repeated API requests.

Each API request has this state:

```json
{
  "query": "2025 annual report",
  "candidate": {
    "filename": "annual-report-2025.txt",
    "path": "finance/annual-report-2025.txt",
    "passage": "...document text...",
    "passage_index": 0
  }
}
```

The question generalizes the recipe's Noul to “does this document match the search
intent?” Its criteria explicitly allow filename/title/author searches and semantic
content matches. The score is TypeSafe's probability of that yes/no judgment,
between 0 and 1, sorted descending. Equal scores are ordered by path. These are
model judgments, not guaranteed relevance measurements.

## Speed and coverage

The default shortlist makes API work independent of corpus size once the index
is built. The initial run still reads the corpus; subsequent runs still scan file
metadata. The index consumes disk space proportional to the text, including
overlap and search structures. Application memory does not hold the entire corpus
or create one task per file; it holds active passages, the shortlist, and top-k
results. SQLite may use temporary disk space for search operations.

Keyword retrieval can miss semantic matches with different wording. In shortlist
mode, only one passage per selected file is judged; the rest of that file is not
sent to TypeSafe. Increase the shortlist or select exhaustive scoring for broader
coverage:

```sh
# Fast shortlist: at most 200 files get semantic judgments.
uv run jev-rerank --docs ./documents --query 'access recovery' --top 10 --candidates 200

# Full semantic coverage, including files without any matching query words.
uv run jev-rerank --docs ./documents --query 'songs about feeling isolated' --top 10 --candidates 0
```

Exhaustive mode makes one request per passage unless cached. It can cost more and
take much longer for a large corpus. Max-passage scoring favors a document with a
strong local match; it does not assess whole-document constraints or combine facts
spread across separate passages. Larger passages can help when context matters.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--docs`, `--query`, `--top` | required | Corpus directory, search intent, result count |
| `--candidates` | `max(100, 10 * top)` | Distinct files to shortlist; `0` scores everything |
| `--concurrency` | `16` | Maximum simultaneous API calls |
| `--model` | `jev-1.13.0` | Model ID; also accepts `TYPESAFE_DEFAULT_MODEL` |
| `--cache-dir` | `$XDG_CACHE_HOME/jev-rerank` or `~/.cache/jev-rerank` | Local text index and score cache |
| `--chunk-chars` | `4000` | Passage size, 256–16000 characters; overlap is 10%, capped at 400 |
| `--timeout` | `60` | SDK request timeout, seconds |
| `--retries` | `4` | SDK retries for transient failures |
| `--rebuild` | off | Reread all files regardless of metadata |
| `--no-score-cache` | off | Bypass cached judgments |
| `--json` | off | Structured output |
| `--quiet` | off | Suppress progress, retain warnings |

Use `jev-rerank --help` for the full CLI. The top count must be positive; an explicit
nonzero candidate count must be at least the top count. Fewer results are returned
if the corpus contains fewer valid files. An empty corpus returns no results.

`TYPESAFE_BASE_URL` overrides the API endpoint; `TYPESAFE_ENDPOINT` is also accepted
for compatibility with the cookbook. Do not put credentials in the URL.

The model version is pinned so cached results remain meaningful. Score cache keys
include the query, question definition, endpoint, model, relative filename/path,
passage index, and content hash. Model aliases may change remotely; use
`--no-score-cache` when fresh alias results are needed. Indexes are isolated by
corpus directory and passage size. The cache holds document text locally and grows
with new query/document pairs; removing the cache directory forces a fresh index
and fresh judgments. A key is required even for a fully cached invocation.

Selected filenames, relative paths, passages, and queries are sent to TypeSafe.
Query length and serialized state are bounded to avoid oversized requests; use
smaller passages if an unusually large state is rejected. No text is silently cut
to fit the API. A permanent API failure returns exit code 1 without emitting a
partial ranking. Completed scores remain cached; Ctrl-C exits with code 130.

## Development

```sh
uv sync
uv run pytest
uv run ruff check .
```

Tests use temporary corpora and a mocked TypeSafe HTTP transport; they make no
external API calls. See the [Python SDK](https://docs.typesafe.ai/sdk/python.md)
and [model documentation](https://docs.typesafe.ai/models.md) for service details.
