# jev-rerank

A Python CLI for finding text files or SQLite field values that match a natural-language query,
using [TypeSafe's reranking recipe](https://docs.typesafe.ai/cookbooks/rerank_typesafe.md).
Both filenames and document contents are searchable. Selected SQLite text fields
can also be searched together in one ranking.

## Install and run

Requires Python 3.11+ and SQLite with FTS5 (included in most Python distributions).

```sh
uv sync
export TYPESAFE_API_KEY='your-key'
uv run jev-rerank --dir ./lyrics --query 'Dua Lipa Hallucinate' --top 5
```

Or install the CLI into your environment:

```sh
python -m pip install .
jev-rerank --dir ./documents --query 'how to recover a lost account' --top 10
```

`python -m jev_rerank` works as well. Obtain an API key from the
[TypeSafe console](https://console.typesafe.ai/).

Output is tab-separated `rank`, `score`, and path relative to `--dir`. Progress
and warnings go to stderr, so stdout can be redirected or piped. Use `--json` for
results plus corpus counts, cache hits, API usage, and elapsed time. JSON results
also include the zero-based index of the winning passage.

```sh
uv run jev-rerank --dir ./documents --query '2025 annual report' --top 10 --json
```

`--docs` remains an alias for `--dir`.

Add `--show` to include the complete text behind each returned path, for both files
and SQLite field values:

```sh
uv run jev-rerank --dir ./lyrics --query 'loneliness' --top 5 --show
```

With `--json --show`, each result has a `text` property. Otherwise, the text is a
fourth tab-separated column encoded as a JSON string, escaping embedded tabs,
newlines, and quotes so every result stays on one line. Long values are restored
from all their indexed passages without duplicating overlap. The text comes from
the indexed snapshot used for the search; showing it makes no additional API calls.
Full values are loaded only for returned results when `--show` is enabled, so very
large matches can increase memory usage and output size.

## Query criteria

`--query` accepts plain text, or the same text plus an explicit **yes/no** pair
that replaces the ranking question's default criteria. Use it to say where the
line falls when a plain query is too broad:

```sh
uv run jev-rerank --dir ./lyrics --top 5 --query 'a song about missing love
| yes: the song talks about missing romantic love
| no: the song talks about something non-romantic, whatever it is like e.g. love for books or music'
```

`|` separates the sections and spaces around it are optional. The first section
is the query; every later section carries a mandatory `yes:` or `no:` prefix.
Sections may span several lines.

The same search as inline JSON, which is easier to generate from a script:

```sh
uv run jev-rerank --dir ./lyrics --top 5 --query '{
  "query": "a song about missing love",
  "yes": "the song talks about missing romantic love",
  "no": "the song talks about something non-romantic, like love for books or music"
}'
```

The JSON form is strict JSON, so a multi-line value uses `\n` escapes, and only
`query`, `yes` and `no` are accepted. In both forms `yes` and `no` are
both-or-neither: supplying one without the other is an error, because a Noul
needs both sides of the judgment. A plain query containing a literal `|` must
use the JSON form.

Criteria are part of the score cache key, so editing them re-asks the model.
With `--json` they are echoed back in a `criteria` object next to `query`.

## SQLite records

Use `--db` instead of `--dir` and repeat `--table-field TABLE_NAME.FIELD_NAME` to
search the union of the selected fields, including fields from different tables:

```sh
uv run jev-rerank --db ./library.db \
  --table-field songs.lyrics \
  --table-field articles.body \
  --query 'feeling isolated' --top 10 --json
```

Each non-NULL field value is a separate candidate. All selected fields share the
same shortlist and top-k ranking. Fields from one row are not concatenated; that
row can appear more than once if different selected fields match. Equal text in
different records keeps its source identity. Repeating the same selector does not
duplicate candidates.

Selected columns must have a declared text type such as `TEXT`, `CLOB`, `VARCHAR`,
`NVARCHAR`, or `CHAR`, following [SQLite's text affinity rules](https://www.sqlite.org/datatype3.html#determination_of_column_affinity).
Missing tables/columns and columns with numeric, BLOB, or undeclared types are
rejected before scoring. A type literally named `STRING` has numeric affinity in
SQLite and is also rejected. Because SQLite can store BLOBs even in text columns,
stored non-text values are skipped with warnings. NULLs are silently skipped and
counted in `stats.skipped`; empty strings remain searchable candidates.

The source database is opened read-only, and schema validation and row scans use
one consistent snapshot. Only the selected fields and record identifiers are read.
Primary keys, including composite keys in `WITHOUT ROWID` tables, identify records.
Rows without a usable primary key use an accessible `rowid` alias; tables with
neither are rejected. Rowids may change after database maintenance, so declared
primary keys provide more stable identifiers. Selectors accept table/field names
with spaces or quotes when shell-quoted, but names containing dots are not supported.

JSON results contain a `source` object alongside `rank`, `score`, `passage`, and
a unique `sqlite://...` locator in `path`:

```json
{
  "type": "sqlite",
  "database": "/path/to/library.db",
  "table": "songs",
  "field": "lyrics",
  "key": {"id": 42}
}
```

That source metadata, the query, and the selected text passage are included in the
TypeSafe state. Text output uses the locator as its third column; `--json` provides
the table, field, and key without needing to parse it. BLOB primary keys are encoded
as `{"blob_hex": "..."}`.

SQLite records use the same chunking, BM25 retrieval, concurrency, and score caching
as files. Each run streams the selected rows and hashes their values, reindexing
only changed values and removing deleted/NULL values. This detects committed WAL
changes as well as edits to the main database. The corpus is not loaded into memory,
but a single row's selected values must fit in memory. Index caches are separated
by database path, selected field set, and passage size. `--rebuild` reindexes every
value; `--candidates 0` scores every passage of every selected text value.

## How it scales

1. Recursively scan non-hidden regular files. UTF-8 and UTF-8-with-BOM text is
   accepted regardless of extension. Binary, invalid UTF-8, and unreadable files
   are skipped with warnings; symlinks and hidden entries are ignored. Empty files
   are indexed so they can match by filename.
2. Stream files into overlapping passages (4,000 characters by default). Store the
   complete text in a persistent SQLite FTS5 index on disk. Later runs check file
   size, modification time, and change time, and reread only changed files. Deleted
   files are removed from the index. There is no silent truncation of long files.
3. For large collections, BM25 selects up to `max(500, 10 * top)` distinct files,
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

Each file-scoring API request has this state:

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

One Noul question serves both sources: it generalizes the recipe's Noul to
“does this candidate text match the search intent?” `candidate.passage` is always
the text being judged; the remaining fields only say where it came from, either a
filename and path or a table, field and key. Its default criteria allow
filename/title/author searches as well as semantic content matches, and
[query criteria](#query-criteria) replace them. The score is TypeSafe's
probability of that yes/no judgment, between 0 and 1, sorted descending. Equal
scores are ordered by path. These are model judgments, not guaranteed relevance
measurements.

## Speed and coverage

Up to 500 documents the default scores everything, so a natural-language query
is never narrowed by keyword retrieval first. Beyond that the shortlist makes API
work independent of corpus size once the index is built. The initial run still reads the corpus; subsequent runs still scan file
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
uv run jev-rerank --dir ./documents --query 'access recovery' --top 10 --candidates 200

# Full semantic coverage, including files without any matching query words.
uv run jev-rerank --dir ./documents --query 'songs about feeling isolated' --top 10 --candidates 0
```

Exhaustive mode makes one request per passage unless cached. It can cost more and
take much longer for a large corpus. Max-passage scoring favors a document with a
strong local match; it does not assess whole-document constraints or combine facts
spread across separate passages. Larger passages can help when context matters.

## Options

| Option | Default | Purpose |
| --- | --- | --- |
| `--dir` / `--docs` | one source required | Corpus directory; mutually exclusive with `--db` |
| `--db` | one source required | SQLite database; requires `--table-field` |
| `--table-field` | required with `--db` | Repeatable `TABLE.FIELD` selector; fields share one ranking |
| `--query`, `--top` | required | Search intent (see [Query criteria](#query-criteria)) and result count |
| `--candidates` | `max(500, 10 * top)` | Distinct files or field values to shortlist; `0` scores everything |
| `--concurrency` | `16` | Maximum simultaneous API calls |
| `--model` | `jev-1.13.0` | Model ID; also accepts `TYPESAFE_DEFAULT_MODEL` |
| `--cache-dir` | `$XDG_CACHE_HOME/jev-rerank` or `~/.cache/jev-rerank` | Local text index and score cache |
| `--chunk-chars` | `4000` | Passage size, 256–16000 characters; overlap is 10%, capped at 400 |
| `--timeout` | `60` | SDK request timeout, seconds |
| `--retries` | `4` | SDK retries for transient failures |
| `--rebuild` | off | Reindex all source text even if unchanged |
| `--no-score-cache` | off | Bypass cached judgments |
| `--json` | off | Structured output |
| `--show` | off | Include the complete matched file or database field text |
| `--quiet` | off | Suppress progress, retain warnings |

Use `jev-rerank --help` for the full CLI. The top count must be positive; an explicit
nonzero candidate count must be at least the top count. Fewer results are returned
if the corpus contains fewer valid files or field values. An empty corpus returns no results.

`TYPESAFE_BASE_URL` overrides the API endpoint; `TYPESAFE_ENDPOINT` is also accepted
for compatibility with the cookbook. Do not put credentials in the URL.

The model version is pinned so cached results remain meaningful. Score cache keys
include the query, the question definition with any yes/no criteria, endpoint, model, relative filename/path,
passage index, and content hash. Model aliases may change remotely; use
`--no-score-cache` when fresh alias results are needed. Indexes are isolated by
corpus directory and passage size. The cache holds document text locally and grows
with new query/document pairs; removing the cache directory forces a fresh index
and fresh judgments. A key is required even for a fully cached invocation.

Selected filenames, relative paths, passages, queries, and SQLite source metadata
are sent to TypeSafe.
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
