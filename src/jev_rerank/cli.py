from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from .index import Index
from .query import parse_query
from .rerank import DEFAULT_MODEL, RankingStats, RerankError, rerank
from .sqlite_source import SQLiteSource, parse_table_field


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return number


def timeout_value(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def default_candidates(top: int) -> int:
    """Score everything up to a mid-sized corpus; shortlist only beyond it.

    A floor well above a few hundred documents keeps small and medium corpora
    fully semantic, which is what a natural-language query usually needs; BM25
    keyword shortlisting only takes over once the corpus is genuinely large.
    """
    return max(500, 10 * top)


def cache_directory() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "jev-rerank"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Search text files or SQLite fields with BM25 and TypeSafe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sources = result.add_mutually_exclusive_group(required=True)
    sources.add_argument(
        "--dir", "--docs", dest="docs", type=Path, help="Text files directory"
    )
    sources.add_argument("--db", type=Path, help="SQLite database to read")
    result.add_argument(
        "--table-field",
        action="append",
        default=[],
        metavar="TABLE.FIELD",
        help="Text field to search with --db; repeat to search their union",
    )
    result.add_argument(
        "--query",
        required=True,
        help="What the best documents should match: plain text, "
        "'TEXT|yes: CRITERION|no: CRITERION', or "
        """{"query": ..., "yes": ..., "no": ...}""",
    )
    result.add_argument(
        "--top", required=True, type=positive, help="Number of results to return"
    )
    result.add_argument(
        "--candidates",
        type=nonnegative,
        help="Shortlist size (default: max(500, 10 * top)); 0 scores every passage",
    )
    result.add_argument(
        "--concurrency", type=positive, default=16, help="Maximum in-flight requests"
    )
    result.add_argument(
        "--model",
        default=os.environ.get("TYPESAFE_DEFAULT_MODEL") or DEFAULT_MODEL,
        help="TypeSafe model ID; pin a version for reproducible cached results",
    )
    result.add_argument(
        "--cache-dir",
        type=Path,
        default=cache_directory(),
        help="Directory for the persistent text index and score cache",
    )
    result.add_argument(
        "--chunk-chars",
        type=positive,
        default=4000,
        help="Characters per passage, with 10%% overlap (maximum 400 characters)",
    )
    result.add_argument(
        "--timeout",
        type=timeout_value,
        default=60.0,
        help="SDK request timeout in seconds",
    )
    result.add_argument(
        "--retries",
        type=nonnegative,
        default=4,
        help="Retries for transient API failures",
    )
    result.add_argument(
        "--rebuild",
        action="store_true",
        help="Reindex all source text, even if unchanged",
    )
    result.add_argument(
        "--no-score-cache",
        action="store_true",
        help="Do not read or write cached scores",
    )
    result.add_argument(
        "--json", action="store_true", help="Write structured JSON to stdout"
    )
    result.add_argument(
        "--show",
        action="store_true",
        help="Include the full text of each returned file or database field value",
    )
    result.add_argument(
        "--quiet", action="store_true", help="Hide progress; still report warnings"
    )
    return result


async def run(args: argparse.Namespace) -> int:
    start = time.monotonic()
    query = args.parsed_query
    root = (args.db or args.docs).expanduser().resolve()
    candidates = args.candidates
    if candidates is None:
        candidates = default_candidates(args.top)
    cache_dir = args.cache_dir.expanduser().resolve()

    def status(message: str) -> None:
        if not args.quiet:
            print(message, file=sys.stderr, flush=True)

    def warning(message: str) -> None:
        print(f"Warning: {message}", file=sys.stderr, flush=True)

    index = None
    units = "field values" if args.db else "files"
    try:
        status(f"Refreshing index for {root} ...")
        if args.db:
            with SQLiteSource(root, args.table_field) as source:
                source_id = json.dumps(["sqlite", source.table_fields])
                identity = hashlib.sha256(
                    f"{root}\0{args.chunk_chars}\0{source_id}".encode()
                ).hexdigest()[:24]
                index = Index(
                    cache_dir / f"{identity}.sqlite3",
                    root,
                    args.chunk_chars,
                    source_id=source_id,
                )
                indexed = index.sync_records(
                    source.records(), warning, rebuild=args.rebuild
                )
                output_source = {"db": str(root), "table_fields": source.table_fields}
        else:
            # Retain the existing cache identity for file searches.
            identity = hashlib.sha256(
                f"{root}\0{args.chunk_chars}".encode()
            ).hexdigest()[:24]
            index = Index(cache_dir / f"{identity}.sqlite3", root, args.chunk_chars)
            indexed = index.sync(warning, rebuild=args.rebuild)
            output_source = {"docs": str(root)}
        status(
            f"Indexed {indexed.documents:,} {units} / {indexed.passages:,} passages "
            f"({indexed.updated:,} updated, {indexed.skipped:,} skipped, "
            f"{indexed.removed:,} removed)."
        )
        exhaustive = candidates == 0 or indexed.documents <= candidates
        if exhaustive:
            status(f"Scoring all {indexed.passages:,} passages with TypeSafe ...")
        else:
            status(
                f"Reranking up to {candidates:,} {units} from a BM25 shortlist "
                "(one passage per candidate); --candidates 0 scores everything."
            )
        last_progress = time.monotonic()

        def progress(stats: RankingStats) -> None:
            nonlocal last_progress
            now = time.monotonic()
            if now - last_progress >= 2:
                status(
                    f"Scored {stats.scored:,} passages "
                    f"({stats.cache_hits:,} cached, {stats.api_calls:,} API calls) ..."
                )
                last_progress = now

        # The SDK handles connection pooling, Retry-After, 429 and 5xx backoff.
        # TYPESAFE_ENDPOINT is accepted for compatibility with the cookbook.
        endpoint = (
            (
                os.environ.get("TYPESAFE_BASE_URL")
                or os.environ.get("TYPESAFE_ENDPOINT")
                or "https://api.typesafe.ai"
            )
            .strip()
            .rstrip("/")
        )
        if not indexed.documents:
            results, ranked = [], RankingStats()
        else:
            async with AsyncTypeSafeClient(
                model=args.model,
                base_url=endpoint,
                timeout=args.timeout,
                retry=RetryPolicy(max_retries=args.retries),
            ) as client:
                results, ranked = await rerank(
                    index.candidates(query.text, candidates, indexed.documents),
                    query=query,
                    top=args.top,
                    concurrency=args.concurrency,
                    model=args.model,
                    endpoint=endpoint,
                    client=client,
                    index=index,
                    use_cache=not args.no_score_cache,
                    progress=progress,
                )
        elapsed = time.monotonic() - start
        if args.json:
            print(
                json.dumps(
                    {
                        "query": query.text,
                        # Only present when --query supplied explicit criteria;
                        # the parser guarantees yes and no come as a pair.
                        **(
                            {"criteria": {"yes": query.yes, "no": query.no}}
                            if query.yes is not None
                            else {}
                        ),
                        **output_source,
                        "model": args.model,
                        "exhaustive": exhaustive,
                        "results": [
                            {
                                "rank": rank,
                                **{
                                    key: value
                                    for key, value in asdict(result).items()
                                    if value is not None
                                },
                                **(
                                    {"text": index.document_text(result.path)}
                                    if args.show
                                    else {}
                                ),
                            }
                            for rank, result in enumerate(results, 1)
                        ],
                        "stats": {
                            **asdict(indexed),
                            **asdict(ranked),
                            "elapsed_seconds": elapsed,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for rank, result in enumerate(results, 1):
                # Escape tabs/newlines in unusual filenames; normal paths stay readable.
                path = json.dumps(result.path, ensure_ascii=False)[1:-1]
                line = f"{rank}\t{result.score:.6f}\t{path}"
                if args.show:
                    text = json.dumps(
                        index.document_text(result.path), ensure_ascii=False
                    )
                    line += f"\t{text}"
                print(line)
        status(
            f"Returned {len(results)} {units} in {elapsed:.2f}s; "
            f"{ranked.api_calls:,} API calls, {ranked.cache_hits:,} cache hits, "
            f"{ranked.input_tokens:,} input tokens."
        )
        return 0
    finally:
        if index is not None:
            index.close()


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.docs is not None:
        if not args.docs.expanduser().is_dir():
            argument_parser.error("--dir must be an existing directory")
        if args.table_field:
            argument_parser.error("--table-field requires --db")
    else:
        if not args.db.expanduser().is_file():
            argument_parser.error("--db must be an existing SQLite database file")
        if not args.table_field:
            argument_parser.error("--db requires at least one --table-field")
        for field in args.table_field:
            try:
                parse_table_field(field)
            except ValueError as exc:
                argument_parser.error(str(exc))
    # The cap bounds the whole spec, criteria included, before it is parsed.
    if len(args.query.encode()) > 8192:
        argument_parser.error("--query must be at most 8192 UTF-8 bytes")
    try:
        args.parsed_query = parse_query(args.query)
    except ValueError as exc:
        argument_parser.error(str(exc))
    if args.candidates and args.candidates < args.top:
        argument_parser.error(
            "--candidates must be at least --top, or 0 for all candidates"
        )
    if not 256 <= args.chunk_chars <= 16000:
        argument_parser.error("--chunk-chars must be between 256 and 16000")
    if not os.environ.get("TYPESAFE_API_KEY", "").strip():
        argument_parser.error("Set TYPESAFE_API_KEY in your environment")
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("Interrupted. Completed scores have been cached.", file=sys.stderr)
        return 130
    except (OSError, sqlite3.Error, TypeSafeError, RerankError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
