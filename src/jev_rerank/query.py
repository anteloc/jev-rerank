"""Parse the --query argument into a search intent plus optional Noul criteria.

Three forms are accepted, in this order of detection:

1. **Inline JSON** -- the first non-space character is ``{``::

       {"query": "a song about missing love",
        "yes": "the song talks about missing romantic love",
        "no": "the song talks about something non-romantic"}

   Strict JSON, so a multi-line value is written with ``\\n`` escapes while the
   document itself may be pretty-printed across lines.

2. **Pipe-tagged** -- the argument contains ``|``::

       a song about missing love|yes: missing romantic love|no: love for books

   ``|`` is the separator (surrounding spaces optional). The first section is
   the query; every later section must carry a mandatory ``yes:`` or ``no:``
   prefix. Literal newlines are allowed anywhere.

3. **Plain text** -- anything else is the query on its own.

In both extended forms ``yes`` and ``no`` are both-or-neither: supplying one
without the other is an error, because a Noul needs both sides of the judgment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

TAGS = ("yes", "no")


@dataclass(frozen=True)
class Query:
    """A search intent, optionally with explicit Noul criteria.

    ``yes`` and ``no`` are either both set or both ``None``; when set they
    replace the default criteria of the ranking question.
    """

    text: str
    yes: str | None = None
    no: str | None = None


def parse_query(value: str) -> Query:
    """Return the `Query` written in `value`, or raise `ValueError` if malformed."""
    stripped = value.strip()
    if stripped.startswith("{"):
        return _from_json(stripped)
    if "|" in stripped:
        return _from_pipes(stripped)
    return _build(stripped)


def _from_json(value: str) -> Query:
    """Read the inline JSON form, rejecting unknown fields so typos surface."""
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"--query starts with '{{' but is not valid JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("--query JSON must be an object")
    unknown = sorted(set(payload) - {"query", *TAGS})
    if unknown:
        allowed = ", ".join(repr(name) for name in ("query", *TAGS))
        raise ValueError(
            f"Unknown --query field(s) {', '.join(map(repr, unknown))}; "
            f"expected {allowed}"
        )
    if "query" not in payload:
        raise ValueError("--query JSON must contain a 'query' field")
    for name, text in payload.items():
        if not isinstance(text, str):
            raise ValueError(f"--query field {name!r} must be a string")
    return _build(payload["query"], payload.get("yes"), payload.get("no"))


def _from_pipes(value: str) -> Query:
    """Read the pipe form: the first section is the query, the rest are tagged."""
    text, *sections = value.split("|")
    criteria: dict[str, str] = {}
    for section in sections:
        section = section.strip()
        tag = next((t for t in TAGS if section.startswith(f"{t}:")), None)
        if tag is None:
            raise ValueError(
                "Each --query section after '|' must start with 'yes:' or 'no:'. "
                "Use the JSON form for a query containing a '|' character."
            )
        if tag in criteria:
            raise ValueError(f"--query has more than one '{tag}:' section")
        criteria[tag] = section[len(tag) + 1 :]
    return _build(text, criteria.get("yes"), criteria.get("no"))


def _build(text: str, yes: str | None = None, no: str | None = None) -> Query:
    """Apply the validation shared by all three forms."""
    text = text.strip()
    yes = yes.strip() if yes is not None else None
    no = no.strip() if no is not None else None
    if not text:
        raise ValueError("--query must not be blank")
    if (yes is None) != (no is None):
        raise ValueError("--query needs both a 'yes' and a 'no' criterion, or neither")
    for name, criterion in zip(TAGS, (yes, no), strict=True):
        if criterion == "":
            raise ValueError(f"The --query '{name}' criterion must not be blank")
    return Query(text, yes, no)
