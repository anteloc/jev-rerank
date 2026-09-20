"""The --query grammar: plain text, pipe-tagged criteria, and inline JSON."""

import pytest

from jev_rerank.query import Query, parse_query


@pytest.mark.parametrize(
    "value, expected",
    [
        # Plain text: the whole argument is the search intent.
        ("a song about missing love", Query("a song about missing love")),
        ("  padded query \n", Query("padded query")),
        # Pipe form: spaces around the separators and tags are optional.
        ("q | yes: y | no: n", Query("q", "y", "n")),
        ("q|yes:y|no:n", Query("q", "y", "n")),
        ("q | no: n | yes: y", Query("q", "y", "n")),
        # Pipe form carries literal newlines in every section.
        (
            "a song\nabout love\n| yes: romantic\nloss\n| no: books\nor music",
            Query("a song\nabout love", "romantic\nloss", "books\nor music"),
        ),
        # JSON form, on one line and pretty-printed with escaped newlines.
        ('{"query": "q", "yes": "y", "no": "n"}', Query("q", "y", "n")),
        (
            '{\n  "query": "q",\n  "yes": "line one\\nline two",\n  "no": "n"\n}',
            Query("q", "line one\nline two", "n"),
        ),
        ('{"query": "q"}', Query("q")),
    ],
)
def test_accepted_forms(value, expected):
    assert parse_query(value) == expected


@pytest.mark.parametrize(
    "value, message",
    [
        # yes and no are both-or-neither, in either syntax.
        ("q | yes: y", "both a 'yes' and a 'no'"),
        ("q | no: n", "both a 'yes' and a 'no'"),
        ('{"query": "q", "yes": "y"}', "both a 'yes' and a 'no'"),
        # Nothing may be blank.
        ("   ", "must not be blank"),
        ("q | yes:  | no: n", "must not be blank"),
        # Pipe sections after the first need a mandatory tag.
        ("rock | roll", "must start with 'yes:' or 'no:'"),
        ("q | maybe: m | yes: y | no: n", "must start with 'yes:' or 'no:'"),
        ("q | yes: a | yes: b | no: n", "more than one"),
        # JSON typos surface instead of being silently dropped.
        ('{"query": "q", "yes": "y", "no": "n", "maybe": "m"}', "Unknown"),
        ('{"query": "q", "yes": 1, "no": "n"}', "must be a string"),
        ('{"yes": "y", "no": "n"}', "must contain a 'query'"),
        ('{"query": "q",}', "not valid JSON"),
        ('{"query": "q"', "not valid JSON"),
    ],
)
def test_rejected_values(value, message):
    with pytest.raises(ValueError, match=message):
        parse_query(value)


def test_pipe_error_points_at_the_json_form():
    """A plain query containing '|' has an escape hatch; say which one."""
    with pytest.raises(ValueError, match="JSON"):
        parse_query("rock | roll")
