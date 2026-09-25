"""Read a union of selected SQLite text fields from a consistent snapshot."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


def identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def is_text_type(declared_type: str) -> bool:
    """Follow SQLite's affinity rules, including INTEGER taking precedence."""
    declared_type = declared_type.upper()
    return "INT" not in declared_type and any(
        marker in declared_type for marker in ("CHAR", "CLOB", "TEXT")
    )


def parse_table_field(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) != 2 or not all(parts) or "\x00" in value:
        raise ValueError(
            f"Invalid --table-field {value!r}; expected TABLE_NAME.FIELD_NAME"
        )
    return parts[0], parts[1]


def json_key(value: Any) -> Any:
    # BLOB primary keys are valid even when the selected field must be TEXT.
    return {"blob_hex": value.hex()} if isinstance(value, bytes) else value


@dataclass(frozen=True)
class Record:
    path: str
    value: Any
    source: dict[str, Any]


@dataclass
class Selection:
    table: str
    fields: list[str]
    primary_key: list[str]
    rowid: str | None
    synthetic_rowid: bool = False


class SQLiteSource:
    def __init__(self, database: Path, table_fields: Sequence[str]):
        self.database = database.expanduser().resolve()
        if not self.database.is_file():
            raise ValueError("--db must be an existing SQLite database file")
        if not table_fields:
            raise ValueError("--db requires at least one --table-field")
        self.db = sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA query_only=ON")
            self.db.execute("BEGIN")
            self.selections = self._selections(table_fields)
        except BaseException:
            self.db.close()
            raise

    def __enter__(self) -> SQLiteSource:
        return self

    def __exit__(self, *args: Any) -> None:
        self.db.close()

    @property
    def table_fields(self) -> list[str]:
        return [f"{s.table}.{field}" for s in self.selections for field in s.fields]

    def _selections(self, table_fields: Sequence[str]) -> list[Selection]:
        selected: dict[str, Selection] = {}
        for value in table_fields:
            table, field = parse_table_field(value)
            found = self.db.execute(
                "SELECT name, type FROM sqlite_schema WHERE type IN ('table', 'view') "
                "AND name = ? COLLATE NOCASE",
                (table,),
            ).fetchone()
            if not found or found["name"].lower().startswith("sqlite_"):
                raise ValueError(
                    f"Table {table!r} does not exist or is not a user table or view"
                )
            table = found["name"]
            is_view = found["type"] == "view"
            columns = self.db.execute(
                "SELECT name, type, pk FROM pragma_table_xinfo(?)", (table,)
            ).fetchall()
            # Let SQLite apply its own identifier case rules.
            column = self.db.execute(
                "SELECT name, type FROM pragma_table_xinfo(?) "
                "WHERE name = ? COLLATE NOCASE",
                (table, field),
            ).fetchone()
            if column is None:
                raise ValueError(f"Field {field!r} does not exist in table {table!r}")
            field = column["name"]
            # Computed view columns (e.g. `a || b`) have no declared type even
            # though they always yield text; only reject a *known* non-text type.
            view_column_type_is_unknown = is_view and not column["type"]
            if not view_column_type_is_unknown and not is_text_type(column["type"]):
                raise ValueError(
                    f"{table}.{field} must have a declared text type "
                    "(TEXT, CLOB, VARCHAR, etc.); "
                    f"found {column['type'] or 'untyped'!r}"
                )
            if table not in selected:
                keys = [
                    c["name"] for c in sorted(columns, key=lambda c: c["pk"]) if c["pk"]
                ]
                names = {c["name"].lower() for c in columns}
                rowid = next(
                    (name for name in ("rowid", "_rowid_", "oid") if name not in names),
                    None,
                )
                if rowid:
                    try:
                        self.db.execute(
                            f"SELECT t.{identifier(rowid)} "
                            f"FROM {identifier(table)} AS t LIMIT 0"
                        )
                    except sqlite3.OperationalError:
                        rowid = None  # Views and WITHOUT ROWID tables have no rowid.
                synthetic_rowid = False
                if not keys and not rowid:
                    if not is_view:
                        raise ValueError(
                            f"Table {table!r} needs a primary key or accessible rowid"
                        )
                    # Views have no primary key or rowid of their own; number
                    # their rows within this snapshot so each gets a key.
                    candidates = ("_row_number_", "__row_number__", "___row_number___")
                    rowid = next(name for name in candidates if name not in names)
                    synthetic_rowid = True
                selected[table] = Selection(table, [], keys, rowid, synthetic_rowid)
            if field not in selected[table].fields:
                selected[table].fields.append(field)
        for selection in selected.values():
            selection.fields.sort()
        return sorted(selected.values(), key=lambda s: s.table)

    def records(self) -> Iterator[Record]:
        for selection in self.selections:
            key_columns = selection.primary_key
            names = [*key_columns]
            if selection.rowid:
                names.append(selection.rowid)
            names.extend(selection.fields)
            projection = ", ".join(
                f"ROW_NUMBER() OVER () AS {identifier(name)}"
                if selection.synthetic_rowid and name == selection.rowid
                else f"t.{identifier(name)}"
                for name in names
            )
            cursor = self.db.execute(
                f"SELECT {projection} FROM {identifier(selection.table)} AS t"
            )
            key_count = len(key_columns)
            field_offset = key_count + bool(selection.rowid)
            for row in cursor:
                if key_columns and all(row[i] is not None for i in range(key_count)):
                    key = {name: json_key(row[i]) for i, name in enumerate(key_columns)}
                elif selection.rowid:
                    key = {selection.rowid: row[key_count]}
                else:
                    raise ValueError(f"NULL primary key in table {selection.table!r}")
                encoded_key = quote(
                    json.dumps(key, ensure_ascii=False, sort_keys=True), safe=""
                )
                for offset, field in enumerate(selection.fields, field_offset):
                    source = {
                        "type": "sqlite",
                        "database": str(self.database),
                        "table": selection.table,
                        "field": field,
                        "key": key,
                    }
                    path = (
                        f"sqlite://{quote(selection.table, safe='')}/"
                        f"{quote(field, safe='')}?key={encoded_key}"
                    )
                    yield Record(path, row[offset], source)
