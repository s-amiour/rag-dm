"""Local trace store (v4T02): one trace per question or indexing run, one span per step.

Usage:

    with traces.trace("request", kind="question", corpus=corpus, question=q) as t:
        with t.span("retrieve.bm25") as s:
            s.set(results=[{"id": "doc_md:3", "rank": 1, "score": 8.6}])
        t.set_outcome("answered")
        t.set(fingerprint=fp)
    traces.get_trace(t.id)  # -> {"trace": {...}, "spans": [root span with "children"]}

Rules (DECISIONS, tracing entry):
- The first argument of trace() names the root span; keyword arguments other than kind and corpus are
  attributes of that root span. Spans opened inside another span get it as parent.
- An exception inside a span marks it status="error" with a short message and is RE-RAISED: tracing never
  hides an error. An exception that leaves the trace sets the outcome to "error".
- The store itself never breaks the caller: if the trace cannot be written (disk full, locked or
  unwritable file), it is logged and dropped.
- One write per trace, when it closes, in one transaction.
- Attribute values are JSON values. A string longer than MAX_STRING characters is NOT stored, not even in
  part, and is replaced by a short note: that keeps chunk texts and documents out of the store by accident
  (store chunk ids; the text is already in the index).

Storage: SQLite file data/traces/traces.sqlite (git-ignored); the environment variable TRACES_DB overrides
the path (tests point it at a temporary folder). Standard library only.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = ROOT / "data" / "traces" / "traces.sqlite"
MAX_STRING = 500
MAX_ERROR = 200

# Outcome of a question trace (our own definition, ticket v4T02 section 3). Other kinds may use their own.
OUTCOMES = ("answered", "abstained", "invalid", "error")

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    duration_ms REAL,
    outcome     TEXT,
    fingerprint TEXT,
    corpus      TEXT,
    attributes  TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS spans (
    id          TEXT PRIMARY KEY,
    trace_id    TEXT NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    parent_id   TEXT,
    name        TEXT NOT NULL,
    started_at  TEXT NOT NULL,
    duration_ms REAL,
    status      TEXT NOT NULL,
    error       TEXT,
    attributes  TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS spans_by_trace ON spans(trace_id);
CREATE INDEX IF NOT EXISTS traces_by_start ON traces(started_at);
"""


def db_path() -> Path:
    return Path(os.environ.get("TRACES_DB") or DEFAULT_DB)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _new_id() -> str:
    return uuid.uuid4().hex


def describe(exc: BaseException) -> str:
    """Short error text for a span: exception type and the first MAX_ERROR characters of its message."""
    return f"{type(exc).__name__}: {str(exc)[:MAX_ERROR]}"


def clean(value: Any) -> Any:
    """A JSON value safe to store. Long strings are replaced by a note (never stored, even in part);
    other objects become their str() and follow the same rule."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if len(value) > MAX_STRING:
            log.warning("trace attribute of %d characters not stored (limit %d)", len(value), MAX_STRING)
            return f"[not stored: {len(value)} characters, limit {MAX_STRING}]"
        return value
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [clean(v) for v in value]
    return clean(str(value))


class Span:
    def __init__(self, name: str, parent_id: str | None):
        self.id = _new_id()
        self.name = name
        self.parent_id = parent_id
        self.started_at = _now()
        self._t0 = time.perf_counter()
        self.duration_ms: float | None = None
        self.status = "ok"
        self.error: str | None = None
        self.attributes: dict[str, Any] = {}

    def set(self, **attributes: Any) -> "Span":
        self.attributes.update(clean(attributes))
        return self

    def _close(self) -> None:
        self.duration_ms = round((time.perf_counter() - self._t0) * 1000, 3)


class Trace:
    """Collects spans in memory; trace() writes them when the trace closes."""

    def __init__(self, kind: str, corpus: str | None):
        self.id = _new_id()
        self.kind = kind
        self.corpus = corpus
        self.started_at = _now()
        self.duration_ms: float | None = None
        self.outcome: str | None = None
        self.fingerprint: str | None = None
        self.attributes: dict[str, Any] = {}
        self.spans: list[Span] = []
        self.root: Span | None = None
        self.stored = False
        self._stack: list[Span] = []

    @contextmanager
    def span(self, name: str, **attributes: Any) -> Iterator[Span]:
        s = Span(name, self._stack[-1].id if self._stack else None)
        s.set(**attributes)
        self.spans.append(s)
        self._stack.append(s)
        try:
            yield s
        except BaseException as exc:
            s.status = "error"
            s.error = describe(exc)
            raise
        finally:
            s._close()
            self._stack.pop()

    def set(self, **fields: Any) -> "Trace":
        """Trace-level fields: outcome, fingerprint and corpus have their own columns; anything else goes
        into the trace's attributes."""
        for key in ("outcome", "fingerprint", "corpus"):
            if key in fields:
                setattr(self, key, clean(fields.pop(key)))
        self.attributes.update(clean(fields))
        return self

    def set_outcome(self, outcome: str) -> "Trace":
        self.outcome = outcome
        return self


@contextmanager
def trace(name: str, *, kind: str, corpus: str | None = None, **attributes: Any) -> Iterator[Trace]:
    """Open a trace whose root span is called `name` (attributes go on the root span); written on close."""
    t = Trace(kind, corpus)
    try:
        with t.span(name, **attributes) as root:
            t.root = root
            yield t
    except BaseException:
        t.outcome = "error"
        raise
    finally:
        if t.root is not None:
            t.duration_ms = t.root.duration_ms
        t.stored = _save(t)


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path, timeout=5)
    con.executescript(SCHEMA)
    return con


def _save(t: Trace) -> bool:
    """Write the trace and its spans in one transaction. Never raises: a failure is logged."""
    try:
        path = db_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        con = _connect(path)
        try:
            with con:
                con.execute(
                    "INSERT INTO traces (id, kind, started_at, duration_ms, outcome, fingerprint, corpus, attributes)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (t.id, t.kind, t.started_at, t.duration_ms, t.outcome, t.fingerprint, t.corpus,
                     json.dumps(t.attributes)),
                )
                con.executemany(
                    "INSERT INTO spans (id, trace_id, parent_id, name, started_at, duration_ms, status, error, attributes)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    [(s.id, t.id, s.parent_id, s.name, s.started_at, s.duration_ms, s.status, s.error,
                      json.dumps(s.attributes)) for s in t.spans],
                )
        finally:
            con.close()
        return True
    except Exception as exc:  # the store must never break a question
        log.warning("trace %s not stored: %s", t.id, describe(exc))
        return False


def get_trace(trace_id: str) -> dict | None:
    """{"trace": {...}, "spans": [root spans, each with "children"]}, or None if there is no such trace."""
    path = db_path()
    if not path.exists():
        return None
    con = _connect(path)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM traces WHERE id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        spans = con.execute(
            "SELECT * FROM spans WHERE trace_id = ? ORDER BY started_at, rowid", (trace_id,)
        ).fetchall()
    finally:
        con.close()
    head = dict(row)
    head["attributes"] = json.loads(head["attributes"])
    nodes = {}
    for s in spans:
        node = dict(s)
        node["attributes"] = json.loads(node["attributes"])
        node["children"] = []
        nodes[node["id"]] = node
    roots = []
    for node in nodes.values():
        parent = nodes.get(node["parent_id"]) if node["parent_id"] else None
        (parent["children"] if parent else roots).append(node)
    return {"trace": head, "spans": roots}
