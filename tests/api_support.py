"""Shared offline fixtures for the API tests (v3T15).

ApiTestCase gives every test:
- a temporary data folder with a small fixture corpus for "kestrel" and "own" and two questions each
  (core.DATA, core.INDEX_DIR, main.DATA and the evaluation output folder all point there);
- a fake dense retriever with a fixed ranking (no model download);
- a fake LLM that cites the first retrieved chunk and counts its calls (no API key, no quota);
- a network guard: any attempt to open a socket fails the test;
- `self.graph_module(...)` to plug in a fake graph retriever, or to make it look missing.

Not collected as a test module (its name does not start with "test").
"""
from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import app
from app import bm25, core, evaluation, retrieval
from app import evaluate as cli_eval

DOCS = {
    "own": {
        "a-errors.txt": "Error E47 means the grinder is blocked by a foreign object. Remove the beans and restart.",
        "b-descaling.txt": "To remove calcium deposits, run the descaling programme with the descaling liquid.",
        "c-milk.txt": "Clean the milk nozzle after every use with warm water.",
        "d-warranty.txt": "The machine is covered by a two-year warranty from the date of purchase.",
    },
    "kestrel": {
        "runbook.md": "### KST-5031\nCause: a delivery failed. This error is raised by postroom.\n",
        "catalog.md": "### postroom\nSends notification e-mails. Owner: Messaging.\n",
        "teams.md": "| Team | Channel |\n|---|---|\n| Messaging | #oncall-messaging |\n",
        "billing.md": "Invoices can be downloaded from Settings, Plan, Invoices.\n",
    },
}

QUESTIONS = {
    "own": [
        {"id": "O1", "question": "What does error E47 mean?", "type": "exact_identifier",
         "answerable": True, "answer_quote": "Error E47 means the grinder is blocked"},
        {"id": "O2", "question": "How do I get rid of calcium deposits?", "type": "vocabulary",
         "answerable": True, "answer_quote": "run the descaling programme"},
    ],
    "kestrel": [
        {"id": "K1", "question": "Which service raises KST-5031?", "type": "exact_identifier",
         "answerable": True, "answer_quote": "This error is raised by postroom."},
        {"id": "K2", "question": "Where are invoices downloaded?", "type": "exact_identifier",
         "answerable": True, "answer_quote": "Invoices can be downloaded from Settings"},
    ],
}


def chunk_id(name: str) -> str:
    """Every fixture document fits in one chunk at the default chunking."""
    return f"{core.safe_name(name)}:0"


# Fixed dense ranking: reverse alphabetical order of the documents, so the E47 chunk
# (a-errors.txt) is ranked LAST by dense and first by BM25 for "error E47".
DENSE_ORDER = {corpus: [chunk_id(n) for n in sorted(docs, reverse=True)] for corpus, docs in DOCS.items()}


class FakeDense:
    backend = "sentence-transformers"
    build_seconds = 0.0  # read by the evaluation when it warms the indexes
    cache_hit = True

    def __init__(self, corpus: str, records: list[dict]):
        by_id = {r["id"]: r for r in records}
        self.ranked = [by_id[i] for i in DENSE_ORDER[corpus] if i in by_id]

    def search(self, question: str, n: int) -> list[dict]:
        return [{**r, "score": 1.0 - 0.1 * i} for i, r in enumerate(self.ranked[:n])]


class ApiTestCase(unittest.TestCase):
    def setUp(self):
        # imported here, not at the top: without fastapi the API tests are skipped instead of crashing the run
        from fastapi.testclient import TestClient
        from app import main

        self.data = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.data, True)
        for corpus, docs in DOCS.items():
            folder = self.data / "corpora" / corpus
            folder.mkdir(parents=True)
            for name, text in docs.items():
                (folder / name).write_text(text, encoding="utf-8")
        (self.data / "questions").mkdir()
        for corpus, rows in QUESTIONS.items():
            (self.data / "questions" / f"{corpus}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        self.llm_calls = 0
        self.llm_reply = None  # None = valid answer citing the first hit; a dict = that reply, as JSON

        def fake_llm(question, hits):
            self.llm_calls += 1
            reply = self.llm_reply or {"answer": "From the documents.",
                                       "citations": [hits[0]["id"]] if hits else [], "valid": True}
            return json.dumps(reply), {"llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 5,
                                       "latency_s": 0.0, "finish_reason": "stop", "thinking": "off"}

        def fake_get_dense(corpus, size, overlap, records=None, strict=True):
            return FakeDense(corpus, records if records is not None else core.build_chunks(corpus, size, overlap))

        real_connect = socket.socket.connect
        real_create_connection = socket.create_connection
        local = ("127.0.0.1", "::1", "localhost")

        # Loopback stays allowed: on Windows, asyncio (used inside FastAPI's TestClient) connects a
        # socket pair to 127.0.0.1 just to start its event loop. Anything else fails the test.
        def guarded_connect(sock, address, *args, **kwargs):
            if isinstance(address, tuple) and address and address[0] in local:
                return real_connect(sock, address, *args, **kwargs)
            raise AssertionError(f"a test tried to reach the network: {address!r}")

        def guarded_create_connection(address, *args, **kwargs):
            if isinstance(address, tuple) and address and address[0] in local:
                return real_create_connection(address, *args, **kwargs)
            raise AssertionError(f"a test tried to reach the network: {address!r}")

        patches = [
            mock.patch.object(core, "DATA", self.data),
            mock.patch.object(core, "INDEX_DIR", self.data / "index"),
            mock.patch.object(main, "DATA", self.data),
            mock.patch.object(cli_eval, "DEFAULT_OUT", self.data / "results" / "v3"),
            mock.patch.object(retrieval, "get_dense", fake_get_dense),
            mock.patch.object(evaluation, "get_dense", fake_get_dense),
            mock.patch.object(main, "generate_answer_with_usage", fake_llm),
            mock.patch.object(core, "generate_answer_with_usage", fake_llm),
            mock.patch.object(socket.socket, "connect", guarded_connect),
            mock.patch.object(socket, "create_connection", guarded_create_connection),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

        bm25._CACHE.clear()
        core._DENSE_CACHE.clear()
        self.addCleanup(bm25._CACHE.clear)
        self.client = TestClient(main.app)

    def ask(self, **body):
        body.setdefault("question", "error E47")
        return self.client.post("/api/ask", json=body)

    @contextmanager
    def graph_module(self, fake):
        """Plug in a fake app.graph_retriever (a module), or make it look missing (fake=None).

        Both sys.modules and the attribute on the `app` package are patched: once the real module
        has been imported, `from . import graph_retriever` reads the attribute, not sys.modules."""
        missing = object()
        saved = getattr(app, "graph_retriever", missing)
        with mock.patch.dict(sys.modules, {"app.graph_retriever": fake}):
            if fake is None:
                if saved is not missing:
                    delattr(app, "graph_retriever")
            else:
                app.graph_retriever = fake
            try:
                yield
            finally:
                if saved is missing:
                    if hasattr(app, "graph_retriever"):
                        delattr(app, "graph_retriever")
                else:
                    app.graph_retriever = saved


def fake_graph_module(hits: list[dict], path: dict | None = None) -> types.ModuleType:
    """A graph retriever that follows the search contract in app/retrieval.py."""
    module = types.ModuleType("app.graph_retriever")
    module.search = lambda corpus, query, k, chunk_size, overlap: {
        "hits": hits,
        "path": path or {"entities": [{"name": "KST-5031", "type": "ErrorCode", "matched_text": "KST-5031"}],
                         "edges": [], "chunks": [h["id"] for h in hits]},
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }
    return module
