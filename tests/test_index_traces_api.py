"""v4T03: index traces from the app (POST /api/index/{corpus}/rebuild) and on-the-fly index builds
recorded in a question's spans. Offline: fixtures in tests/api_support.py."""
import importlib.util
import os
import sqlite3
import unittest
from unittest import mock

import numpy as np

from app import core, retrieval, traces
from tests.api_support import DOCS, ApiTestCase

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
REAL_GET_DENSE = retrieval.get_dense  # ApiTestCase replaces it with a fake in setUp


class FakeEmbeddingModel:
    """Deterministic 8-dimension vectors from the characters: lets the real Retriever and its caches run."""

    def encode(self, texts, normalize_embeddings=True):
        out = np.array([[float(sum(map(ord, t)) % (i + 7)) + 1.0 for i in range(8)] for t in texts])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def child(span, name):
    return next(c for c in span["children"] if c["name"] == name)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class RebuildTraceTest(ApiTestCase):
    def test_one_rebuild_one_index_trace(self):
        r = self.client.post("/api/index/own/rebuild")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        got = traces.get_trace(d["trace_id"])
        self.assertEqual((got["trace"]["kind"], got["trace"]["corpus"], got["trace"]["outcome"]), ("index", "own", "ok"))
        self.assertRegex(got["trace"]["fingerprint"], r"^[0-9a-f]{6}$")
        [root] = got["spans"]
        self.assertEqual(root["name"], "index")
        self.assertEqual([c["name"] for c in root["children"]], ["load+chunk", "embed", "bm25.build"])
        self.assertEqual(root["attributes"]["trigger"], "rebuild")

        records = core.build_chunks("own", *core.DEFAULT_CHUNKING["own"])
        load = child(root, "load+chunk")["attributes"]
        self.assertEqual((load["documents"], load["chunks"]), (len(DOCS["own"]), len(records)))
        self.assertEqual(load["corpus_hash"], core.records_fingerprint(records))
        embed = child(root, "embed")["attributes"]
        self.assertEqual((embed["model"], embed["chunks"], embed["cache_hit"]), (core.DENSE_MODEL_NAME, len(records), False))
        bm25 = child(root, "bm25.build")["attributes"]
        self.assertEqual((bm25["chunks"], bm25["source"]), (len(records), "built"))
        self.assertGreater(bm25["vocabulary"], 0)
        self.assertIn("tokenizer_version", bm25)

        # today's response (the BM25 info) is still at the top level, the rest is added
        self.assertEqual(d["source"], "built")
        self.assertEqual(d["chunks"], len(records))
        self.assertEqual(d["bm25"]["source"], "built")
        self.assertEqual(d["dense"]["backend"], "sentence-transformers")

    def test_rebuild_uses_the_corpus_default_chunking(self):
        d = self.client.post("/api/index/own/rebuild").json()
        root = traces.get_trace(d["trace_id"])["spans"][0]["attributes"]
        self.assertEqual((root["chunk_size"], root["overlap"]), core.DEFAULT_CHUNKING["own"])  # 500/20, not 800/10
        d = self.client.post("/api/index/own/rebuild?chunk_size=300&overlap=0").json()
        root = traces.get_trace(d["trace_id"])["spans"][0]["attributes"]
        self.assertEqual((root["chunk_size"], root["overlap"]), (300, 0))

    def test_rebuild_forces_the_dense_rebuild(self):
        calls = []

        def recording_get_dense(corpus, size, overlap, records=None, strict=True, force=False):
            calls.append((strict, force))
            return mock.Mock(backend="sentence-transformers", build_seconds=0.5)

        with mock.patch.object(retrieval, "get_dense", recording_get_dense):
            self.client.post("/api/index/own/rebuild")
        self.assertEqual(calls, [(True, True)])

    def test_dense_failure_marks_embed_and_the_trace(self):
        def unavailable(*args, **kwargs):
            raise retrieval.RetrieverUnavailable(retrieval.DENSE_UNAVAILABLE)

        with mock.patch.object(retrieval, "get_dense", unavailable):
            r = self.client.post("/api/index/own/rebuild")
        self.assertEqual(r.status_code, 503)
        self.assertIn("Dense retriever unavailable", r.json()["detail"])
        con = sqlite3.connect(os.environ["TRACES_DB"])
        try:
            trace_id, outcome = con.execute("SELECT id, outcome FROM traces WHERE kind='index'").fetchone()
        finally:
            con.close()
        self.assertEqual(outcome, "error")
        [root] = traces.get_trace(trace_id)["spans"]
        self.assertEqual(child(root, "embed")["status"], "error")
        self.assertNotIn("bm25.build", [c["name"] for c in root["children"]])


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class OnTheFlyBuildTest(ApiTestCase):
    def dense_span(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        root = traces.get_trace(response.json()["trace_id"])["spans"][0]
        return child(root, "retrieve.dense")["attributes"], child(root, "retrieve.bm25")["attributes"]

    def test_first_question_builds_second_reuses_selection_change_rebuilds(self):
        with mock.patch.object(retrieval, "get_dense", REAL_GET_DENSE), \
                mock.patch.object(core, "get_embedding_model", lambda: FakeEmbeddingModel()):
            dense, bm25 = self.dense_span(self.ask(corpus="own", retrievers=["dense", "bm25"]))
            self.assertEqual((dense["index_source"], dense["cache_hit"]), ("built", False))
            self.assertIn("build_seconds", dense)
            self.assertEqual(bm25["index_source"], "built")

            dense, bm25 = self.dense_span(self.ask(corpus="own", retrievers=["dense", "bm25"]))
            self.assertEqual((dense["index_source"], dense["cache_hit"], dense["build_seconds"]), ("memory", True, 0.0))
            self.assertEqual((bm25["index_source"], bm25["build_seconds"]), ("memory", 0.0))

            keep = sorted(DOCS["own"])[:3]
            self.assertEqual(self.client.put("/api/documents/own/selection", json={"selected": keep}).status_code, 200)
            dense, bm25 = self.dense_span(self.ask(corpus="own", retrievers=["dense", "bm25"]))
            self.assertEqual((dense["index_source"], dense["cache_hit"]), ("built", False))
            self.assertEqual(bm25["index_source"], "built")


if __name__ == "__main__":
    unittest.main()
