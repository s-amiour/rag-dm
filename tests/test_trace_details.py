"""v4T04: the endpoints behind the Details panel and the Recent traces list.
Offline: a temporary trace store (TRACES_DB), traces written with app.traces directly."""
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import main, traces


class TraceEndpointsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(os.environ, {"TRACES_DB": str(self.tmp / "traces.sqlite")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(main.app)

    def make(self, kind="question", outcome="answered", corpus="kestrel"):
        with traces.trace("request", kind=kind, corpus=corpus, question="KST-5031?") as t:
            with t.span("retrieve.bm25") as s:
                s.set(results=[{"id": "runbook_md:8", "rank": 1, "score": 3.2}])
            with t.span("generate") as s:
                s.set(**{"gen_ai.usage.input_tokens": 900, "gen_ai.usage.output_tokens": 40})
            t.set_outcome(outcome)
        time.sleep(0.002)  # distinct start times for the ordering test
        return t.id

    def test_detail_returns_the_span_tree(self):
        trace_id = self.make()
        r = self.client.get(f"/api/traces/{trace_id}")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["trace"]["outcome"], "answered")
        [root] = body["spans"]
        self.assertEqual([c["name"] for c in root["children"]], ["retrieve.bm25", "generate"])

    def test_unknown_id_is_404_with_a_plain_message(self):
        self.make()
        r = self.client.get("/api/traces/does-not-exist")
        self.assertEqual(r.status_code, 404)
        self.assertIn("No trace", r.json()["detail"])

    def test_list_is_newest_first_and_limited(self):
        ids = [self.make() for _ in range(3)]
        listed = self.client.get("/api/traces?limit=2").json()
        self.assertEqual([t["id"] for t in listed], ids[::-1][:2])
        self.assertEqual(set(listed[0]), {"id", "kind", "corpus", "outcome", "started_at", "duration_ms", "fingerprint"})

    def test_list_filters_by_kind(self):
        self.make(kind="question")
        indexing = self.make(kind="indexing", outcome=None)
        listed = self.client.get("/api/traces?kind=indexing").json()
        self.assertEqual([t["id"] for t in listed], [indexing])

    def test_empty_store_lists_nothing(self):
        self.assertEqual(self.client.get("/api/traces").json(), [])


if __name__ == "__main__":
    unittest.main()