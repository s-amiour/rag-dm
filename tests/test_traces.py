"""v4T02: the trace store and its span API (app/traces.py). Offline; TRACES_DB points at a temp folder."""
import os
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import traces


class TraceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.db = self.tmp / "traces" / "traces.sqlite"
        patcher = mock.patch.dict(os.environ, {"TRACES_DB": str(self.db)})
        patcher.start()
        self.addCleanup(patcher.stop)

    def count(self, table):
        con = sqlite3.connect(self.db)
        try:
            return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            con.close()

    def test_spans_nest_and_one_trace_is_written(self):
        with traces.trace("request", kind="question", corpus="kestrel", question="KST-5031?") as t:
            with t.span("retrieve.dense") as s:
                s.set(backend="sentence-transformers")
                with t.span("inner"):
                    pass
            with t.span("fuse.rrf"):
                pass
            t.set_outcome("answered")
            t.set(fingerprint="a1b2c3", note="kept")
        self.assertTrue(t.stored)
        self.assertEqual((self.count("traces"), self.count("spans")), (1, 4))

        got = traces.get_trace(t.id)
        head = got["trace"]
        self.assertEqual((head["kind"], head["corpus"], head["outcome"], head["fingerprint"]),
                         ("question", "kestrel", "answered", "a1b2c3"))
        self.assertEqual(head["attributes"], {"note": "kept"})
        self.assertGreaterEqual(head["duration_ms"], 0)

        [root] = got["spans"]
        self.assertEqual(root["name"], "request")
        self.assertIsNone(root["parent_id"])
        self.assertEqual(root["attributes"]["question"], "KST-5031?")
        self.assertEqual([c["name"] for c in root["children"]], ["retrieve.dense", "fuse.rrf"])
        dense = root["children"][0]
        self.assertEqual(dense["parent_id"], root["id"])
        self.assertEqual(dense["attributes"], {"backend": "sentence-transformers"})
        self.assertEqual([c["name"] for c in dense["children"]], ["inner"])
        for span in (root, dense, dense["children"][0], root["children"][1]):
            self.assertEqual(span["status"], "ok")
            self.assertGreaterEqual(span["duration_ms"], 0)

    def test_error_in_a_span_is_recorded_and_reraised(self):
        holder = {}
        with self.assertRaises(ValueError):
            with traces.trace("request", kind="question") as t:
                holder["t"] = t
                t.set_outcome("answered")  # an exception afterwards still ends as "error"
                with t.span("generate"):
                    raise ValueError("boom " + "x" * 500)
        t = holder["t"]
        self.assertTrue(t.stored)
        got = traces.get_trace(t.id)
        self.assertEqual(got["trace"]["outcome"], "error")
        [root] = got["spans"]
        [generate] = root["children"]
        self.assertEqual((root["status"], generate["status"]), ("error", "error"))
        self.assertTrue(generate["error"].startswith("ValueError: boom "))
        self.assertEqual(len(generate["error"]), len("ValueError: ") + traces.MAX_ERROR)

    def test_an_error_caught_inside_the_trace_does_not_fail_it(self):
        with traces.trace("request", kind="question") as t:
            try:
                with t.span("retrieve.graph"):
                    raise RuntimeError("graph missing")
            except RuntimeError:
                pass
            t.set_outcome("answered")
        got = traces.get_trace(t.id)
        self.assertEqual(got["trace"]["outcome"], "answered")
        [root] = got["spans"]
        self.assertEqual(root["status"], "ok")
        self.assertEqual(root["children"][0]["status"], "error")

    def test_a_broken_store_never_breaks_the_caller(self):
        blocker = self.tmp / "not-a-folder"
        blocker.write_text("a file where the folder should be")
        with mock.patch.dict(os.environ, {"TRACES_DB": str(blocker / "traces.sqlite")}):
            with self.assertLogs("app.traces", level="WARNING") as logs:
                with traces.trace("request", kind="question") as t:
                    with t.span("generate"):
                        pass
        self.assertFalse(t.stored)
        self.assertIn("not stored", logs.output[0])

    def test_long_strings_are_never_stored(self):
        chunk_text = "The Kestrel addresses are missing from the allowlist. " * 40  # about 2,000 characters
        with self.assertLogs("app.traces", level="WARNING"):
            with traces.trace("request", kind="question") as t:
                with t.span("retrieve.bm25") as s:
                    s.set(text=chunk_text, results=[{"id": "doc_md:4", "text": chunk_text}])
        raw = self.db.read_bytes()
        self.assertNotIn(b"Kestrel addresses are missing", raw)
        attrs = traces.get_trace(t.id)["spans"][0]["children"][0]["attributes"]
        self.assertTrue(attrs["text"].startswith("[not stored:"))
        self.assertEqual(attrs["results"][0]["id"], "doc_md:4")
        self.assertTrue(attrs["results"][0]["text"].startswith("[not stored:"))

    def test_short_values_are_kept_as_json(self):
        with traces.trace("request", kind="question") as t:
            with t.span("fuse.rrf") as s:
                s.set(rrf_k=60, weights=None, ok=True, ids=("a:1", "b:2"), where=Path("data") / "x")
        attrs = traces.get_trace(t.id)["spans"][0]["children"][0]["attributes"]
        self.assertEqual(attrs["rrf_k"], 60)
        self.assertIsNone(attrs["weights"])
        self.assertIs(attrs["ok"], True)
        self.assertEqual(attrs["ids"], ["a:1", "b:2"])
        self.assertEqual(attrs["where"], str(Path("data") / "x"))

    def test_unknown_trace_is_none(self):
        self.assertIsNone(traces.get_trace("nope"))  # no store yet
        with traces.trace("request", kind="question"):
            pass
        self.assertIsNone(traces.get_trace("nope"))

    def test_each_trace_gets_its_own_id(self):
        ids = set()
        for _ in range(3):
            with traces.trace("request", kind="question") as t:
                pass
            ids.add(t.id)
        self.assertEqual(len(ids), 3)
        self.assertEqual(self.count("traces"), 3)


if __name__ == "__main__":
    unittest.main()
