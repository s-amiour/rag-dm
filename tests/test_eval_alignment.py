"""v3T08: the UI/API evaluation and the CLI share defaults, run names, files and the summary.

BM25 only, so no embedding model is downloaded and no LLM is called. Every run writes
to a temporary folder, never to data/results/v3.
"""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import bm25, core
from app import evaluate as cli_eval

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


class DefaultsTest(unittest.TestCase):
    def test_per_corpus_chunking(self):
        self.assertEqual(core.resolve_chunking("own"), (500, 20))
        self.assertEqual(core.resolve_chunking("kestrel"), (800, 10))
        self.assertEqual(core.resolve_chunking("own", 300, 0), (300, 0))

    def test_reference_name_has_no_suffix(self):
        self.assertEqual(cli_eval.run_name(["dense", "bm25"], corpus="own", size=500, overlap=20,
                                           top_k=10, candidates=10, rrf_k=60), "dense+bm25")

    def test_non_default_settings_are_in_the_name(self):
        name = cli_eval.run_name(["dense", "bm25"], corpus="own", size=800, overlap=10,
                                 top_k=5, candidates=30, rrf_k=60)
        self.assertEqual(name, "dense+bm25 (800-10, top-k 5, cand 30)")


class RunAndSaveTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def test_reference_file_and_summary(self):
        name, result, path = cli_eval.run_and_save("kestrel", ["bm25"], out_dir=self.out)
        self.assertEqual(path.name, "kestrel__bm25.json")
        cfg = result["summary"]["config"]
        self.assertEqual((cfg["chunk_size"], cfg["overlap"], cfg["top_k"]), (800, 10, 10))
        self.assertTrue((self.out / "summary.md").exists())

    def test_one_off_run_keeps_the_reference_and_both_are_in_the_summary(self):
        _, _, ref = cli_eval.run_and_save("kestrel", ["bm25"], out_dir=self.out)
        before = ref.read_bytes()
        _, _, other = cli_eval.run_and_save("kestrel", ["bm25"], candidates=30, out_dir=self.out)
        self.assertNotEqual(ref, other)
        self.assertEqual(ref.read_bytes(), before)
        names = [r["configuration"] for r in json.loads((self.out / "summary.json").read_text(encoding="utf-8"))]
        self.assertEqual(sorted(names), ["bm25", "bm25 (cand 30)"])

    def test_failed_run_restores_the_stopword_setting(self):
        bm25.set_stopwords(True)
        with mock.patch.object(cli_eval, "evaluate_corpus", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                cli_eval.run_and_save("kestrel", ["bm25"], stopwords=False, out_dir=self.out)
        self.assertTrue(bm25.stopwords_enabled())


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ApiMatchesCliTest(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from app import main

        self.out = Path(tempfile.mkdtemp())
        p = mock.patch.object(cli_eval, "DEFAULT_OUT", self.out)
        p.start()
        self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_ui_run_writes_the_same_file_and_metrics_as_the_cli(self):
        cli_dir = Path(tempfile.mkdtemp())
        _, cli_result, _ = cli_eval.run_and_save("own", ["bm25"], out_dir=cli_dir)

        r = self.client.post("/api/evaluate", json={"corpus": "own", "retrievers": ["bm25"]})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["saved_to"].endswith("own__bm25.json"))
        cfg = body["summary"]["config"]
        self.assertEqual((cfg["chunk_size"], cfg["overlap"], cfg["top_k"]), (500, 20, 10))
        self.assertEqual(body["summary"]["hit_rates"], cli_result["summary"]["hit_rates"])
        self.assertEqual(body["summary"]["recall"], cli_result["summary"]["recall"])
        self.assertTrue((self.out / "summary.md").exists())

    def test_config_exposes_the_defaults(self):
        d = self.client.get("/api/config").json()["defaults"]
        self.assertEqual(d["chunking"]["own"], {"chunk_size": 500, "overlap": 20})
        self.assertEqual(d["top_k"], 10)


if __name__ == "__main__":
    unittest.main()
