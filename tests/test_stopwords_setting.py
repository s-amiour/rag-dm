"""v3T17: an evaluation run changes the BM25 stop-word setting for its own duration only.

BM25 only (no model, no LLM); every run writes to a temporary folder.
"""
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import bm25, core
from app import evaluate as cli_eval


class ScopedSettingTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(bm25.set_stopwords, bm25.stopwords_enabled())  # leave the process as we found it

    def test_restores_after_the_block(self):
        bm25.set_stopwords(True)
        with bm25.stopwords_setting(False):
            self.assertFalse(bm25.stopwords_enabled())
        self.assertTrue(bm25.stopwords_enabled())

    def test_restores_after_an_exception(self):
        bm25.set_stopwords(True)
        with self.assertRaises(RuntimeError):
            with bm25.stopwords_setting(False):
                raise RuntimeError("boom")
        self.assertTrue(bm25.stopwords_enabled())


class EvaluationRunTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.addCleanup(bm25.set_stopwords, bm25.stopwords_enabled())

    def test_failed_no_stopwords_run_leaves_stopwords_on(self):
        bm25.set_stopwords(True)
        with mock.patch.object(cli_eval, "evaluate_corpus", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                cli_eval.run_and_save("kestrel", ["bm25"], stopwords=False, out_dir=self.out)
        self.assertTrue(bm25.stopwords_enabled())

    def test_successful_run_restores_the_previous_value_not_true(self):
        bm25.set_stopwords(False)  # what BM25_STOPWORDS=0 sets at import
        cli_eval.run_and_save("kestrel", ["bm25"], stopwords=True, out_dir=self.out)
        self.assertFalse(bm25.stopwords_enabled())

    def test_run_after_the_ablation_uses_stopwords_again(self):
        # the order `matrix` uses: the stop-word ablation, then a normal BM25 configuration
        bm25.set_stopwords(True)
        index_dir = Path(tempfile.mkdtemp())
        with mock.patch.object(core, "INDEX_DIR", index_dir):
            _, ablation, _ = cli_eval.run_and_save("kestrel", ["bm25"], stopwords=False, out_dir=self.out)
            _, after, _ = cli_eval.run_and_save("own", ["bm25"], out_dir=self.out)
        self.assertFalse(ablation["summary"]["config"]["bm25_stopwords"])
        self.assertTrue(after["summary"]["config"]["bm25_stopwords"])
        self.assertTrue((index_dir / "bm25-own-500-20-sw1.json").exists())  # the index name matches the setting

    def test_evaluate_py_never_forces_stopwords_back_on(self):
        source = Path(cli_eval.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"set_stopwords\(\s*True\s*\)", source))


if __name__ == "__main__":
    unittest.main()
