"""v3T07: dense refuses the TF-IDF fallback in the API unless it is explicitly allowed.

The embedding model is made to fail with a patch, so no model is downloaded and
no LLM is called (generate_answer_with_usage is replaced by a fake).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app import core, main
from app import evaluate as cli_eval
from app.retrieval import DENSE_UNAVAILABLE

FAKE_ANSWER = (json.dumps({"answer": "ok", "citations": [], "valid": True}),
               {"llm_calls": 1, "prompt_tokens": 1, "completion_tokens": 1, "latency_s": 0.0})

ASK = {"corpus": "kestrel", "question": "What does KST-5031 mean?", "retrievers": ["dense"]}
EVAL = {"corpus": "kestrel", "retrievers": ["dense"]}


def model_fails():
    raise OSError("simulated: sentence-transformers cannot load")


class StrictDenseApiTest(unittest.TestCase):
    def setUp(self):
        core._DENSE_CACHE.clear()  # a cached MiniLM or TF-IDF retriever must not hide the behaviour
        self.client = TestClient(main.app)
        patches = [
            mock.patch.object(core, "get_embedding_model", side_effect=model_fails),
            mock.patch.object(main, "generate_answer_with_usage", return_value=FAKE_ANSWER),
            mock.patch.object(main, "save_json"),  # never write result files from a test
            mock.patch.object(cli_eval, "DEFAULT_OUT", Path(tempfile.mkdtemp())),  # v3T08: /api/evaluate saves via the CLI path
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(core._DENSE_CACHE.clear)

    def test_ask_is_strict_by_default(self):
        r = self.client.post("/api/ask", json=ASK)
        self.assertEqual(r.status_code, 503)
        self.assertIn(DENSE_UNAVAILABLE, r.json()["detail"])

    def test_evaluate_is_strict_by_default(self):
        r = self.client.post("/api/evaluate", json=EVAL)
        self.assertEqual(r.status_code, 503)
        self.assertIn(DENSE_UNAVAILABLE, r.json()["detail"])

    def test_ask_with_fallback_says_tfidf(self):
        r = self.client.post("/api/ask", json={**ASK, "allow_tfidf_fallback": True})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["dense_backend"], "tfidf")

    def test_evaluate_with_fallback_records_it(self):
        r = self.client.post("/api/evaluate", json={**EVAL, "allow_tfidf_fallback": True})
        self.assertEqual(r.status_code, 200)
        cfg = r.json()["summary"]["config"]
        self.assertEqual(cfg["dense_backend"], "tfidf")
        self.assertTrue(cfg["allow_tfidf_fallback"])

    def test_cached_tfidf_is_not_reused_by_a_strict_call(self):
        self.client.post("/api/ask", json={**ASK, "allow_tfidf_fallback": True})  # caches a TF-IDF retriever
        r = self.client.post("/api/ask", json=ASK)
        self.assertEqual(r.status_code, 503)

    def test_bm25_only_is_not_affected(self):
        r = self.client.post("/api/ask", json={**ASK, "retrievers": ["bm25"]})
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.json()["dense_backend"])


class ConfigTest(unittest.TestCase):
    def test_dense_availability_follows_the_install(self):
        client = TestClient(main.app)
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assertFalse(client.get("/api/config").json()["available"]["dense"])


if __name__ == "__main__":
    unittest.main()
