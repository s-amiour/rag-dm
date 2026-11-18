"""v4T05: cross-encoder reranking. Offline: a fake cross-encoder replaces the real model.

What is checked: the reorder, the cut made AFTER reranking, the ranks recorded (on the hits and in
the trace), the 30-candidate pool, a clear error when the model cannot load, and that rerank off
gives exactly V3's results."""
import importlib.util
import sys
import unittest
from unittest import mock

from app import core, rerank, retrieval, traces
from tests.api_support import ApiTestCase, chunk_id

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


class FakeCrossEncoder:
    """Scores a (question, chunk) pair from a table keyed by chunk text; unknown texts score 0.
    Records every call, so a test can see what the reranker was given."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return [self.scores.get(text, 0.0) for _, text in pairs]


def hit(i, score):
    return {"id": f"doc:{i}", "document": "doc.md", "index": i, "text": f"text {i}", "score": score}


class RerankUnitTest(unittest.TestCase):
    def test_reorders_cuts_after_reranking_and_records_ranks(self):
        fused = [hit(i, 1.0 - i / 10) for i in range(5)]  # fusion order: doc:0 .. doc:4
        model = FakeCrossEncoder({"text 4": 9.0, "text 2": 5.0, "text 0": 1.0})

        out = rerank.rerank("q", fused, top_k=2, model=model)

        # all 5 were scored: the cut comes after reranking, so doc:4 (last after fusion) can win
        self.assertEqual(len(model.calls[0]), 5)
        self.assertEqual([h["id"] for h in out], ["doc:4", "doc:2"])
        self.assertEqual([(h["rank_before"], h["rank_after"]) for h in out], [(5, 1), (3, 2)])
        self.assertEqual(out[0]["score"], 9.0)
        self.assertEqual(out[0]["rerank_score"], 9.0)
        self.assertAlmostEqual(out[0]["fused_score"], 0.6)

    def test_ties_keep_the_fusion_order(self):
        fused = [hit(i, 0.5) for i in range(3)]
        out = rerank.rerank("q", fused, top_k=3, model=FakeCrossEncoder({}))
        self.assertEqual([h["rank_before"] for h in out], [1, 2, 3])

    def test_pool_is_thirty_unless_more_was_asked(self):
        self.assertEqual(rerank.pool_size(10, 5), 30)
        self.assertEqual(rerank.pool_size(50, 5), 50)

    def test_model_that_cannot_load_is_a_clear_error(self):
        with mock.patch.object(rerank, "_MODEL", None), \
                mock.patch.dict(sys.modules, {"sentence_transformers": None}):
            with self.assertRaises(rerank.RerankerUnavailable) as ctx:
                rerank.get_model()
        self.assertIn(rerank.MODEL_NAME, str(ctx.exception))
        self.assertIn("switch reranking off", str(ctx.exception))


class FakeDenseMany:
    """Dense retriever over 40 chunks with a fixed order; remembers how many it was asked for."""
    backend = "sentence-transformers"
    source = "memory"
    build_seconds = 0.0

    def __init__(self, records):
        self.records = records
        self.asked = []

    def search(self, question, n):
        self.asked.append(n)
        return [{**r, "score": 1.0 - i / 100} for i, r in enumerate(self.records[:n])]


class RetrieveWithRerankTest(unittest.TestCase):
    def setUp(self):
        self.records = [{"id": f"doc:{i}", "document": "doc.md", "index": i, "text": f"text {i}"}
                        for i in range(40)]
        self.dense = FakeDenseMany(self.records)
        patcher = mock.patch.object(retrieval, "get_dense", lambda *a, **k: self.dense)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_retrieve(self, **kw):
        return retrieval.retrieve("kestrel", "q", 800, 10, ["dense"], depth=5, candidates=10,
                                  records=self.records, **kw)

    def test_rerank_gets_a_30_chunk_pool_and_keeps_the_top_k(self):
        # doc:25 is rank 26 for dense: outside plain fusion's 10, inside the reranker's 30
        model = FakeCrossEncoder({"text 25": 9.0, "text 3": 4.0})
        with mock.patch.object(rerank, "get_model", lambda: model):
            out = self.run_retrieve(rerank=True)

        self.assertEqual(self.dense.asked, [30])
        self.assertEqual(len(model.calls[0]), 30)
        self.assertEqual(len(out["hits"]), 5)
        self.assertEqual(out["hits"][0]["id"], "doc:25")
        self.assertEqual((out["hits"][0]["rank_before"], out["hits"][0]["rank_after"]), (26, 1))
        self.assertEqual(out["candidates"], 30)
        self.assertEqual(out["reranker"], rerank.MODEL_NAME)
        self.assertIn("rerank_s", out["timings"])

    def test_rerank_off_is_v3(self):
        def no_model():
            raise AssertionError("the cross-encoder must not be touched when reranking is off")

        with mock.patch.object(rerank, "get_model", no_model):
            default = self.run_retrieve()
            off = self.run_retrieve(rerank=False)

        self.assertEqual(self.dense.asked, [10, 10])
        expected = [f"doc:{i}" for i in range(5)]
        for out in (default, off):
            self.assertEqual([h["id"] for h in out["hits"]], expected)
            self.assertTrue(all("rank_before" not in h for h in out["hits"]))
            self.assertEqual((out["candidates"], out["reranker"]), (10, None))
            self.assertNotIn("rerank_s", out["timings"])

    def test_model_failure_is_never_skipped_silently(self):
        def broken():
            raise rerank.RerankerUnavailable("Reranker unavailable: test")

        with mock.patch.object(rerank, "get_model", broken):
            with self.assertRaises(retrieval.RetrieverUnavailable):
                self.run_retrieve(rerank=True)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class RerankApiTest(ApiTestCase):
    def setUp(self):
        super().setUp()
        # fixture corpus: the descaling chunk gets the best cross-encoder score
        texts = {r["text"]: (9.0 if "descaling" in r["text"] else 0.0) for r in core.build_chunks("own", 500, 20)}
        self.model = FakeCrossEncoder(texts)
        patcher = mock.patch.object(rerank, "get_model", lambda: self.model)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_ask_with_rerank_records_ranks_and_a_rerank_span(self):
        r = self.ask(corpus="own", retrievers=["dense", "bm25"], top_k=2, rerank=True)
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual(d["reranker"], rerank.MODEL_NAME)
        self.assertEqual(d["retrieved"][0]["id"], chunk_id("b-descaling.txt"))
        self.assertEqual(d["retrieved"][0]["rank_after"], 1)
        self.assertEqual(len(d["retrieved"]), 2)

        [root] = traces.get_trace(d["trace_id"])["spans"]
        self.assertTrue(root["attributes"]["rerank"])
        names = [c["name"] for c in root["children"]]
        self.assertEqual(names, ["retrieve.dense", "retrieve.bm25", "fuse.rrf", "rerank", "generate", "validate"])
        span = next(c for c in root["children"] if c["name"] == "rerank")["attributes"]
        self.assertEqual(span["model"], rerank.MODEL_NAME)
        self.assertEqual(span["candidates"], 4)  # the whole fixture corpus fits in the pool
        self.assertEqual([h["rank_after"] for h in span["results"]], [1, 2])
        self.assertEqual(len(span["dropped"]), 2)
        self.assertEqual((span["llm_calls"], span["cost_usd"]), (0, 0.0))

    def test_rerank_changes_the_fingerprint_and_off_keeps_it(self):
        off = self.ask(corpus="own", retrievers=["dense", "bm25"]).json()
        default = self.client.post("/api/ask", json={"corpus": "own", "question": "error E47",
                                                     "retrievers": ["dense", "bm25"]}).json()
        on = self.ask(corpus="own", retrievers=["dense", "bm25"], rerank=True).json()

        fp = lambda d: traces.get_trace(d["trace_id"])["trace"]["fingerprint"]
        self.assertEqual(fp(off), fp(default))
        self.assertNotEqual(fp(off), fp(on))
        self.assertIsNone(off["reranker"])
        self.assertTrue(all("rank_before" not in h for h in off["retrieved"]))

    def test_model_that_cannot_load_is_a_503_that_says_so(self):
        def broken():
            raise rerank.RerankerUnavailable(f"Reranker unavailable: the cross-encoder {rerank.MODEL_NAME} could not load")

        with mock.patch.object(rerank, "get_model", broken):
            r = self.ask(corpus="own", rerank=True)
        self.assertEqual(r.status_code, 503)
        self.assertIn("cross-encoder", r.json()["detail"])
        self.assertEqual(self.llm_calls, 0)  # no answer is generated from unreranked chunks

    def test_evaluation_run_with_rerank_is_its_own_file(self):
        from app import evaluate as cli_eval

        name_off, off, _ = cli_eval.run_and_save("own", ["dense", "bm25"])
        name_on, on, path = cli_eval.run_and_save("own", ["dense", "bm25"], rerank=True)

        self.assertEqual(name_on, name_off + " (rerank)")
        self.assertIn("rerank", path.name)
        self.assertNotEqual(off["summary"]["fingerprint"], on["summary"]["fingerprint"])
        self.assertEqual(on["summary"]["config"]["reranker"], rerank.MODEL_NAME)
        self.assertIsNone(off["summary"]["config"]["reranker"])
        self.assertIsNotNone(on["summary"]["cost_per_question"]["rerank_ms_median"])
        self.assertIsNone(off["summary"]["cost_per_question"]["rerank_ms_median"])
        self.assertEqual(on["summary"]["cost_per_question"]["llm_calls"], 0)


if __name__ == "__main__":
    unittest.main()
