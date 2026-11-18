"""v4T02: question traces written by /api/ask, outcomes, gen_ai attributes, cost, and the config
fingerprint shared by questions and evaluation runs. Offline: fixtures in tests/api_support.py."""
import importlib.util
import os
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

from app import core, traces
from tests.api_support import DOCS, ApiTestCase, chunk_id, fake_graph_module

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


def names(span):
    return [c["name"] for c in span["children"]]


def child(span, name):
    return next(c for c in span["children"] if c["name"] == name)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class QuestionTraceTest(ApiTestCase):
    def trace_of(self, response):
        self.assertEqual(response.status_code, 200, response.text)
        got = traces.get_trace(response.json()["trace_id"])
        self.assertIsNotNone(got)
        return got

    def trace_count(self):
        con = sqlite3.connect(os.environ["TRACES_DB"])
        try:
            return con.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
        finally:
            con.close()

    def test_one_question_one_trace_with_a_span_per_step(self):
        r = self.ask(corpus="own", retrievers=["dense", "bm25"], top_k=4)
        got = self.trace_of(r)
        self.assertEqual(self.trace_count(), 1)
        head = got["trace"]
        self.assertEqual((head["kind"], head["corpus"], head["outcome"]), ("question", "own", "answered"))
        self.assertRegex(head["fingerprint"], r"^[0-9a-f]{6}$")

        [root] = got["spans"]
        self.assertEqual(root["name"], "request")
        self.assertEqual(names(root), ["retrieve.dense", "retrieve.bm25", "fuse.rrf", "generate", "validate"])
        attrs = root["attributes"]
        self.assertEqual(attrs["question"], "error E47")
        self.assertEqual(attrs["retrievers"], ["dense", "bm25"])
        self.assertEqual((attrs["top_k"], attrs["candidates"], attrs["rrf_k"]), (4, 10, 60))
        self.assertEqual((attrs["chunk_size"], attrs["overlap"]), core.DEFAULT_CHUNKING["own"])

        dense = child(root, "retrieve.dense")["attributes"]
        self.assertEqual(dense["backend"], "sentence-transformers")
        self.assertEqual([h["rank"] for h in dense["results"]], list(range(1, len(dense["results"]) + 1)))
        bm25 = child(root, "retrieve.bm25")["attributes"]
        self.assertEqual(bm25["results"][0]["id"], chunk_id("a-errors.txt"))
        self.assertIn("stopwords", bm25)
        fused = child(root, "fuse.rrf")["attributes"]
        self.assertEqual(fused["rrf_k"], 60)
        self.assertEqual([h["id"] for h in fused["results"]], [h["id"] for h in r.json()["retrieved"]])
        e47 = next(h for h in fused["results"] if h["id"] == chunk_id("a-errors.txt"))
        self.assertEqual(set(e47["sources"]), {"dense", "bm25"})

        validate = child(root, "validate")["attributes"]
        self.assertTrue(validate["valid"])
        self.assertEqual(validate["cited_ids"], [r.json()["retrieved"][0]["id"]])

    def test_single_retriever_has_no_fusion_span(self):
        got = self.trace_of(self.ask(corpus="own", retrievers=["bm25"]))
        self.assertEqual(names(got["spans"][0]), ["retrieve.bm25", "generate", "validate"])

    def test_existing_response_fields_are_unchanged(self):
        d = self.ask(corpus="own", retrievers=["dense", "bm25"]).json()
        for key in ("answer", "valid", "validation_error", "citations", "retrieved", "retrievers", "backend",
                    "dense_backend", "graph_path", "timings", "usage", "trace_id"):
            self.assertIn(key, d)
        self.assertIn("total_s", d["timings"])
        self.assertIn("generation_latency_s", d["usage"])

    def test_outcome_abstained(self):
        self.llm_reply = {"answer": "That is not in the documents.", "citations": [], "valid": True}
        got = self.trace_of(self.ask(corpus="own", retrievers=["bm25"]))
        self.assertEqual(got["trace"]["outcome"], "abstained")

    def test_outcome_invalid(self):
        self.llm_reply = {"answer": "", "citations": [], "valid": False, "error": "1 validation error(s): bad JSON"}
        got = self.trace_of(self.ask(corpus="own", retrievers=["bm25"]))
        self.assertEqual(got["trace"]["outcome"], "invalid")
        validate = child(got["spans"][0], "validate")["attributes"]
        self.assertEqual((validate["valid"], validate["validation_error"]), (False, "1 validation error(s): bad JSON"))

    def test_outcome_error_when_the_llm_fails(self):
        class RateLimitError(Exception):
            pass

        def busy(question, hits):
            raise RateLimitError("429 Too Many Requests")

        with mock.patch("app.main.generate_answer_with_usage", busy):
            r = self.ask(corpus="own", retrievers=["bm25"])
        self.assertEqual(r.status_code, 503)
        self.assertEqual(self.trace_count(), 1)
        con = sqlite3.connect(os.environ["TRACES_DB"])
        try:
            trace_id, outcome = con.execute("SELECT id, outcome FROM traces").fetchone()
        finally:
            con.close()
        self.assertEqual(outcome, "error")
        [root] = traces.get_trace(trace_id)["spans"]
        generate = child(root, "generate")
        self.assertEqual(generate["status"], "error")
        self.assertEqual(generate["error"], "RateLimitError: 429 Too Many Requests")
        self.assertEqual(root["status"], "error")
        self.assertNotIn("validate", names(root))

    def test_a_400_is_traced_as_an_error_too(self):
        r = self.ask(corpus="own", retrievers=["graph"])
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.trace_count(), 1)

    def test_gen_ai_attributes_and_cost(self):
        def fake(question, hits):
            return ('{"answer": "A.", "citations": ["%s"], "valid": true}' % hits[0]["id"],
                    {"llm_calls": 1, "prompt_tokens": 1200, "completion_tokens": 300, "latency_s": 0.1,
                     "finish_reason": "stop", "thinking": "off", "request_model": "vendor/model-a",
                     "response_model": "vendor/model-a-2026-11"})

        prices = {"LLM_PRICE_IN_PER_1M": "0.5", "LLM_PRICE_OUT_PER_1M": "2"}
        with mock.patch("app.main.generate_answer_with_usage", fake), mock.patch.dict(os.environ, prices):
            got = self.trace_of(self.ask(corpus="own", retrievers=["bm25"]))
        gen = child(got["spans"][0], "generate")["attributes"]
        self.assertEqual(gen["gen_ai.request.model"], "vendor/model-a")
        self.assertEqual(gen["gen_ai.response.model"], "vendor/model-a-2026-11")
        self.assertEqual((gen["gen_ai.usage.input_tokens"], gen["gen_ai.usage.output_tokens"]), (1200, 300))
        self.assertEqual(gen["gen_ai.response.finish_reasons"], ["stop"])
        self.assertEqual(gen["thinking"], "off")
        self.assertAlmostEqual(gen["cost_usd"], 1200 / 1e6 * 0.5 + 300 / 1e6 * 2)
        self.assertIs(gen["price_known"], True)

    def test_unknown_price_is_free_tier(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_PRICE_")}
        with mock.patch.dict(os.environ, env, clear=True):
            got = self.trace_of(self.ask(corpus="own", retrievers=["bm25"]))
        gen = child(got["spans"][0], "generate")["attributes"]
        self.assertEqual((gen["cost_usd"], gen["price_known"]), (0.0, False))

    def test_graph_span_holds_entities_and_results(self):
        runbook = chunk_id("runbook.md")
        path = {"entities": [{"name": "KST-5031", "type": "ErrorCode", "matched_text": "kst-5031"}],
                "edges": [{"source": "KST-5031", "relation": "raised_by", "target": "postroom", "hop": 1,
                           "chunk_ids": [runbook]}],
                "chunks": [runbook]}
        with self.graph_module(fake_graph_module([{"id": runbook, "score": 1.0}], path)):
            got = self.trace_of(self.ask(corpus="kestrel", question="KST-5031", retrievers=["bm25", "graph"]))
        graph = child(got["spans"][0], "retrieve.graph")["attributes"]
        self.assertEqual(graph["entities"], [{"name": "KST-5031", "type": "ErrorCode"}])
        self.assertEqual(graph["edges"], [{"source": "KST-5031", "relation": "raised_by", "target": "postroom", "hop": 1}])
        self.assertEqual(graph["results"][0]["id"], runbook)
        self.assertEqual(graph["llm_calls"], 0)

    def test_no_chunk_text_in_the_store(self):
        self.ask(corpus="own", retrievers=["dense", "bm25"])
        raw = Path(os.environ["TRACES_DB"]).read_bytes()
        for text in DOCS["own"].values():
            self.assertNotIn(text[:40].encode(), raw)


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class FingerprintApiTest(ApiTestCase):
    def test_question_and_evaluation_share_the_fingerprint_and_the_run_file_has_it(self):
        r = self.ask(corpus="own", retrievers=["dense", "bm25"], top_k=core.DEFAULT_TOP_K)
        asked = traces.get_trace(r.json()["trace_id"])["trace"]["fingerprint"]
        e = self.client.post("/api/evaluate", json={"corpus": "own", "retrievers": ["dense", "bm25"]})
        self.assertEqual(e.status_code, 200, e.text)
        summary = e.json()["summary"]
        self.assertEqual(summary["fingerprint"], asked)
        self.assertEqual(summary["config"]["fingerprint"], asked)
        self.assertEqual(core.config_fingerprint(summary["pipeline_config"]), asked)
        saved = (self.data / "results" / "v3" / "own__dense+bm25.json").read_text(encoding="utf-8")
        self.assertIn(f'"fingerprint": "{asked}"', saved)

    def test_changing_top_k_changes_the_fingerprint(self):
        a = self.ask(corpus="own", retrievers=["bm25"], top_k=3).json()["trace_id"]
        b = self.ask(corpus="own", retrievers=["bm25"], top_k=4).json()["trace_id"]
        c = self.ask(corpus="own", retrievers=["bm25"], top_k=3).json()["trace_id"]
        fa, fb, fc = (traces.get_trace(i)["trace"]["fingerprint"] for i in (a, b, c))
        self.assertNotEqual(fa, fb)
        self.assertEqual(fa, fc)


class FingerprintUnitTest(unittest.TestCase):
    RECORDS = [{"id": "doc_md:0", "text": "one"}, {"id": "doc_md:1", "text": "two"}]

    def config(self, **changes):
        args = dict(chunk_size=800, overlap=10, top_k=5, candidates=10, retrievers=["dense", "bm25"], rrf_k=60)
        records = changes.pop("records", self.RECORDS)
        args.update(changes)
        return core.pipeline_config("kestrel", records, **args)

    def test_every_field_is_present(self):
        self.assertEqual(set(self.config()), {
            "answer_prompt_version", "graph_prompt_version", "llm_model", "embedding_model", "chunk_size",
            "overlap", "top_k", "candidates", "retrievers", "rrf_k", "weights", "stopwords", "reranker",
            "corpus_hash"})

    def test_same_config_same_fingerprint_and_order_does_not_matter(self):
        a = core.config_fingerprint(self.config())
        self.assertRegex(a, r"^[0-9a-f]{6}$")
        self.assertEqual(a, core.config_fingerprint(self.config(retrievers=["bm25", "dense", "bm25"])))

    def test_any_change_changes_the_fingerprint(self):
        base = core.config_fingerprint(self.config())
        changed = [
            self.config(top_k=6),
            self.config(retrievers=["dense"]),
            self.config(candidates=30),
            self.config(rrf_k=10),
            self.config(weights={"dense": 2.0}),
            self.config(records=[{"id": "doc_md:0", "text": "one, edited"}, {"id": "doc_md:1", "text": "two"}]),
        ]
        with mock.patch.object(core, "ANSWER_PROMPT_VERSION", "2"):
            changed.append(self.config())
        with mock.patch.dict(os.environ, {"LLM_MODEL": "another/model"}):
            changed.append(self.config())
        for cfg in changed:
            self.assertNotEqual(core.config_fingerprint(cfg), base, cfg)

    def test_fields_that_do_not_apply_are_none(self):
        cfg = self.config(retrievers=["bm25"])
        self.assertIsNone(cfg["embedding_model"])
        self.assertIsNone(cfg["graph_prompt_version"])
        self.assertIsNone(cfg["reranker"])
        self.assertIsNone(self.config(retrievers=["dense"])["stopwords"])


class CostTest(unittest.TestCase):
    def test_prices_set(self):
        with mock.patch.dict(os.environ, {"LLM_PRICE_IN_PER_1M": "3", "LLM_PRICE_OUT_PER_1M": "15"}):
            self.assertEqual(core.llm_cost(2_000_000, 100_000), (7.5, True))

    def test_prices_missing_or_unreadable(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_PRICE_")}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(core.llm_cost(1000, 1000), (0.0, False))
        with mock.patch.dict(os.environ, {"LLM_PRICE_IN_PER_1M": "", "LLM_PRICE_OUT_PER_1M": "2"}):
            self.assertEqual(core.llm_cost(1000, 1000), (0.0, False))


if __name__ == "__main__":
    unittest.main()
