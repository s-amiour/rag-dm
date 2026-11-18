"""v3T15: the API's V3 behaviour, offline: retriever toggles, error codes, attribution, graph path,
/api/config availability, and the evaluation endpoint. Fixtures: tests/api_support.py."""
import importlib.util
import unittest

from tests.api_support import DENSE_ORDER, ApiTestCase, chunk_id, fake_graph_module

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
E47 = chunk_id("a-errors.txt")


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class TogglesTest(ApiTestCase):
    def test_no_retriever_is_a_400(self):
        r = self.ask(corpus="own", retrievers=[])
        self.assertEqual(r.status_code, 400)
        self.assertIn("Switch on at least one retriever", r.json()["detail"])

    def test_unknown_retriever_is_rejected_by_the_request_model(self):
        self.assertEqual(self.ask(corpus="own", retrievers=["foo"]).status_code, 422)

    def test_graph_on_own_corpus_is_a_400(self):
        r = self.ask(corpus="own", retrievers=["graph"])
        self.assertEqual(r.status_code, 400)
        self.assertIn("only built for the Kestrel corpus", r.json()["detail"])

    def test_graph_module_missing_is_a_503(self):
        with self.graph_module(None):
            r = self.ask(corpus="kestrel", question="KST-5031", retrievers=["graph"])
        self.assertEqual(r.status_code, 503)
        self.assertIn("Graph retriever not available", r.json()["detail"])

    def test_duplicate_retriever_counts_once(self):
        r = self.ask(corpus="own", retrievers=["bm25", "bm25"])
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["retrievers"], ["bm25"])


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class AttributionTest(ApiTestCase):
    def test_single_retriever_keeps_its_own_order_and_ranks(self):
        d = self.ask(corpus="own", retrievers=["dense"], top_k=4).json()
        self.assertEqual([h["id"] for h in d["retrieved"]], DENSE_ORDER["own"][:4])
        self.assertEqual([h["sources"]["dense"]["rank"] for h in d["retrieved"]], [1, 2, 3, 4])
        self.assertTrue(all(set(h["sources"]) == {"dense"} for h in d["retrieved"]))

    def test_every_fused_reference_says_who_found_it(self):
        d = self.ask(corpus="own", retrievers=["dense", "bm25"], top_k=4).json()
        for hit in d["retrieved"]:
            self.assertTrue(hit["sources"], hit["id"])
            for source in hit["sources"].values():
                self.assertIn("rank", source)
                self.assertIn("score", source)
        e47 = next(h for h in d["retrieved"] if h["id"] == E47)
        self.assertEqual(set(e47["sources"]), {"dense", "bm25"})  # found by both
        self.assertEqual(e47["sources"]["bm25"]["rank"], 1)
        self.assertEqual(e47["sources"]["dense"]["rank"], len(DENSE_ORDER["own"]))  # dense ranks it last

    def test_citations_are_retrieved_chunks_with_their_sources(self):
        d = self.ask(corpus="own", retrievers=["dense", "bm25"]).json()
        retrieved = {h["id"] for h in d["retrieved"]}
        self.assertTrue(d["citations"])
        for c in d["citations"]:
            self.assertIn(c["id"], retrieved)
            self.assertTrue(c["sources"])
        self.assertEqual(self.llm_calls, 1)

    def test_invalid_llm_reply_is_flagged_not_shown(self):  # v3T12 behaviour at the API level
        self.llm_reply = {"answer": "", "citations": [], "valid": False, "error": "1 validation error(s)"}
        d = self.ask(corpus="own", retrievers=["bm25"]).json()
        self.assertFalse(d["valid"])
        self.assertEqual(d["citations"], [])
        self.assertTrue(d["answer"])  # the readable "could not read the answer" message


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class GraphPathTest(ApiTestCase):
    def test_graph_on_returns_the_path_and_drops_unknown_ids(self):
        runbook = chunk_id("runbook.md")
        fake = fake_graph_module([{"id": runbook, "score": 1.0}, {"id": "not-in-this-chunking:9", "score": 0.5}])
        with self.graph_module(fake):
            d = self.ask(corpus="kestrel", question="KST-5031", retrievers=["dense", "bm25", "graph"]).json()
        self.assertEqual(d["graph_path"]["entities"][0]["name"], "KST-5031")
        ids = {h["id"] for h in d["retrieved"]}
        self.assertNotIn("not-in-this-chunking:9", ids)
        self.assertIn("graph", next(h for h in d["retrieved"] if h["id"] == runbook)["sources"])

    def test_graph_off_has_no_path(self):
        d = self.ask(corpus="kestrel", question="KST-5031", retrievers=["dense", "bm25"]).json()
        self.assertIsNone(d["graph_path"])


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ConfigTest(ApiTestCase):
    def test_graph_availability_follows_the_module(self):
        with self.graph_module(fake_graph_module([])):
            self.assertTrue(self.client.get("/api/config").json()["available"]["graph"])
        with self.graph_module(None):
            self.assertFalse(self.client.get("/api/config").json()["available"]["graph"])


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class EvaluateTest(ApiTestCase):
    def test_default_run_makes_no_llm_call_and_saves_in_the_temp_folder(self):
        r = self.client.post("/api/evaluate", json={"corpus": "own", "retrievers": ["dense", "bm25"]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(self.llm_calls, 0)
        self.assertTrue((self.data / "results" / "v3" / "own__dense+bm25.json").exists())

    def test_no_retriever_is_a_400(self):
        r = self.client.post("/api/evaluate", json={"corpus": "own", "retrievers": []})
        self.assertEqual(r.status_code, 400)


if __name__ == "__main__":
    unittest.main()
