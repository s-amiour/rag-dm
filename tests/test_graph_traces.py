"""v4T03: the index trace of a graph build, with the fake LLM of tests/test_graph.py (no key, no network)."""
import os
import unittest
from unittest import mock

from app import graph, traces
from tests.test_graph import RECORDS, FakeLLM, GraphTestCase, rec

SKIPPED = rec("billing.md", 0, "Refunds are processed within ten working days.")
FOUR = [*RECORDS, SKIPPED]
# runbook: valid at once; catalog: invalid then valid (retried); team table: valid with 0 relations;
# billing: invalid twice, nothing to keep (skipped)
SCRIPT = {"service-catalog_md:0": ["not json at all"], "billing_md:0": ["nope", "nope"]}


class NoCalls(FakeLLM):
    def complete(self, messages):
        raise AssertionError("the LLM must not be called")


def extract_spans(trace_id):
    [root] = traces.get_trace(trace_id)["spans"]
    return root, {c["attributes"]["chunk_id"]: c for c in root["children"] if c["name"] == "graph.extract"}


class GraphBuildTraceTest(GraphTestCase):
    def build_traced(self, llm, records=FOUR, **kw):
        with traces.trace("index", kind="index", corpus="kestrel") as t:
            summary = self.run_build(llm, records=records, trace=t, **kw)
        return t, summary

    def test_one_trace_one_span_per_chunk_failures_visible(self):
        prices = {"GRAPH_LLM_MODEL": "fake-model", "GRAPH_LLM_PRICE_IN_PER_1M": "1", "GRAPH_LLM_PRICE_OUT_PER_1M": "2"}
        with mock.patch.dict(os.environ, prices):
            t, s = self.build_traced(FakeLLM(SCRIPT))
        self.assertEqual((s["extracted"], s["retried"], s["skipped"]), (3, 1, 1))
        got = traces.get_trace(t.id)
        self.assertEqual((got["trace"]["kind"], got["trace"]["outcome"]), ("index", "ok"))
        root, spans = extract_spans(t.id)
        self.assertEqual([c["name"] for c in root["children"]],
                         ["load+chunk", *["graph.extract"] * 4, "graph.resolve"])
        self.assertEqual(root["attributes"]["trigger"], "graph.build")
        self.assertEqual(root["attributes"]["build_id"], s["build_id"])
        self.assertEqual(root["attributes"]["prompt_version"], graph.PROMPT_VERSION)

        ok = spans["support-runbook-errors_md:0"]
        self.assertEqual((ok["status"], ok["attributes"]["result"], ok["attributes"]["attempts"]), ("ok", "ok", 1))
        self.assertEqual(ok["attributes"]["relations"], 2)

        retried = spans["service-catalog_md:0"]["attributes"]
        self.assertEqual((retried["result"], retried["attempts"], retried["call_outcomes"]),
                         ("retried", 2, ["invalid", "retried"]))
        self.assertEqual((retried["gen_ai.usage.input_tokens"], retried["gen_ai.usage.output_tokens"]), (200, 40))
        self.assertEqual(retried["gen_ai.request.model"], "fake-model")
        self.assertAlmostEqual(retried["cost_usd"], 200 / 1e6 * 1 + 40 / 1e6 * 2)
        self.assertIs(retried["price_known"], True)

        empty = spans["service-catalog_md:1"]
        self.assertEqual((empty["status"], empty["attributes"]["result"], empty["attributes"]["relations"]), ("ok", "ok", 0))

        skipped = spans["billing_md:0"]
        self.assertEqual((skipped["status"], skipped["attributes"]["result"]), ("error", "skipped"))
        self.assertTrue(skipped["error"])
        self.assertLessEqual(len(skipped["error"]), traces.MAX_ERROR)
        self.assertNotIn("raw_reply", skipped["attributes"])

        resolve = next(c for c in root["children"] if c["name"] == "graph.resolve")["attributes"]
        self.assertEqual((resolve["nodes"], resolve["edges"]), (s["entities"], s["relations"]))

    def test_tokens_per_chunk_equal_the_call_log(self):
        t, _ = self.build_traced(FakeLLM(SCRIPT))
        _, spans = extract_spans(t.id)
        for chunk_id, span in spans.items():
            [(prompt, completion, calls)] = self.query(
                "SELECT COALESCE(SUM(prompt_tokens),0), COALESCE(SUM(completion_tokens),0), COUNT(*) "
                "FROM llm_calls WHERE chunk_id=?", chunk_id)
            a = span["attributes"]
            self.assertEqual((a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"], a["attempts"]),
                             (prompt, completion, calls), chunk_id)

    def test_salvaged_chunk_is_partial_and_ok(self):
        invented = ('{"entities": [{"name": "postroom", "type": "Service"}, {"name": "Ghost", "type": "Team"}], '
                    '"relations": []}')
        t, s = self.build_traced(FakeLLM({"service-catalog_md:0": [invented, invented]}), records=RECORDS)
        self.assertEqual(s["partial"], 1)
        _, spans = extract_spans(t.id)
        span = spans["service-catalog_md:0"]
        self.assertEqual((span["status"], span["attributes"]["result"], span["attributes"]["dropped_items"]),
                         ("ok", "partial", 1))

    def test_cached_rerun_flags_every_chunk_and_makes_no_call(self):
        self.build_traced(FakeLLM(SCRIPT))
        t, s = self.build_traced(NoCalls())
        self.assertEqual(s["run"]["llm_calls"], 0)
        _, spans = extract_spans(t.id)
        self.assertEqual(len(spans), 4)
        for chunk_id, span in spans.items():
            a = span["attributes"]
            self.assertEqual((a["cached"], a["result"], a["attempts"], a["gen_ai.usage.input_tokens"]),
                             (True, "cached", 0, 0), chunk_id)
        self.assertEqual(spans["billing_md:0"]["status"], "error")           # still missing from the graph
        self.assertEqual(spans["billing_md:0"]["attributes"]["cached_status"], "skipped")
        self.assertEqual(spans["service-catalog_md:1"]["attributes"]["relations"], 0)

    def test_fatal_error_stops_the_build_and_the_trace_says_so(self):
        llm = FakeLLM({"support-runbook-errors_md:0": [graph.FatalLLMError("AuthenticationError: 401")]})
        t, s = self.build_traced(llm, records=RECORDS)          # as today: the build stops calling and reports
        self.assertIn("401", s["aborted"])
        got = traces.get_trace(t.id)
        self.assertEqual(got["trace"]["outcome"], "error")
        _, spans = extract_spans(t.id)
        first = spans["support-runbook-errors_md:0"]
        self.assertEqual((first["status"], first["attributes"]["result"]), ("error", "error"))
        self.assertIn("401", first["error"])
        for chunk_id in ("service-catalog_md:0", "service-catalog_md:1"):
            self.assertEqual(spans[chunk_id]["attributes"]["result"], "not_attempted")

    def test_an_exception_in_the_build_closes_the_trace_and_propagates(self):
        holder = {}
        with self.assertRaises(RuntimeError):
            with traces.trace("index", kind="index", corpus="kestrel") as t:
                holder["t"] = t
                self.run_build(FakeLLM(), records=[], trace=t)
        got = traces.get_trace(holder["t"].id)
        self.assertEqual(got["trace"]["outcome"], "error")
        [root] = got["spans"]
        self.assertEqual(root["status"], "error")
        self.assertEqual(root["children"][0]["attributes"]["chunks"], 0)

    def test_without_a_trace_the_build_is_unchanged(self):
        traced, _ = self.build_traced(FakeLLM(SCRIPT))
        first = self.query("SELECT chunk_id, attempt, outcome, prompt_tokens FROM llm_calls ORDER BY id")
        graph.db_path("kestrel").unlink()
        s = self.run_build(FakeLLM(SCRIPT), records=FOUR)
        second = self.query("SELECT chunk_id, attempt, outcome, prompt_tokens FROM llm_calls ORDER BY id")
        self.assertEqual(first, second)
        self.assertEqual((s["extracted"], s["retried"], s["skipped"]), (3, 1, 1))


if __name__ == "__main__":
    unittest.main()
