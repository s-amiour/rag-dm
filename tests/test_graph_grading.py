"""Grading a sample of graph triples (v3T06), on a fixture graph built with a fake LLM."""
import contextlib
import io
import json
from pathlib import Path
from unittest import mock

from app import graph, graph_grading
from tests.test_graph import FakeLLM, GraphTestCase, RECORDS


class GradingFixture(GraphTestCase):
    """A small graph built with the fake LLM (no test of its own)."""

    def setUp(self):
        super().setUp()
        graph.build("kestrel", records=RECORDS, llm=FakeLLM(), delay_s=0, log=lambda *_: None)
        self.out = Path(self.tmp.name) / "v3"

    def draw(self, n=3, seed=42):
        con = graph.connect("kestrel")
        try:
            return graph_grading.sample(con, RECORDS, n, seed)
        finally:
            con.close()

    def triples(self, data):
        return [(t["source"], t["relation"], t["target"]) for t in data["triples"]]


class SampleTests(GradingFixture):
    def test_same_seed_same_graph_same_triples(self):
        self.assertEqual(self.triples(self.draw()), self.triples(self.draw()))
        self.assertEqual(self.draw()["meta"]["graph_hash"], self.draw()["meta"]["graph_hash"])

    def test_another_seed_another_order(self):
        self.assertNotEqual(self.triples(self.draw(n=4, seed=1)), self.triples(self.draw(n=4, seed=2)))

    def test_rebuilt_graph_gets_another_hash(self):
        before = self.draw()["meta"]["graph_hash"]
        graph.build("kestrel", records=RECORDS, llm=FakeLLM(), delay_s=0, log=lambda *_: None, rules=())
        self.assertEqual(self.draw()["meta"]["graph_hash"], before)       # nothing to merge: same graph
        con = graph.connect("kestrel")
        con.execute("DELETE FROM evidence WHERE kind='edge' AND owner_id=(SELECT MIN(id) FROM edges)")
        con.execute("DELETE FROM edges WHERE id=(SELECT MIN(id) FROM edges)")
        con.commit(); con.close()
        self.assertNotEqual(self.draw()["meta"]["graph_hash"], before)

    def test_each_triple_has_what_the_grader_needs(self):
        data = self.draw(n=10)
        self.assertEqual(data["meta"]["n"], 4)                             # the fixture has 4 edges
        self.assertEqual(data["meta"]["seed"], 42)
        for t in data["triples"]:
            self.assertTrue(t["evidence"])
            for e in t["evidence"]:
                self.assertTrue(e["chunk_id"] and e["quote"])
                self.assertIn(e["quote"].replace("**", "")[:20], e["excerpt"].replace("**", ""))
            self.assertEqual((t["grade"], t["note"]), ("", ""))

    def test_a_graded_file_is_not_overwritten(self):
        data = self.draw()
        path = graph_grading.write_sample(data, self.out)
        graded = json.loads(path.read_text(encoding="utf-8"))
        graded["triples"][0]["grade"] = "correct"
        path.write_text(json.dumps(graded), encoding="utf-8")
        with self.assertRaises(FileExistsError):
            graph_grading.write_sample(data, self.out)
        graph_grading.write_sample(data, self.out, force=True)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["triples"][0]["grade"], "")


class ChunkCommandTests(GraphTestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(graph.core, "build_chunks", lambda *a: RECORDS), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = graph.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_prints_the_whole_chunk(self):
        code, out, _ = self.run_cli("chunk", "service-catalog_md:0")
        self.assertEqual(code, 0)
        self.assertIn("Owner: **Messaging**.", out)
        self.assertIn("postroom replaced the older mailman service", out)

    def test_unknown_chunk(self):
        code, _, err = self.run_cli("chunk", "service-catalog_md:99")
        self.assertEqual(code, 1)
        self.assertIn("No chunk", err)


class GradesTests(GradingFixture):
    def graded(self):
        data = self.draw(n=4)
        for t in data["triples"]:
            t.update(grade="correct", graded_chunk=t["evidence"][0]["chunk_id"], note="stated in the chunk")
        data["meta"].update(graded_by="A", reviewed_by="B")
        return data

    def test_counts_and_no_problem(self):
        data = self.graded()
        data["triples"][0]["grade"] = "wrong"
        res = graph_grading.check(data, data["meta"]["graph_hash"])
        self.assertEqual(res["counts"], {"correct": 3, "wrong": 1, "unsupported": 0})
        self.assertEqual((res["problems"], res["warnings"]), ([], []))

    def test_only_the_three_grades(self):
        data = self.graded()
        data["triples"][0]["grade"] = "partly correct"
        res = graph_grading.check(data)
        self.assertEqual(len(res["problems"]), 1)
        self.assertIn("not one of correct, wrong, unsupported", res["problems"][0])

    def test_note_and_evidence_chunk_required(self):
        data = self.graded()
        data["triples"][0]["note"] = ""
        data["triples"][1]["graded_chunk"] = "some-other-chunk:0"
        problems = graph_grading.check(data)["problems"]
        self.assertTrue(any("note" in p for p in problems))
        self.assertTrue(any("graded_chunk" in p for p in problems))

    def test_warnings_ungraded_reviewer_and_rebuilt_graph(self):
        data = self.draw(n=4)
        res = graph_grading.check(data, current_hash="something else")
        self.assertEqual(res["ungraded"], 4)
        self.assertEqual(len(res["warnings"]), 4)
        self.assertTrue(any("graph changed" in w for w in res["warnings"]))

    def test_render_has_counts_table_and_excerpts(self):
        data = self.graded()
        text = graph_grading.render(data, graph_grading.check(data))
        self.assertIn("**Counts: correct 4 / wrong 0 / unsupported 0**", text)
        self.assertIn("seed 42", text)
        self.assertIn(data["meta"]["graph_hash"][:16], text)
        self.assertEqual(text.count("| correct |"), 4)
        self.assertIn("## Chunk excerpts", text)
