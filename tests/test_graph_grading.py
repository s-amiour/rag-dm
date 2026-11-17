"""Grading a sample of graph triples (v3T06), on a fixture graph built with a fake LLM."""
import json
import tempfile
from pathlib import Path

from app import graph, graph_grading
from tests.test_graph import FakeLLM, GraphTestCase, RECORDS


class SampleTests(GraphTestCase):
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
