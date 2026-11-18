"""v4T07: Concept map explorer. neighborhood() on a hand-made graph and on the fixture graph of the
graph-search tests, and GET /api/graph/concept with its concept.lookup trace. No LLM, no network."""
import importlib.util
import os
import unittest
from pathlib import Path
from unittest import mock

from app import graph_search, traces
from app.retrieval import RetrieverUnavailable
from tests.api_support import ApiTestCase, chunk_id, fake_graph_module
from tests.test_graph_search import SearchFixture

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None


def toy(edges, node_chunks=None, aliases=None, types=None):
    """A LoadedGraph from [(source, relation, target, [edge chunks])]."""
    ids, names = {}, {}
    for s, _, t, _ in edges:
        for n in (s, t):
            if n not in ids:
                ids[n] = len(ids) + 1
                names[ids[n]] = (n, (types or {}).get(n, "Service"))
    g = graph_search.LoadedGraph("toy-build", names, keys={n.lower(): {i} for n, i in ids.items()})
    for eid, (s, rel, t, chunks) in enumerate(edges, start=1):
        g.edges[eid] = (ids[s], rel, ids[t])
        g.neighbours.setdefault(ids[s], []).append((ids[t], eid))
        g.neighbours.setdefault(ids[t], []).append((ids[s], eid))
        g.edge_chunks[eid] = chunks
    for n, chunks in (node_chunks or {}).items():
        g.node_chunks[ids[n]] = chunks
    for n, written in (aliases or {}).items():
        g.aliases[ids[n]] = written
        for a in written:
            g.alias_keys.setdefault(a.lower(), set()).add(ids[n])
    return g, ids


CHAIN = [("KST-5031", "raised_by", "postroom", ["runbook_md:8"]),
         ("postroom", "owned_by", "Messaging", ["catalog_md:2"]),
         ("postroom", "replaces", "mailman", ["catalog_md:2"])]
TYPES = {"KST-5031": "ErrorCode", "Messaging": "Team"}


class NeighborhoodTests(unittest.TestCase):
    def test_neighbours_relations_directions_and_chunks(self):
        g, _ = toy(CHAIN, node_chunks={"postroom": ["catalog_md:1", "catalog_md:2", "runbook_md:8"]}, types=TYPES)
        out = graph_search.neighborhood(g, "postroom",
                                        chunk_docs={"catalog_md:1": "catalog.md", "catalog_md:2": "catalog.md",
                                                    "runbook_md:8": "runbook.md"})
        self.assertEqual((out["found"], out["matched_by"]), (True, "name"))
        [c] = out["concepts"]
        self.assertEqual((c["name"], c["type"], c["degree"], c["more_edges"]), ("postroom", "Service", 3, 0))
        self.assertEqual([(e["relation"], e["direction"], e["neighbour"]["name"], e["chunks"]) for e in c["edges"]], [
            ("owned_by", "out", "Messaging", ["catalog_md:2"]),
            ("raised_by", "in", "KST-5031", ["runbook_md:8"]),
            ("replaces", "out", "mailman", ["catalog_md:2"]),
        ])
        self.assertEqual(c["edges"][1]["neighbour"]["type"], "ErrorCode")
        self.assertEqual(c["documents"], [{"document": "catalog.md", "chunks": ["catalog_md:1", "catalog_md:2"]},
                                          {"document": "runbook.md", "chunks": ["runbook_md:8"]}])

    def test_found_by_alias_and_by_text(self):
        g, _ = toy(CHAIN, aliases={"postroom": ["letterbox"]}, types=TYPES)
        out = graph_search.neighborhood(g, "Letterbox")
        self.assertEqual((out["matched_by"], [c["name"] for c in out["concepts"]]), ("alias", ["postroom"]))
        self.assertEqual(out["concepts"][0]["aliases"], ["letterbox"])
        out = graph_search.neighborhood(g, "error KST-5031 again")       # more than the name: the question matcher
        self.assertEqual((out["matched_by"], [c["name"] for c in out["concepts"]]), ("text", ["KST-5031"]))

    def test_unknown_concept_is_not_an_error(self):
        g, _ = toy(CHAIN, types=TYPES)
        out = graph_search.neighborhood(g, "KST-5038")
        self.assertEqual((out["found"], out["matched_by"], out["concepts"]), (False, "none", []))

    def test_hub_cap_limits_the_list_and_flags_hub_neighbours(self):
        edges = [("Hub", "has_feature", f"F{i:02}", [f"c:{i}"]) for i in range(20)] + [("Alpha", "depends_on", "Hub", ["c:a"])]
        g, _ = toy(edges)
        [hub] = graph_search.neighborhood(g, "Hub")["concepts"]
        self.assertEqual((hub["degree"], len(hub["edges"]), hub["more_edges"]), (21, graph_search.HUB_CAP, 21 - graph_search.HUB_CAP))
        [a] = graph_search.neighborhood(g, "Alpha")["concepts"]
        self.assertEqual((a["edges"][0]["neighbour"]["name"], a["edges"][0]["neighbour"]["hub"]), ("Hub", True))
        self.assertEqual(a["edges"][0]["neighbour"]["degree"], 21)

    def test_a_concept_without_edges(self):
        g, ids = toy(CHAIN, types=TYPES)
        g.names[99] = ("Platform", "Team")
        g.keys["platform"] = {99}
        g.node_chunks[99] = ["catalog_md:3"]
        [c] = graph_search.neighborhood(g, "Platform")["concepts"]
        self.assertEqual((c["degree"], c["edges"], c["chunks"]), (0, [], ["catalog_md:3"]))


class FixtureGraphTests(SearchFixture):
    """The real build path (graph.build + graph_search.load) on the fixture of test_graph_search."""

    def concept(self, name):
        out = graph_search.neighborhood(graph_search.load("kestrel", 800, 10), name)
        self.assertTrue(out["found"], name)
        return out["concepts"][0], out["matched_by"]

    def test_kst_5031_shows_postroom_with_its_source_chunk(self):
        c, _ = self.concept("KST-5031")
        [edge] = [e for e in c["edges"] if e["neighbour"]["name"] == "postroom"]
        self.assertEqual((edge["relation"], edge["direction"]), ("raised_by", "out"))
        self.assertIn("support-runbook-errors_md:8", edge["chunks"])

    def test_postroom_links_three_ways(self):
        c, _ = self.concept("postroom")
        self.assertEqual(sorted((e["relation"], e["neighbour"]["name"]) for e in c["edges"]),
                         [("owned_by", "Messaging"), ("raised_by", "KST-5031"), ("replaces", "mailman")])

    def test_mailman_is_its_own_concept_not_postroom(self):
        c, _ = self.concept("mailman")
        self.assertEqual(c["name"], "mailman")
        self.assertEqual([(e["relation"], e["direction"], e["neighbour"]["name"]) for e in c["edges"]],
                         [("replaces", "in", "postroom")])

    def test_a_merged_name_finds_the_merged_node(self):
        c, _ = self.concept("the Messaging team")
        self.assertEqual(c["name"], "Messaging")
        self.assertEqual(c["chunks"], ["service-catalog_md:2", "service-catalog_md:3"])


KESTREL_CHAIN = [("KST-5031", "raised_by", "postroom", [chunk_id("runbook.md")]),
                 ("postroom", "owned_by", "Messaging", [chunk_id("catalog.md")])]


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class ConceptApiTest(ApiTestCase):
    def setUp(self):
        super().setUp()
        g, _ = toy(KESTREL_CHAIN, node_chunks={"KST-5031": [chunk_id("runbook.md")]}, types=TYPES)
        p = mock.patch.object(graph_search, "load", lambda *a, **k: g)
        p.start()
        self.addCleanup(p.stop)

    def get(self, name, **params):
        return self.client.get("/api/graph/concept", params={"name": name, **params})

    def test_lookup_returns_the_neighborhood_and_the_passages(self):
        r = self.get("KST-5031")
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertTrue(d["found"])
        self.assertIsNone(d["message"])
        [c] = d["concepts"]
        self.assertEqual([(e["relation"], e["neighbour"]["name"]) for e in c["edges"]], [("raised_by", "postroom")])
        self.assertEqual(c["documents"], [{"document": "runbook.md", "chunks": [chunk_id("runbook.md")]}])
        self.assertIn("raised by postroom", d["chunk_texts"][chunk_id("runbook.md")])

    def test_one_concept_lookup_trace_without_chunk_text(self):
        d = self.get("postroom").json()
        got = traces.get_trace(d["trace_id"])
        self.assertEqual((got["trace"]["kind"], got["trace"]["corpus"], got["trace"]["outcome"]),
                         ("concept.lookup", "kestrel", "found"))
        [root] = got["spans"]
        self.assertEqual(root["attributes"]["concept"], "postroom")
        self.assertEqual([c["name"] for c in root["children"]], ["graph.load", "concept.match"])
        match = root["children"][1]["attributes"]
        self.assertEqual((match["matched_by"], match["concepts"], match["edges"]),
                         ("name", [{"name": "postroom", "type": "Service"}], 2))
        raw = Path(os.environ["TRACES_DB"]).read_bytes()
        self.assertNotIn(b"raised by postroom", raw)

    def test_unknown_concept_is_a_plain_message(self):
        d = self.get("KST-5038").json()
        self.assertFalse(d["found"])
        self.assertIn("No concept called", d["message"])
        self.assertEqual(traces.get_trace(d["trace_id"])["trace"]["outcome"], "not_found")

    def test_missing_graph_is_a_503_with_a_plain_message(self):
        def missing(*a, **k):
            raise RetrieverUnavailable("Graph is missing for this chunking (800/10). Run the graph build: ...")

        with mock.patch.object(graph_search, "load", missing):
            r = self.get("KST-5031")
        self.assertEqual(r.status_code, 503)
        self.assertIn("The concept map needs the Kestrel concept graph", r.json()["detail"])

    def test_own_corpus_is_a_400(self):
        self.assertEqual(self.get("E47", corpus="own").status_code, 400)

    def test_ask_about_this_question_goes_through_the_graph(self):
        """What the tab's button sends: Kestrel, graph on, a question naming the concept."""
        hits = [{"id": chunk_id("runbook.md"), "score": 1.0}]
        with self.graph_module(fake_graph_module(hits)):
            r = self.ask(corpus="kestrel", question="What do the documents say about KST-5031?",
                         retrievers=["dense", "bm25", "graph"])
        self.assertEqual(r.status_code, 200, r.text)
        [root] = traces.get_trace(r.json()["trace_id"])["spans"]
        graph = next(c for c in root["children"] if c["name"] == "retrieve.graph")
        self.assertEqual(graph["attributes"]["results"][0]["id"], chunk_id("runbook.md"))


if __name__ == "__main__":
    unittest.main()
