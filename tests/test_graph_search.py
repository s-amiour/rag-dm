"""Graph search (v3T03) on the fixture graph of the resolution tests: no LLM, no network."""
from unittest import mock

from app import core, graph, graph_search
from app.retrieval import RetrieverUnavailable
from tests.test_graph import FakeLLM, GraphTestCase, rec
from tests.test_graph_resolve import RECORDS, SCRIPT


class SearchFixture(GraphTestCase):
    """runbook (KST-5031 raised_by postroom) -> catalog (postroom owned_by Messaging, replaces mailman)
    -> team table ("the Messaging team", merged into Messaging)."""

    def setUp(self):
        super().setUp()
        graph_search._CACHE.clear()
        graph.build("kestrel", records=RECORDS, llm=FakeLLM({k: list(v) for k, v in SCRIPT.items()}),
                    delay_s=0, log=lambda *_: None)
        p = mock.patch.object(core, "build_chunks", lambda *a: RECORDS)
        p.start()
        self.addCleanup(p.stop)


class LoadTests(SearchFixture):
    def test_loads_nodes_edges_evidence_and_aliases(self):
        g = graph_search.load("kestrel", 800, 10)
        self.assertEqual(sorted(n for n, _ in g.names.values()), ["KST-5031", "Messaging", "mailman", "postroom"])
        self.assertEqual(len(g.edges), 3)
        messaging = next(i for i, (n, _) in g.names.items() if n == "Messaging")
        self.assertIn(messaging, g.keys["messaging"])
        self.assertEqual(g.node_chunks[messaging], ["service-catalog_md:2", "service-catalog_md:3"])

    def test_cached_per_build(self):
        self.assertIs(graph_search.load("kestrel", 800, 10), graph_search.load("kestrel", 800, 10))

    def test_stale_graph_raises_with_the_rebuild_message(self):
        changed = [*RECORDS[:2], rec("service-catalog.md", 3, RECORDS[2]["text"] + " (edited)")]
        with self.assertRaises(RetrieverUnavailable) as ctx:
            graph_search.load("kestrel", 800, 10, records=changed)
        self.assertIn("out of date for this chunking", str(ctx.exception))
        self.assertIn("python -m app.graph build", str(ctx.exception))

    def test_missing_graph_raises(self):
        graph.db_path("kestrel").unlink()
        with self.assertRaises(RetrieverUnavailable) as ctx:
            graph_search.load("kestrel", 800, 10)
        self.assertIn("missing", str(ctx.exception))


class MatchTests(SearchFixture):
    def names(self, question):
        return [(e["name"], e["matched_text"]) for e in graph_search.match_entities(
            graph_search.load("kestrel", 800, 10), question)]

    def test_identifier_and_alias(self):
        self.assertEqual(self.names("Error KST-5031 happened twice. Which team should I contact?"),
                         [("KST-5031", "kst-5031")])
        self.assertEqual(self.names("Who owns the Messaging team?"), [("Messaging", "messaging team")])

    def test_kst_5031_never_matches_kst_5038(self):
        self.assertEqual(self.names("What is KST-5038?"), [])

    def test_generic_and_stop_words_alone_match_nothing(self):
        self.assertEqual(self.names("Which team or service should support contact, and where?"), [])

    def test_one_node_once_longest_match_first(self):
        self.assertEqual(self.names("The mailman service and mailman are failing"), [("mailman", "mailman service")])

    def test_alias_used_only_when_no_node_has_the_name(self):
        g = graph_search.load("kestrel", 800, 10)
        postroom = next(i for i, (n, _) in g.names.items() if n == "postroom")
        g.alias_keys.setdefault("letterbox", set()).add(postroom)          # an alias no node is named
        self.assertEqual(self.names("Is letterbox down?"), [("postroom", "letterbox")])


def toy_graph(edges, chunk_of_edge=lambda eid: f"c:{eid}"):
    """A hand-made LoadedGraph: edges = [(source name, relation, target name)]."""
    names, ids = {}, {}
    for s, _, t in edges:
        for n in (s, t):
            if n not in ids:
                ids[n] = len(ids) + 1
                names[ids[n]] = (n, "Service")
    g = graph_search.LoadedGraph("toy", names, keys={n.lower(): {i} for n, i in ids.items()})
    for eid, (s, rel, t) in enumerate(edges, start=1):
        g.edges[eid] = (ids[s], rel, ids[t])
        g.neighbours.setdefault(ids[s], []).append((ids[t], eid))
        g.neighbours.setdefault(ids[t], []).append((ids[s], eid))
        g.edge_chunks[eid] = [chunk_of_edge(eid)]
    return g, ids


class WalkTests(SearchFixture):
    def test_two_hops_both_directions(self):
        g = graph_search.load("kestrel", 800, 10)
        start = next(i for i, (n, _) in g.names.items() if n == "KST-5031")
        w = graph_search.walk(g, [start])
        hops = {g.names[n][0]: h for n, h in w.reached.items()}
        self.assertEqual(hops, {"KST-5031": 0, "postroom": 1, "Messaging": 2, "mailman": 2})
        self.assertEqual(sorted(w.followed.values()), [1, 2, 2])

    def test_hub_above_the_cap_is_reached_but_not_expanded(self):
        edges = [("A", "depends_on", "Hub")] + [("Hub", "has_feature", f"F{i}") for i in range(16)]
        g, ids = toy_graph(edges)
        w = graph_search.walk(g, [ids["A"]])
        self.assertIn(ids["Hub"], w.reached)
        self.assertEqual(w.capped, [ids["Hub"]])
        self.assertNotIn(ids["F0"], w.reached)
        g, ids = toy_graph(edges[:15])                                    # 15 edges: not above the cap
        self.assertIn(ids["F0"], graph_search.walk(g, [ids["A"]]).reached)

    def test_a_start_entity_is_expanded_even_if_it_is_a_hub(self):
        g, ids = toy_graph([("Hub", "has_feature", f"F{i}") for i in range(20)])
        w = graph_search.walk(g, [ids["Hub"]])
        self.assertEqual((len(w.reached), w.capped), (21, []))


class RankTests(SearchFixture):
    Q = "Error KST-5031 happened twice. Which team should I contact, and where?"

    def search(self, question, k=10):
        return graph_search.search_graph(graph_search.load("kestrel", 800, 10), question, k,
                                         {r["id"] for r in RECORDS})

    def test_two_hop_chain_returns_both_chunks(self):
        out = self.search(self.Q)
        ids = [h["id"] for h in out["hits"]]
        # hop 1: the runbook entry; hop 2: the team table, reached through "the Messaging team" merge
        self.assertEqual(ids, ["support-runbook-errors_md:8", "service-catalog_md:2", "service-catalog_md:3"])
        self.assertEqual([h["score"] for h in out["hits"]], [1.0, 0.5, 1 / 3])

    def test_path_for_the_ui(self):
        path = self.search(self.Q)["path"]
        self.assertEqual([(e["name"], e["matched_text"]) for e in path["entities"]], [("KST-5031", "kst-5031")])
        self.assertEqual(path["edges"][0], {"source": "KST-5031", "relation": "raised_by", "target": "postroom",
                                            "hop": 1, "chunk_ids": ["support-runbook-errors_md:8"]})
        self.assertEqual({e["hop"] for e in path["edges"]}, {1, 2})
        self.assertEqual(path["chunks"], [h["id"] for h in self.search(self.Q)["hits"]])

    def test_no_entity_gives_empty_hits_and_path(self):
        out = self.search("How do I change my password?")
        self.assertEqual(out["hits"], [])
        self.assertEqual((out["path"]["entities"], out["path"]["edges"], out["path"]["chunks"]), ([], [], []))

    def test_same_input_same_ranking(self):
        self.assertEqual(self.search(self.Q), self.search(self.Q))

    def test_k_and_unknown_chunk_ids(self):
        self.assertEqual(len(self.search(self.Q, k=1)["hits"]), 1)
        g = graph_search.load("kestrel", 800, 10)
        out = graph_search.search_graph(g, self.Q, 10, {"support-runbook-errors_md:8"})
        self.assertEqual([h["id"] for h in out["hits"]], ["support-runbook-errors_md:8"])

    def test_ties_broken_by_supported_edges_then_chunk_id(self):
        g, ids = toy_graph([("Alpha", "depends_on", "Beta"), ("Alpha", "depends_on", "Gamma")],
                           chunk_of_edge=lambda eid: "c:shared" if eid == 1 else "c:other")
        g.edge_chunks[2] = ["c:shared", "c:z"]
        g.keys = {"alpha": {ids["Alpha"]}}
        out = graph_search.search_graph(g, "what about alpha", 10)
        self.assertEqual([h["id"] for h in out["hits"]], ["c:shared", "c:z"])   # 2 edges, then 1


class ContractTests(SearchFixture):
    Q = RankTests.Q

    def test_search_contract(self):
        from app import graph_retriever
        out = graph_retriever.search("kestrel", self.Q, 10, 800, 10)
        self.assertEqual(set(out), {"hits", "path", "usage"})
        self.assertEqual(out["usage"], {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        self.assertTrue(all(set(h) == {"id", "score"} for h in out["hits"]))
        self.assertTrue({"entities", "edges", "chunks"} <= set(out["path"]))

    def test_retrieve_with_graph_only_and_with_fusion(self):
        from app import retrieval
        out = retrieval.retrieve("kestrel", self.Q, 800, 10, ["graph"], depth=5, records=RECORDS)
        self.assertEqual(out["hits"][0]["id"], "support-runbook-errors_md:8")
        self.assertEqual(out["hits"][0]["sources"], {"graph": {"rank": 1, "score": 1.0}})
        self.assertTrue(out["graph_path"]["edges"])
        self.assertEqual(out["usage"], {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        fused = retrieval.retrieve("kestrel", self.Q, 800, 10, ["bm25", "graph"], depth=5, records=RECORDS)
        self.assertEqual(fused["retrievers"], ["bm25", "graph"])
        self.assertTrue(any("graph" in h["sources"] for h in fused["hits"]))

    def test_stale_graph_is_a_503_not_an_empty_result(self):
        from app import retrieval
        changed = [*RECORDS[:2], rec("service-catalog.md", 3, RECORDS[2]["text"] + " (edited)")]
        with mock.patch.object(core, "build_chunks", lambda *a: changed), \
                self.assertRaises(retrieval.RetrieverUnavailable) as ctx:
            retrieval.retrieve("kestrel", self.Q, 800, 10, ["graph"], records=changed)
        self.assertIn("Run the graph build", str(ctx.exception))

    def test_other_corpus(self):
        from app import graph_retriever
        with self.assertRaises(RetrieverUnavailable):
            graph_retriever.search("own", self.Q, 10, 500, 20)
