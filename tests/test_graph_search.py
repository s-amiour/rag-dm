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
