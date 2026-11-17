"""Query-time side of the concept graph (ticket v3T03): load the resolved graph,
find the question's entities, walk, rank chunks. No LLM call.

app/graph_retriever.py wraps this in the search() contract of retrieval.py.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from . import core
from . import graph as graph_store
from . import graph_resolve
from .retrieval import RetrieverUnavailable


@dataclass
class LoadedGraph:
    build_id: str
    names: dict[int, tuple[str, str]]                         # node id -> (name, type)
    keys: dict[str, set[int]] = field(default_factory=dict)   # normalised name or alias -> node ids
    edges: dict[int, tuple[int, str, int]] = field(default_factory=dict)   # edge id -> (source, relation, target)
    neighbours: dict[int, list[tuple[int, int]]] = field(default_factory=dict)  # node -> [(other node, edge id)]
    node_chunks: dict[int, list[str]] = field(default_factory=dict)
    edge_chunks: dict[int, list[str]] = field(default_factory=dict)

    def degree(self, node: int) -> int:
        return len({eid for _, eid in self.neighbours.get(node, [])})


_CACHE: dict[tuple, LoadedGraph] = {}


def load(corpus: str, chunk_size: int, overlap: int,
         records: list[dict] | None = None) -> LoadedGraph:
    """The resolved graph for this corpus and chunking, from memory if this build was
    already loaded. A missing graph, or one built for other chunks, raises
    RetrieverUnavailable (503 in the API) instead of silently returning nothing."""
    if records is None:
        records = core.build_chunks(corpus, chunk_size, overlap)
    status = graph_store.graph_status(corpus, chunk_size, overlap, records=records)
    if status["state"] != "fresh":
        raise RetrieverUnavailable(
            f"Graph is {'missing' if status['state'] == 'missing' else 'out of date'} for this chunking "
            f"({chunk_size}/{overlap}). Run the graph build: "
            f"python -m app.graph build --corpus {corpus} --chunk-size {chunk_size} --overlap {overlap}"
        )

    key = (corpus, chunk_size, overlap, status["fingerprint"], status.get("build_id", ""))
    if key in _CACHE:
        return _CACHE[key]

    con = graph_store.connect(corpus)
    try:
        g = LoadedGraph(status.get("build_id", ""),
                        {i: (n, t) for i, n, t in con.execute("SELECT id, name, type FROM nodes")})
        keys: dict[str, set[int]] = defaultdict(set)
        for i, (name, _) in g.names.items():
            keys[graph_resolve.normalise_name(name)].add(i)
        for i, alias in con.execute("SELECT node_id, alias FROM aliases"):
            keys[graph_resolve.normalise_name(alias)].add(i)
        g.keys = dict(keys)
        neighbours: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for eid, src, rel, dst in con.execute("SELECT id, source_id, relation, target_id FROM edges ORDER BY id"):
            g.edges[eid] = (src, rel, dst)
            neighbours[src].append((dst, eid))
            neighbours[dst].append((src, eid))
        g.neighbours = dict(neighbours)
        for kind, target in (("node", g.node_chunks), ("edge", g.edge_chunks)):
            for owner, chunk in con.execute(
                    "SELECT DISTINCT owner_id, chunk_id FROM evidence WHERE kind=? ORDER BY chunk_id", (kind,)):
                target.setdefault(owner, []).append(chunk)
    finally:
        con.close()

    _CACHE.clear()  # one build per process is enough; a rebuild replaces it
    _CACHE[key] = g
    return g
