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
from .bm25 import STOPWORDS
from .retrieval import RetrieverUnavailable

MAX_NGRAM = 4
MAX_HOPS = 2
HUB_CAP = 15   # an intermediate node with more edges than this is reached but not expanded
# Single words that are nodes of our graph but name nothing specific in a question: as start
# entities they would pull in unrelated chunks ("team" is both a Team and the Team plan, "support"
# a Service node). Measured on the 12 Kestrel questions, DECISIONS D-29.
GENERIC_WORDS = frozenset({
    "team", "teams", "service", "services", "support", "admin", "admins", "owner", "plan", "plans",
    "kestrel", "workspace", "user", "users", "customer", "customers", "error", "errors", "feature",
})


@dataclass
class LoadedGraph:
    build_id: str
    names: dict[int, tuple[str, str]]                         # node id -> (name, type)
    keys: dict[str, set[int]] = field(default_factory=dict)   # normalised name -> node ids
    alias_keys: dict[str, set[int]] = field(default_factory=dict)  # normalised alias -> node ids
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
    # a test build (--limit or --rules) is also "fresh": refuse it, so results never come from a half-built graph
    if status.get("chunks_missing", "0") != "0" or status.get("rules") != ",".join(graph_resolve.RULES):
        raise RetrieverUnavailable(
            "Graph is partial (built with --limit) or built without entity resolution (--rules). "
            f"Rebuild it: python -m app.graph build --corpus {corpus}"
        )

    key = (corpus, chunk_size, overlap, status["fingerprint"], status.get("build_id", ""))
    if key in _CACHE:
        return _CACHE[key]

    con = graph_store.connect(corpus)
    try:
        g = LoadedGraph(status.get("build_id", ""),
                        {i: (n, t) for i, n, t in con.execute("SELECT id, name, type FROM nodes")})
        keys: dict[str, set[int]] = defaultdict(set)
        alias_keys: dict[str, set[int]] = defaultdict(set)
        for i, (name, _) in g.names.items():
            keys[graph_resolve.normalise_name(name)].add(i)
        for i, alias in con.execute("SELECT node_id, alias FROM aliases"):
            alias_keys[graph_resolve.normalise_name(alias)].add(i)
        g.keys, g.alias_keys = dict(keys), dict(alias_keys)
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


def question_words(question: str) -> list[str]:
    """Folded, lowercase words of a question, punctuation removed (hyphens inside words kept:
    kst-5031, sign-on). Unlike normalise_name, no article or suffix is stripped here: that
    happens per n-gram, so "team" at the end of a sentence is not lost."""
    text = graph_resolve._PUNCT.sub(" ", graph_resolve.fold(question))
    return [w.strip("-") for w in text.split() if w.strip("-")]


def match_entities(g: LoadedGraph, question: str) -> list[dict]:
    """Nodes named in the question, without an LLM.

    The question is normalised with entity resolution's own function, then cut into
    n-grams of 4 down to 1 words; each n-gram, normalised again ("the Messaging team"
    -> "messaging"), is looked up among node names and aliases. Longer matches win and
    a word belongs to one match only. Identifiers match exactly because they are
    compared as whole strings (kst-5031 is not kst-5038). Stop words and the generic
    words above never match on their own."""
    words = question_words(question)
    used = [False] * len(words)
    found: list[dict] = []
    seen: set[int] = set()

    for n in range(min(MAX_NGRAM, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            if any(used[i:i + n]):
                continue
            gram = words[i:i + n]
            if gram[0] in STOPWORDS or gram[-1] in STOPWORDS:  # "the messaging": the shorter n-gram will match
                continue
            if all(w in GENERIC_WORDS for w in gram):
                continue
            key = graph_resolve.normalise_name(" ".join(gram))
            if not key or key in STOPWORDS or key in GENERIC_WORDS:
                continue
            # a node's own name first; an alias only when no node has that name ("mailman" is the
            # mailman node, even if the LLM also listed mailman as an alias of postroom)
            nodes = g.keys.get(key) or g.alias_keys.get(key)
            if not nodes:
                continue
            for node in sorted(nodes):
                if node not in seen:
                    seen.add(node)
                    name, typ = g.names[node]
                    found.append({"id": node, "name": name, "type": typ, "matched_text": " ".join(gram)})
            used[i:i + n] = [True] * n

    return found


@dataclass
class Walk:
    reached: dict[int, int]     # node -> hop at which it was first reached (start entities: 0)
    followed: dict[int, int]    # edge -> hop at which it was followed
    capped: list[int]           # intermediate nodes not expanded because of HUB_CAP


def walk(g: LoadedGraph, starts: list[int], max_hops: int = MAX_HOPS, hub_cap: int = HUB_CAP) -> Walk:
    """Breadth-first from the start entities, edges in both directions, up to max_hops.
    Start entities are always expanded; a node reached on the way with more than
    hub_cap edges is kept but not expanded, so "Kestrel" or "workspace" cannot pull
    in half the corpus."""
    reached = {node: 0 for node in starts}
    followed: dict[int, int] = {}
    capped: list[int] = []
    frontier = sorted(set(starts))

    for hop in range(1, max_hops + 1):
        nxt = []
        for node in frontier:
            if reached[node] > 0 and g.degree(node) > hub_cap:
                capped.append(node)
                continue
            for other, eid in g.neighbours.get(node, []):
                followed.setdefault(eid, hop)
                if other not in reached:
                    reached[other] = hop
                    nxt.append(other)
        frontier = sorted(nxt)

    return Walk(reached, followed, sorted(set(capped)))


def search_graph(g: LoadedGraph, question: str, k: int, valid_ids: set[str] | None = None) -> dict:
    """Question -> ranked chunks + the path behind them.

    Candidates: the evidence chunks of every node reached (start entities at hop 0) and of
    every edge followed, each at its best hop. Score = 1 / (1 + best hop); ties: more
    followed edges supported by the chunk first, then chunk id, so the ranking is
    deterministic. Only chunks of the current chunking (valid_ids) are returned.
    No entity in the question: empty hits and an empty path, never an error."""
    entities = match_entities(g, question)
    w = walk(g, [e["id"] for e in entities])

    best: dict[str, int] = {}
    support: dict[str, int] = defaultdict(int)
    for node, hop in w.reached.items():
        for chunk in g.node_chunks.get(node, []):
            best[chunk] = min(best.get(chunk, hop), hop)
    for eid, hop in w.followed.items():
        for chunk in g.edge_chunks.get(eid, []):
            best[chunk] = min(best.get(chunk, hop), hop)
            support[chunk] += 1

    ranked = sorted(best, key=lambda c: (best[c], -support[c], c))
    if valid_ids is not None:
        ranked = [c for c in ranked if c in valid_ids]
    hits = [{"id": c, "score": 1.0 / (1 + best[c])} for c in ranked[:k]]

    edges = []
    for eid, hop in sorted(w.followed.items(), key=lambda x: (x[1], x[0])):
        src, rel, dst = g.edges[eid]
        edges.append({"source": g.names[src][0], "relation": rel, "target": g.names[dst][0],
                      "hop": hop, "chunk_ids": g.edge_chunks.get(eid, [])})
    path = {
        "entities": entities,
        "edges": edges,
        "chunks": [h["id"] for h in hits],
        "capped": [g.names[n][0] for n in w.capped],
    }
    return {"hits": hits, "path": path}
