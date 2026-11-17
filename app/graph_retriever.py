"""Graph retriever (ticket v3T03): the search() contract of app/retrieval.py.

Walks the resolved concept graph (T1 extraction + T2 resolution) from the entities
named in the question and returns their evidence chunks, best hop first. Query time
makes no LLM call. Strategy and alternatives: DECISIONS.md D-29.
"""
from __future__ import annotations

from typing import Any

from . import core, graph_search
from .graph import GRAPH_CORPORA
from .retrieval import RetrieverUnavailable


def search(corpus: str, query: str, k: int, chunk_size: int, overlap: int) -> dict[str, Any]:
    if corpus not in GRAPH_CORPORA:
        raise RetrieverUnavailable(f"The concept graph is only built for: {', '.join(GRAPH_CORPORA)}.")
    records = core.build_chunks(corpus, chunk_size, overlap)
    g = graph_search.load(corpus, chunk_size, overlap, records=records)
    out = graph_search.search_graph(g, query, k, {r["id"] for r in records})
    return {
        "hits": out["hits"],
        "path": out["path"],
        "usage": {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
    }
