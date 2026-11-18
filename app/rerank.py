"""Cross-encoder reranking (v4T05).

Dense and BM25 score the question and a chunk separately, then compare. A cross-encoder reads
the question and the chunk together, in one pass, so it can judge whether this chunk answers
this question. It is too slow to run on every chunk, so it only reorders the short list that
fusion already found: fusion keeps CANDIDATES chunks, the cross-encoder scores each
(question, chunk) pair, and the best top-k are kept.

It runs locally (sentence-transformers): no LLM call, no tokens, cost 0.
If the model cannot load, the request fails with RerankerUnavailable. Reranking is never
skipped silently: a result that says "reranked" was reranked.
"""
from __future__ import annotations

import time
from typing import Any

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many fused chunks the cross-encoder gets to reorder. More than the 10 of plain fusion
# (D-18): a reranker can only promote a chunk it is given, so a gold chunk at fusion rank 15
# can only reach the top 5 if it is in the pool.
CANDIDATES = 30

_MODEL = None
LOAD_SECONDS = 0.0


class RerankerUnavailable(RuntimeError):
    """Reranking was asked for, but the cross-encoder cannot load."""


def pool_size(candidates: int, depth: int) -> int:
    """How many chunks each retriever returns, and fusion keeps, when reranking is on."""
    return max(int(candidates), int(depth), CANDIDATES)


def get_model():
    """Load the cross-encoder once per process (the first load downloads it, about 90 MB)."""
    global _MODEL, LOAD_SECONDS

    if _MODEL is None:
        started = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder

            _MODEL = CrossEncoder(MODEL_NAME)
        except Exception as exc:
            raise RerankerUnavailable(
                f"Reranker unavailable: the cross-encoder {MODEL_NAME} could not load ({exc}). "
                "Fix the sentence-transformers install (or the download), or switch reranking off."
            ) from exc

        LOAD_SECONDS = time.perf_counter() - started

    return _MODEL


def rerank(question: str, hits: list[dict[str, Any]], top_k: int, model=None) -> list[dict[str, Any]]:
    """Reorder `hits` (best first, as fusion ranked them) by cross-encoder score, keep the top_k.

    Each kept hit carries rank_before (its rank after fusion), rank_after (its rank now),
    rerank_score (the cross-encoder's score, now also `score`) and fused_score (the score
    fusion gave it). Ties keep the fusion order, so the result is deterministic."""
    if not hits:
        return []

    model = model if model is not None else get_model()
    scores = [float(s) for s in model.predict([(question, h["text"]) for h in hits])]
    order = sorted(range(len(hits)), key=lambda i: -scores[i])  # sorted() is stable: ties keep fusion order

    return [
        {
            **hits[i],
            "fused_score": float(hits[i].get("score", 0.0)),
            "score": scores[i],
            "rerank_score": scores[i],
            "rank_before": i + 1,
            "rank_after": new_rank,
        }
        for new_rank, i in enumerate(order[:top_k], start=1)
    ]
