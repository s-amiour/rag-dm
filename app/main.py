from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables before importing core.
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from . import evaluate as cli_eval  # the CLI's run-and-save path, shared with /api/evaluate (v3T08)
from . import bm25 as bm25_module
from . import graph_search
from . import retrieval as retrieval_module
from . import traces
from .core import (
    DATA,
    DEFAULT_CANDIDATES,
    DEFAULT_CHUNKING,
    DEFAULT_RRF_K,
    DEFAULT_TOP_K,
    DENSE_MODEL_NAME,
    records_fingerprint,
    resolve_chunking,
    INVALID_ANSWER_MESSAGE,
    build_chunks,
    config_fingerprint,
    corpus_status,
    read_corpus,
    load_questions,
    generate_answer,
    generate_answer_with_usage,
    llm_cost,
    llm_request_options,
    pipeline_config,
    words_for_corpus,
    question_words,
    safe_name,
    save_json
)

from .bm25 import get_bm25
from .retrieval import (
    RETRIEVERS,
    RetrieverUnavailable,
    default_retrievers,
    graph_available,
    retrieve
)

app = FastAPI(title="RAG V3", version="3.0.0")

app.mount(
    "/static",
    StaticFiles(directory=ROOT / "app" / "static"),
    name="static"
)


RetrieverName = Literal["dense", "bm25", "graph"]


class AskRequest(BaseModel):
    corpus: Literal["kestrel", "own"] = "own"
    question: str = Field(min_length=2, max_length=4000)
    # None = the corpus default (core.DEFAULT_CHUNKING), the chunking that was evaluated
    chunk_size: int | None = Field(default=None, ge=100, le=10000)
    overlap: int | None = Field(default=None, ge=0, le=89)
    top_k: int = Field(default=5, ge=1, le=20)
    # Which retrievers are switched on. Several on => RRF fusion.
    retrievers: list[RetrieverName] = Field(default_factory=default_retrievers)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    candidates: int = Field(default=10, ge=1, le=200)
    # False (default): dense refuses the TF-IDF fallback, as the CLI does (503 if MiniLM cannot load).
    allow_tfidf_fallback: bool = False


class EvalRequest(BaseModel):
    corpus: Literal["kestrel", "own"] = "own"
    # None = the same defaults as the CLI (core.DEFAULT_*), so a UI run matches the reference run
    chunk_size: int | None = Field(default=None, ge=100, le=10000)
    overlap: int | None = Field(default=None, ge=0, le=89)
    top_k: int | None = Field(default=None, ge=1, le=20)
    generate_answers: bool = False
    retrievers: list[RetrieverName] = Field(default_factory=default_retrievers)
    rrf_k: int | None = Field(default=None, ge=1, le=1000)
    candidates: int | None = Field(default=None, ge=1, le=200)
    weights: dict[str, float] | None = None
    allow_tfidf_fallback: bool = False  # same meaning as the CLI's --allow-tfidf-fallback
    # True: fail instead of silently measuring TF-IDF when MiniLM cannot load.
    strict_dense: bool = False


@app.get("/", response_class=HTMLResponse)
def home():
    return (
        ROOT / "app" / "static" / "index.html"
    ).read_text(encoding="utf-8")


@app.get("/api/status")
def status():
    return {
        # each corpus at the chunking that is evaluated (core.DEFAULT_CHUNKING), v3T18
        "corpora": {corpus: corpus_status(corpus) for corpus in DEFAULT_CHUNKING},
        "llm_configured": bool(
            os.getenv("LLM_API_KEY")
            and os.getenv("LLM_BASE_URL")
            and os.getenv("LLM_MODEL")
        )
    }


@app.get("/api/documents")
def documents(corpus: Literal["kestrel", "own"] = "own"):
    folder = DATA / "corpora" / corpus

    all_files = sorted([
        *folder.glob("*.txt"),
        *folder.glob("*.md")
    ]) if folder.exists() else []

    selection_file = folder / "selection.json"

    selected = (
        set(json.loads(selection_file.read_text(encoding="utf-8")))
        if selection_file.exists()
        else {p.name for p in all_files}
    )

    return [
        {
            "name": p.name,
            "characters": len(p.read_text(encoding="utf-8")),
            "selected": p.name in selected
        }
        for p in all_files
    ]


def _update_selection_file(folder: Path, add: str | None = None, remove: str | None = None):
    """Keep selection.json in step with uploads and deletes.

    No selection.json means "every document is selected", so there is
    nothing to update. If it exists, a new upload is added to it (otherwise
    the upload would be silently left out of retrieval) and a deleted
    document is removed from it.
    """
    selection_file = folder / "selection.json"

    if not selection_file.exists():
        return

    try:
        selected = set(json.loads(selection_file.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return  # unreadable: leave it, read_corpus ignores it too

    if add:
        selected.add(add)
    if remove:
        selected.discard(remove)

    save_json(selection_file, sorted(selected))


class SelectionRequest(BaseModel):
    selected: list[str]


@app.put("/api/documents/{corpus}/selection")
def update_selection(
    corpus: Literal["kestrel", "own"],
    req: SelectionRequest
):
    folder = DATA / "corpora" / corpus

    allowed = {
        p.name for p in [
            *folder.glob("*.txt"),
            *folder.glob("*.md")
        ]
    } if folder.exists() else set()

    if any(name not in allowed for name in req.selected):
        raise HTTPException(
            400,
            "Selection contains an unknown document"
        )

    save_json(folder / "selection.json", sorted(set(req.selected)))

    return {
        "selected": sorted(set(req.selected)),
        "available": len(allowed)
    }


@app.post("/api/documents/{corpus}")
async def upload_document(
    corpus: Literal["kestrel", "own"],
    file: UploadFile = File(...)
):
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise HTTPException(
            415,
            "Only UTF-8 .txt and .md documents are supported."
        )

    folder = DATA / "corpora" / corpus
    folder.mkdir(parents=True, exist_ok=True)

    name = safe_name(Path(file.filename or "").stem) + suffix

    content = await file.read()

    if len(content) > 20_000_000:
        raise HTTPException(413, "File exceeds 20 MB.")

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            400,
            "Document must be UTF-8 encoded."
        )

    (folder / name).write_text(text, encoding="utf-8")
    _update_selection_file(folder, add=name)

    return {
        "uploaded": name,
        "characters": len(text)
    }


@app.delete("/api/documents/{corpus}/{name}")
def delete_document(
    corpus: Literal["kestrel", "own"],
    name: str
):
    original = Path(name)
    suffix = original.suffix.lower()

    if suffix not in {".txt", ".md"}:
        raise HTTPException(400, "Unsupported document type")

    path = (
        DATA
        / "corpora"
        / corpus
        / (safe_name(original.stem) + suffix)
    )

    if not path.exists():
        raise HTTPException(404, "Document not found")

    path.unlink()
    _update_selection_file(path.parent, remove=path.name)

    return {"deleted": path.name}


@app.get("/api/config")
def config():
    return {
        "retrievers": list(RETRIEVERS),
        "available": {
            "dense": importlib.util.find_spec("sentence_transformers") is not None,
            "bm25": True,
            "graph": graph_available()
        },
        "default": default_retrievers(),
        "defaults": {
            "chunking": {c: {"chunk_size": s, "overlap": o} for c, (s, o) in DEFAULT_CHUNKING.items()},
            "top_k": DEFAULT_TOP_K,
            "candidates": DEFAULT_CANDIDATES,
            "rrf_k": DEFAULT_RRF_K
        },
        "graph_corpora": ["kestrel"]
    }


@app.post("/api/index/{corpus}/rebuild")
def rebuild_index(
    corpus: Literal["kestrel", "own"],
    chunk_size: int | None = None,
    overlap: int | None = None
):
    """Force a rebuild of the dense embeddings and the BM25 index (both are normally rebuilt
    automatically when the documents, their selection, or the chunking change). No chunking given:
    the corpus default (core.DEFAULT_CHUNKING), as in /api/ask. One `index` trace (v4T03)."""
    size, overlap = resolve_chunking(corpus, chunk_size, overlap)

    with traces.trace("index", kind="index", corpus=corpus,
                      chunk_size=size, overlap=overlap, trigger="rebuild") as trace:
        with trace.span("load+chunk") as span:
            records = build_chunks(corpus, size, overlap)
            span.set(documents=len({r["document"] for r in records}), chunks=len(records),
                     corpus_hash=records_fingerprint(records))

        if not records:
            raise HTTPException(400, f"No documents found for {corpus}. Upload the corpus documents first.")

        # the fingerprint of the default pipeline over these chunks, so an index trace can be matched
        # with the questions and evaluation runs that used it
        trace.set(fingerprint=config_fingerprint(pipeline_config(
            corpus, records, chunk_size=size, overlap=overlap, top_k=DEFAULT_TOP_K,
            candidates=DEFAULT_CANDIDATES, retrievers=default_retrievers(), rrf_k=DEFAULT_RRF_K
        )))

        try:
            with trace.span("embed") as span:
                dense = retrieval_module.get_dense(corpus, size, overlap, records=records, strict=True, force=True)
                dense_info = {
                    "backend": dense.backend,
                    "model": DENSE_MODEL_NAME,
                    "chunks": len(records),
                    "cache_hit": False,
                    "build_seconds": round(float(dense.build_seconds), 3)
                }
                span.set(**dense_info)
        except RetrieverUnavailable as e:
            raise HTTPException(503, str(e))

        with trace.span("bm25.build") as span:
            retriever = get_bm25(corpus, size, overlap, records=records, force=True)
            span.set(
                chunks=retriever.info.get("chunks"),
                vocabulary=retriever.info.get("terms"),
                stopwords=bm25_module.stopwords_enabled(),
                tokenizer_version=bm25_module.TOKENIZER_VERSION,
                source=retriever.info.get("source"),
                build_seconds=round(float(retriever.info.get("seconds") or 0.0), 4)
            )

        trace.set_outcome("ok")

        # today's response (the BM25 info) is kept at the top level; the rest is added
        return {
            **retriever.info,
            "trace_id": trace.id,
            "chunks": len(records),
            "dense": dense_info,
            "bm25": retriever.info
        }


def question_outcome(valid: bool, cited_ids: list[str]) -> str:
    """Outcome of an answered request (v4T02): invalid reply, abstention (a valid reply that cites
    nothing: the prompt asks the model to say the answer is not in the documents), or answered.
    "error" is set by the trace itself when an exception ends the request."""
    if not valid:
        return "invalid"
    return "answered" if cited_ids else "abstained"


@app.post("/api/ask")
def ask(req: AskRequest):
    chunk_size, overlap = resolve_chunking(req.corpus, req.chunk_size, req.overlap)

    # One trace per question (v4T02). An exception closes it with outcome "error" before the
    # HTTP error reaches the user. T09: pass the redacted question here.
    with traces.trace(
        "request",
        kind="question",
        corpus=req.corpus,
        question=req.question,
        retrievers=list(dict.fromkeys(req.retrievers)),
        top_k=req.top_k,
        candidates=req.candidates,
        rrf_k=req.rrf_k,
        chunk_size=chunk_size,
        overlap=overlap
    ) as trace:
        records = build_chunks(
            req.corpus,
            chunk_size,
            overlap
        )

        if not records:
            raise HTTPException(
                400,
                f"No documents found for {req.corpus}. Upload the corpus documents first."
            )

        trace.set(fingerprint=config_fingerprint(pipeline_config(
            req.corpus, records, chunk_size=chunk_size, overlap=overlap, top_k=req.top_k,
            candidates=req.candidates, retrievers=req.retrievers, rrf_k=req.rrf_k
        )))

        try:
            out = retrieve(
                req.corpus,
                req.question,
                chunk_size,
                overlap,
                req.retrievers,
                depth=req.top_k,
                candidates=req.candidates,
                rrf_k=req.rrf_k,
                strict_dense=not req.allow_tfidf_fallback,
                records=records,
                trace=trace
            )
        except ValueError as e:
            raise HTTPException(400, str(e))
        except RetrieverUnavailable as e:
            raise HTTPException(503, str(e))

        hits = out["hits"]

        # Errors here are about reaching the LLM (missing config, network,
        # timeout). A reply that arrives but is malformed is not an error:
        # parse_llm_answer flags it with valid=False.
        try:
            with trace.span("generate") as span:
                span.set(**{"gen_ai.request.model": os.getenv("LLM_MODEL")})
                raw, gen_usage = generate_answer_with_usage(req.question, hits)
                cost_usd, price_known = llm_cost(gen_usage["prompt_tokens"], gen_usage["completion_tokens"])
                span.set(**{
                    "gen_ai.request.model": gen_usage.get("request_model") or os.getenv("LLM_MODEL"),
                    "gen_ai.response.model": gen_usage.get("response_model"),
                    "gen_ai.usage.input_tokens": gen_usage["prompt_tokens"],
                    "gen_ai.usage.output_tokens": gen_usage["completion_tokens"],
                    "gen_ai.response.finish_reasons": [gen_usage.get("finish_reason")],
                    "thinking": gen_usage.get("thinking"),
                    "cost_usd": cost_usd,
                    "price_known": price_known
                })
        except Exception as e:
            raise HTTPException(503, str(e))

        with trace.span("validate") as span:
            answer = json.loads(raw)
            valid = answer["valid"]
            cited_ids = [hit["id"] for hit in hits if hit["id"] in answer["citations"]]
            span.set(
                valid=valid,
                validation_error=answer.get("error"),
                cited_ids=cited_ids,
                dropped_citations=answer.get("dropped_citations", 0)
            )

        trace.set_outcome(question_outcome(valid, cited_ids))

        usage = {
            "llm_calls": out["usage"]["llm_calls"] + gen_usage["llm_calls"],
            "prompt_tokens": (
                out["usage"]["prompt_tokens"] + gen_usage["prompt_tokens"]
            ),
            "completion_tokens": (
                out["usage"]["completion_tokens"] + gen_usage["completion_tokens"]
            ),
            "generation_latency_s": gen_usage["latency_s"],
            "finish_reason": gen_usage.get("finish_reason"),
            "thinking": gen_usage.get("thinking")
        }

        return {
            "answer": answer["answer"] if valid else answer.get("user_message", INVALID_ANSWER_MESSAGE),
            "valid": valid,
            "validation_error": answer.get("error"),
            "citations": [
                {**hit, "citation": hit["id"]}
                for hit in hits
                if hit["id"] in answer["citations"]
            ],
            "retrieved": hits,
            "retrievers": out["retrievers"],
            "backend": ", ".join(
                f"{name}={detail if isinstance(detail, str) else detail.get('index')}"
                for name, detail in out["backend"].items()
            ),
            "dense_backend": out["backend"].get("dense"),  # "sentence-transformers", "tfidf", or None if dense is off
            "graph_path": out["graph_path"],
            "timings": out["timings"],
            "usage": usage,
            "trace_id": trace.id  # opens this answer's trace (Details panel, v4T04)
        }


CONCEPT_NOT_FOUND = (
    "No concept called “{name}” in the Kestrel concept graph. Try an error code (KST-5031), "
    "a service (postroom) or a team (Messaging)."
)
GRAPH_NOT_READY = "The concept map needs the Kestrel concept graph, and it is not ready: {detail}"


@app.get("/api/graph/concept")
def graph_concept(
    name: str = Query(min_length=1, max_length=200),
    corpus: Literal["kestrel", "own"] = "kestrel"
):
    """Concept map explorer (v4T07): a concept of the Kestrel graph, its one-hop edges with their
    relations and source chunks, and the documents that mention it. No LLM call. One
    `concept.lookup` trace per lookup; the chunk texts are in the response (for the tab), never
    in the trace."""
    with traces.trace("concept.lookup", kind="concept.lookup", corpus=corpus, concept=name) as trace:
        if corpus not in retrieval_module.GRAPH_CORPORA:
            raise HTTPException(400, "The concept graph is only built for the Kestrel corpus.")

        size, overlap = resolve_chunking(corpus)

        try:
            with trace.span("graph.load") as span:
                records = build_chunks(corpus, size, overlap)
                g = graph_search.load(corpus, size, overlap, records=records)
                span.set(build_id=g.build_id, nodes=len(g.names), edges=len(g.edges))
        except RetrieverUnavailable as e:
            raise HTTPException(503, GRAPH_NOT_READY.format(detail=e))

        with trace.span("concept.match") as span:
            chunk_docs = {r["id"]: r["document"] for r in records}
            out = graph_search.neighborhood(g, name, chunk_docs=chunk_docs)
            span.set(
                matched_by=out["matched_by"],
                concepts=[{"name": c["name"], "type": c["type"]} for c in out["concepts"]],
                edges=sum(len(c["edges"]) for c in out["concepts"]),
                more_edges=sum(c["more_edges"] for c in out["concepts"]),
                chunk_ids=sorted({ch for c in out["concepts"] for ch in c["chunks"]})
            )

        trace.set_outcome("found" if out["found"] else "not_found")
        texts = {r["id"]: r["text"] for r in records}
        wanted = {ch for c in out["concepts"] for ch in c["chunks"]} | {
            ch for c in out["concepts"] for e in c["edges"] for ch in e["chunks"]}

        return {
            **out,
            "message": None if out["found"] else CONCEPT_NOT_FOUND.format(name=name),
            "chunk_texts": {ch: texts[ch] for ch in sorted(wanted) if ch in texts},
            "trace_id": trace.id
        }


@app.get("/api/words/{corpus}")
def words(
    corpus: Literal["kestrel", "own"],
    chunk_size: int = 800,
    overlap: int = 10,
    limit: int = 300
):
    result = words_for_corpus(corpus, chunk_size, overlap)

    result["vocabulary"] = result["vocabulary"][
        :max(1, min(limit, 2000))
    ]

    return result


@app.get("/api/question-words/{corpus}")
def qwords(
    corpus: Literal["kestrel", "own"],
    question: str,
    chunk_size: int = 800,
    overlap: int = 10
):
    return question_words(
        corpus,
        question,
        chunk_size,
        overlap
    )


@app.get("/api/questions/{corpus}")
def questions(corpus: Literal["kestrel", "own"]):
    return load_questions(corpus)


@app.post("/api/evaluate")
def evaluate(req: EvalRequest):
    # Same path as `python -m app.evaluate run`: same defaults, run name, file and summary (v3T08).
    try:
        name, result, path = cli_eval.run_and_save(
            req.corpus,
            req.retrievers,
            chunk_size=req.chunk_size,
            overlap=req.overlap,
            top_k=req.top_k,
            rrf_k=req.rrf_k,
            candidates=req.candidates,
            weights=req.weights,
            generate_answers=req.generate_answers,
            strict_dense=not req.allow_tfidf_fallback,
            out_dir=cli_eval.DEFAULT_OUT
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except (RetrieverUnavailable, RuntimeError) as e:
        raise HTTPException(503, str(e))

    return {
        **result,
        "run_name": name,
        "saved_to": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    }


@app.post("/api/no-context")
def no_context(req: EvalRequest):
    """Run the question set against the configured LLM without retrieved context."""

    questions = load_questions(req.corpus)

    if not questions:
        raise HTTPException(
            400,
            f"No question set found at data/questions/{req.corpus}.jsonl"
        )

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model = os.getenv("LLM_MODEL")

    if not api_key or not base_url or not model:
        raise HTTPException(
            503,
            "LLM configuration is incomplete. "
            "Check LLM_BASE_URL, LLM_API_KEY and LLM_MODEL in .env."
        )

    from openai import OpenAI

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    rows = []

    for q in questions:
        prompt = (
            "Answer using your own knowledge only, without documents or tools. "
            "If unsure, say so.\n\nQUESTION: "
            + q["question"]
        )

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **llm_request_options()
            )

            rows.append({
                "id": q.get("id"),
                "question": q["question"],
                "type": q.get("type"),
                "answer": response.choices[0].message.content or "",
                "finish_reason": response.choices[0].finish_reason
            })

        except Exception as e:
            rows.append({
                "id": q.get("id"),
                "question": q["question"],
                "error": str(e)
            })

    path = DATA / "results" / (
        f"{req.corpus}-no-context-"
        f"{time.strftime('%Y%m%d-%H%M%S')}-"
        f"{time.time_ns() % 1_000_000_000:09d}.json"
    )

    save_json(path, {
        "corpus": req.corpus,
        "mode": "no-context",
        "provider_base_url": base_url,
        "model": model,
        "results": rows
    })

    return {
        "results": rows,
        "saved_to": str(path.relative_to(ROOT))
    }


@app.get("/api/results")
def results():
    folder = DATA / "results"

    return [
        {
            "file": p.name,
            "path": str(p.relative_to(ROOT))
        }
        for p in sorted(folder.rglob("*.json"))
    ] if folder.exists() else []