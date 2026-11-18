# RAG V3: dense + BM25 + concept graph

A FastAPI application for retrieval-grounded Q&A over two local corpora (**Kestrel KB** and our own **Vellora Duo** manual).
V3 adds **BM25** (both corpora), a **concept graph** (Kestrel) and **RRF fusion**; every retriever can be switched on or off
from the UI, the API and the evaluation command. Answer generation uses any OpenAI-compatible API (configured in `.env`).
Retrieval and evaluation need no API key.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env                 # then fill in LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
python -m uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000. The first dense query downloads `all-MiniLM-L6-v2` (needs network once).

| `.env` variable | Meaning |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | answer generation, no-context baseline, graph extraction |
| `DEFAULT_RETRIEVERS` | retrievers ticked by default, e.g. `dense,bm25` (our chosen default, see REPORT.md) |
| `BM25_STOPWORDS` | `1` (default) or `0`: English stop-word list in BM25 tokenization |
| `GRAPH_LLM_MODEL` | optional: model used for graph extraction only (default: `LLM_MODEL`) |
| `GRAPH_CALL_DELAY_S` | optional: seconds between extraction calls, for free-tier rate limits (default `1.0`) |
| `GRAPH_DISABLE_THINKING` | optional: `1` asks the server to switch a reasoning model's hidden thinking off during the graph build (DECISIONS D-24) |
| `LLM_DISABLE_THINKING` | optional: `1` switches a reasoning model's hidden thinking off for answers; without it, nemotron often uses all `LLM_MAX_TOKENS` before answering (DECISIONS D-35) |
| `LLM_MAX_TOKENS` | optional: completion token limit for answers (default `2048`) |

## Switching retrievers on and off

| Where | How |
|---|---|
| **UI** | *Ask* and *Evaluation* tabs: tick `dense`, `BM25`, `concept graph`. Graph is disabled on the own corpus (Kestrel only). |
| **API** | `POST /api/ask` and `POST /api/evaluate` accept `"retrievers": ["dense","bm25","graph"]` (any subset), plus `rrf_k` (default 60) and `candidates` (default 10: how many results each retriever gives to the fusion, see DECISIONS D-18). `GET /api/config` lists what is available. |
| **Command line** | `--retrievers dense,bm25,graph` (see below) |

One retriever on: its own ranking. Several on: **RRF** over their lists, `score = Σ weight / (k + rank)`.
Every result shows which retriever(s) found it and at which rank (`dense #3 · bm25 #1`). With the graph on, the UI also shows
the path: entities found in the question, edges followed, and the chunks they came from.

## BM25

- Own implementation in `app/bm25.py` (inverted index, k1 = 1.5, b = 0.75).
- Persisted in `data/index/bm25-<corpus>-<size>-<overlap>-<sw0|sw1>.json` (git-ignored). A fingerprint of the chunk texts and
  the tokenizer settings is stored inside; **uploading, deleting, selecting documents, or changing chunk size/overlap rebuilds it
  automatically** on the next query. Force a rebuild: `POST /api/index/{corpus}/rebuild?chunk_size=800&overlap=10`.
- Tokenization: lowercase, accents folded, `KST-5031` indexed as `kst-5031` + `kst` + `5031`, small English stop-word list.
  Details and measurements: `REPORT.md` and `DECISIONS.md`.
- Dense embeddings are cached the same way (`data/index/dense-*.npz`).

## Concept graph (Kestrel)

> **Status:** extraction (v3T01), entity resolution (v3T02) and the graph retriever (v3T03) are done; the UI path display is T4.
> The graph has to be built once (below). If it is missing, or was built for other chunks, ticking the graph returns a 503 that says
> which build command to run.

### Graph retrieval

The graph retriever (`app/graph_retriever.py`, DECISIONS D-29) finds the entities named in the question (n-grams matched on node
names and aliases, error codes exact, generic words such as "team" or "support" ignored), walks 2 hops in both directions, and ranks
the evidence chunks of what it reached by hop: 1 / (1 + hop). No LLM call at query time. It is switched on like the other retrievers
(`"retrievers": ["dense","bm25","graph"]`, `--retrievers dense,bm25,graph`, the UI checkbox) and fused with RRF. To see what it does
for one question:

```bash
python -m app.graph search "Error KST-5031 happened on the last two scheduled reports. Which team should support contact?"
```

### Build the graph

```bash
python -m app.graph build --corpus kestrel --limit 3   # first 3 chunks only: cheap check (again: 0 calls)
python -m app.graph build --corpus kestrel             # every chunk (43 at 800/10), needs LLM_* in .env
python -m app.graph status                             # missing / stale / fresh
python -m app.graph check                              # integrity: every node and edge has evidence
python -m app.graph walk KST-5031 --hops 3             # follow the edges from an entity, with evidence chunks
```

The extraction cache lives in `data/index/` (git-ignored), so a fresh checkout pays for extraction once: about 82 calls and 5 minutes.
To reproduce REPORT §7, use the same endpoint and model as DECISIONS D-24 in `.env` (`GRAPH_LLM_MODEL` overrides only the model, not
the URL or the key):

```
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_API_KEY=<your NVIDIA key>
GRAPH_LLM_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
GRAPH_DISABLE_THINKING=1
```

For each chunk (same chunking as the other retrievers, Kestrel 800/10), one LLM call returns entities and
`(source, relation, target, quote)` triples. These are validated against the fixed schema in `app/graph_schema.py` (types,
allowed source/target types, and the quote must be copied from the chunk and name one end of the relation). An invalid reply gets one retry; if the second reply
is still invalid, its valid entities and relations are kept and the invalid ones are dropped and logged (`partial`, DECISIONS D-26);
with `--strict` the whole chunk is skipped instead. A reply with nothing valid is skipped and logged. The build never stops on a bad reply. It prints chunks processed / skipped, entities, relations, LLM calls,
tokens and wall time.

- **Stored in** `data/index/graph-kestrel.sqlite` (git-ignored): tables `nodes`, `aliases`, `edges`, `evidence`, `merges`. Every node
  and edge points to `{doc, chunk_id, quote}`.
- **Rerun = free:** each chunk's extraction is cached by the hash of its text, so a rerun with unchanged documents makes **0 LLM
  calls**, and a rerun after a crash or rate limit only pays for the missing chunks. A changed document, chunking, schema or prompt
  makes the graph `stale` and re-extracts only the chunks that changed. A change of the validation rules (`VALIDATION_VERSION`)
  re-checks the cached extractions without any call. `--retry-skipped` asks again for chunks that failed twice.
- **Cost log** (committed): `data/results/graph/llm_calls.jsonl` (one row per call: model, tokens, latency, outcome),
  `graph_errors.jsonl` (skipped chunks with the raw reply) and `build-kestrel.json` (the summary). Numbers in REPORT §7 come from there.
- **Rate limits:** a 429 is retried with back-off. If a free tier runs out, set another key or `GRAPH_LLM_MODEL` and run the
  command again; finished chunks are not paid again.

### Entity resolution

After extraction, the build merges nodes that are the same entity (0 LLM calls, from the cache), with three rules applied in order:
**identifier** (error codes, IP addresses and plan names match exactly or never: KST-5031 is not KST-5038), **normalised** name
of the same type ("the Messaging team" = "Messaging", "Inès" = "Ines"), and **alias** (a name the extraction listed as an alias). It
never merges across types, nor the two ends of a `replaces` edge (postroom replaced mailman: two entities). Every merge is a row of the
`merges` table and of `data/results/graph/merges.jsonl` (method + evidence chunks); merged names stay as aliases. Merges are
reversible:

```bash
python -m app.graph build --rules none                 # the unmerged graph
python -m app.graph build --rules identifier,normalised # any subset; default: identifier,normalised,alias
```

### Grading a sample of triples

```bash
python -m app.graph sample --n 10 --seed 42     # 10 random edges of the resolved graph -> data/results/v3/graph_triples_graded.json
python -m app.graph chunk service-catalog_md:2  # read an evidence chunk in full while grading
python -m app.graph grades                      # check the grades, write graph_triples_graded.md with the counts
```

A person grades each triple **correct**, **wrong** or **unsupported** against its evidence chunk, following
`data/results/v3/graph_triples_grading_rules.md`, and fills `grade`, `graded_chunk`, `note` (and `graded_by`, `reviewed_by`) in the
JSON file; a second person reviews. The same graph and seed always give the same sample; the file records the seed and a hash of the
graph, and `grades` warns if the graph has been rebuilt since.

Design choices: `DECISIONS.md` D-20 (schema) to D-24 (model choice, hidden reasoning), D-28 (entity resolution).

Integration contract, so the graph plugs into fusion, UI and evaluation: `app/graph_retriever.py` must expose

```python
search(corpus: str, query: str, k: int, chunk_size: int, overlap: int) -> {
    "hits":  [{"id": "<chunk id from core.build_chunks>", "score": float}, ...],   # best first
    "path":  {"entities": [...], "edges": [{"source","relation","target"}, ...], "chunks": ["<chunk id>", ...]},
    "usage": {"llm_calls": int, "prompt_tokens": int, "completion_tokens": int}   # query-time cost, 0 if none
}
```

Chunk ids depend on chunk size and overlap: build the graph with the same chunking as the evaluation
(Kestrel: 800 / 10, the V2 final value).

## Evaluation

```bash
# the three required configurations (+ BM25 alone, BM25 without stop words), both corpora
python -m app.evaluate matrix

# one configuration
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25,graph --top-k 10

# V2 sanity check: dense on the original ten Kestrel questions only
python -m app.evaluate run --corpus kestrel --retrievers dense --ids q01,q02,q03,q04,q05,q06,q07,q08,q09,q10

# weighted RRF, BM25 without stop words, generated answers (LLM calls)
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25,graph --weights dense=1,bm25=1,graph=2
python -m app.evaluate run --corpus own --retrievers bm25 --no-stopwords
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --generate
```

Defaults: V2's final chunking per corpus (Kestrel 800/10, own 500/20), `top_k` 10. Dense runs in **strict mode**: if
sentence-transformers cannot load, the command stops instead of silently measuring TF-IDF (`--allow-tfidf-fallback` overrides it
and the summary then carries a warning).

Output in `data/results/v3/`: one JSON per run (every question, per-hop gold chunks, which retriever found what at which rank,
cost) and `summary.md` / `summary.json`: results table (configurations × corpora × question types), cost per question and index
build, and the automatic list of questions where adding a retriever made things worse.

The summary is **rebuilt from every run file in the folder** (`{corpus}__{name}.json`) after each run, whether it was started
with `run`, `matrix` or the Evaluate button: rerunning one configuration replaces only its own row and keeps the others. The
"Configurations in this folder" table shows when each configuration was run and with which settings, so older results stand out.
To produce a report from **one consistent run**, write it to a fresh folder, e.g.
`python -m app.evaluate matrix --required-only --out-dir data/results/v3-final` (or empty `data/results/v3/` first: run files are
never deleted automatically).

**Metrics.** A question has one *hop* per evidence quote. *Hit@k*: at least one hop has a gold chunk in the top k (V2's metric).
*Recall@k*: share of hops that have a gold chunk in the top k, so it tells whether **all** hops of a chain were retrieved.
`not_in_corpus` questions have no gold chunk and are excluded from both. Gold chunks = chunks containing the quote, same
rule as V2. Types are reported with the assignment's names (`different_vocabulary` → `vocabulary`, `two_hops` → `two_hop`);
the question files are untouched, the original name is kept as `type_raw` in the results.A hop whose quote matches no chunk (for example, a quote cut by a chunk boundary) counts as not retrieved and is listed in `unmatched_hops`.

The UI *Evaluation* tab runs the same code for one configuration.

## Question sets

`data/questions/kestrel.jsonl` (q01–q10 official; **q11, q12 added in V3**, both `two_hop` with a quote per hop) and
`data/questions/own.jsonl`. Old questions are never edited; new ones are appended. `tests/test_questions.py` enforces this.

## Tests

```bash
python -m unittest discover -s tests -t .     # or: python -m pytest -q
```

Covers the tokenizer, BM25 ranking and index persistence/invalidation, RRF, retriever orchestration, dense caching, and the
metrics (per-hop recall, type mapping, regression detection), and graph extraction (`tests/test_graph.py`: schema, every
validation failure, retry then skip, rerun with 0 calls, staleness, 429 back-off, integrity). They use a fake embedding model and a
fake LLM, so no download and no API key are needed.
The browser test (`tests/test_ui_smoke.py`) needs Playwright, which the app itself does not:
`pip install -r requirements-dev.txt`, then once `python -m playwright install chromium`.
Run it alone with `python -m pytest tests/test_ui_smoke.py -rs`; without Playwright it is skipped.

## Repository map

| Path | Content |
|---|---|
| `app/core.py` | chunking, corpora, dense retriever + caches, answer generation |
| `app/bm25.py` | BM25 index, tokenizer, persistence |
| `app/fusion.py` | RRF |
| `app/retrieval.py` | retriever switches, fusion, timings, graph contract |
| `app/graph_schema.py`, `app/graph.py` | concept graph: schema; extraction, validation, storage, build / walk commands |
| `app/graph_resolve.py` | entity resolution: normalisation, identifiers, merge rules |
| `app/evaluation.py`, `app/evaluate.py` | metrics, CLI |
| `app/main.py`, `app/static/index.html` | API and UI |
| `data/` | corpora, questions, results (`results/v3/` = V3, `results/graph/` = graph build logs), derived indexes (`index/`, git-ignored) |
| `REPORT.md`, `DECISIONS.md`, `docs/REPORT_V2.md` | V3 report, decision log, previous report |
| `bm25/`, `concepts/`, `ask_*.py` | course exercises (standalone scripts, not used by the app) |
