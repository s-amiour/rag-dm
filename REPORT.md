# RAG V3: Team Evaluation Report

## 1. Team and release

- Team members:
  - **Sultan AMIOUR**
  - **Muhammad Saad JAVED**
  - **Jaimie MATHANGI**
- Team of 3 (GitHub: `s-amiour`, `Msaadjaved`, `Chimaera-Emerald`); Jaimie joined for V4. The "teams of 3" extra piece applies:
  weighted RRF compared with plain RRF (ticket B7).
- Repository: https://github.com/s-amiour/rag-dm · tag `v3`
- Previous report: `docs/REPORT_V2.md` · decision log: `DECISIONS.md` · how to run: `README.md`

## 2. Summary

- V3 adds **BM25** (both corpora, persistent index), **RRF fusion** with every retriever switchable from the UI, the API and the
  evaluation command, and **Recall@k per hop** next to V2's HitRate@k.
- **Concept graph: extraction, entity resolution, graph retrieval and the path in the UI done (tickets v3T01 to v3T04; sections 6
  and 7).** Graph alone puts both hop chunks of q08 and q11 in its top 10 (section 6). The third required configuration
  (`dense+bm25+graph`) runs. **Adding the graph to dense+bm25 on Kestrel: Hit@1 0.50 → 0.70, Hit@3 0.90 → 1.00, Recall@5 0.79 →
  0.84, `two_hop` Recall@5 0.56 → 0.72, no question worse at its first gold chunk, no LLM call at query time.** It moves found
  hops up; it adds no hop to the top 10 (section 8).
- **BM25 helps, but only once fusion is configured carefully.** On Kestrel, plain RRF with 30 candidates per retriever left Hit@k unchanged
  (0.50 / 0.80 / 0.90 / 1.00 at k = 1, 3, 5, 10) and lowered Recall@3 from 0.72 to 0.54. Limiting each retriever to its 10 best candidates
  (D-18) gives Hit@3 **0.90**, Hit@5 **1.00** and Recall@5 **0.79**, the best of all configurations at k = 3 and 5. It fixes
  `exact_identifier` (Hit@5 0.50 → 1.00). It costs a little on rank 1 (Hit@1 stays 0.50, q05 and q03 drop from rank 1 to 2) and on
  `split_answer` Recall@3 (1.00 → 0.38).
- On our own corpus (13 answerable questions since v3T13): Hit@1 0.62 → 0.69, no loss at k ≥ 3.
- BM25 needs no LLM call, builds in 0.03 to 0.04 s and adds no measurable query latency (15 to 36 ms per query in every configuration
  and run; the differences are noise).
- **Limit of this work:** `two_hop` Recall@10 stays at 0.72 in every configuration, graph included. The missing hop is the same chunk
  for q08 and q12 (the team table): the graph finds it, but only at rank 4 to 5 of its own list, and fusion pushes it out of the top 10
  (section 8). It is also reached through a wrong edge (known failure 17).
- **Caveat:** Kestrel has 10 answerable questions, so one question is worth 0.10; most types have n = 1 to 5. Differences are indicative.
  The `candidates` = 10 setting was chosen after looking at q05 and is scored on the same questions (no held-out set).

## 3. Results

Chunking per corpus: Kestrel 800/10, own 500/20 (V2 finals). Dense backend in every run: `sentence-transformers` (all-MiniLM-L6-v2),
embeddings cached on disk. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. `not_in_corpus` questions are excluded
from both metrics. Kestrel now has 12 questions (q11 and q12 added; q09 and q10 are `not_in_corpus`), so n = 10 answerable (8 in V2).

**V2 check:** `--ids q01..q10` with `dense` reproduces V2 exactly (Hit@1/3/5/10 = 0.62 / 0.88 / 0.88 / 1.00, same per type), so the `dense`
row is V2's baseline re-measured on the extended question set and nothing else changed.

`candidates` = how many results each retriever gives to the fusion. 30 in the first runs (`data/results/v3-candidates30/`,
`data/results/v3-weighted/`), **10 by default since D-18** (`data/results/v3/`, which this table reports).

### Kestrel (800/10)

| Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense (V2 baseline) | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.33 | 0.72 | 0.77 | 0.92 |
| dense | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| dense | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| dense | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | versions | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| dense | two_hop | 3 | 0.33 | 0.67 | 1.00 | 1.00 | 0.17 | 0.39 | 0.56 | 0.72 |
| **dense+bm25 (default)** | **ALL** | 10 | 0.50 | **0.90** | **1.00** | 1.00 | 0.32 | 0.64 | **0.79** | 0.92 |
| dense+bm25 | exact_identifier | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| dense+bm25 | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | two_hop | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.39 | 0.56 | 0.56 | 0.72 |
| *ablation:* dense+bm25, 30 candidates | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.32 | 0.54 | 0.69 | 0.92 |
| *ablation:* dense+bm25, 30 candidates | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| *ablation:* dense+bm25, 30 candidates, weights dense=2 bm25=1 | **ALL** | 10 | 0.60 | 0.70 | 0.90 | 1.00 | 0.34 | 0.54 | 0.77 | 0.92 |
| *ablation:* same, weighted | vocabulary | 1 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| *ablation:* same, weighted | two_hop | 3 | 0.67 | 0.67 | 1.00 | 1.00 | 0.39 | 0.39 | 0.56 | 0.72 |
| **dense+bm25+graph** | **ALL** | 10 | **0.70** | **1.00** | 1.00 | 1.00 | **0.47** | **0.79** | **0.84** | 0.92 |
| dense+bm25+graph | exact_identifier | 2 | **1.00** | **1.00** | 1.00 | 1.00 | **0.75** | **1.00** | 1.00 | 1.00 |
| dense+bm25+graph | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| dense+bm25+graph | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| dense+bm25+graph | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25+graph | two_hop | 3 | **1.00** | 1.00 | 1.00 | 1.00 | **0.56** | **0.72** | **0.72** | 0.72 |

Kestrel only: the graph is built for Kestrel (section 6). Rerun of 18 November 2026 (`data/results/v3/`): the dense and dense+bm25
rows came out identical to the earlier run.

### Own corpus (500/20)

Question set after v3T13 (D-34): 15 questions, 13 answerable (OWN-13 and OWN-14 are `not_in_corpus`). Rerun 18 November 2026; the
earlier 12-question numbers are in git history. The check that 10 and 30 candidates give identical results was made on the earlier set.

| Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | **ALL** | 13 | 0.62 | 0.85 | 0.92 | 0.92 | 0.62 | 0.85 | 0.92 | 0.92 |
| dense | exact_identifier | 5 | 0.60 | 0.80 | 1.00 | 1.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| dense | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dense | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **dense+bm25 (default)** | **ALL** | 13 | **0.69** | 0.85 | 0.92 | 0.92 | **0.69** | 0.85 | 0.92 | 0.92 |
| dense+bm25 | exact_identifier | 5 | **0.80** | 0.80 | 1.00 | 1.00 | **0.80** | 0.80 | 1.00 | 1.00 |
| dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| dense+bm25 | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Reading the tables:

- **`exact_identifier`:** BM25 helps on both corpora (Kestrel Hit@5 0.50 → 1.00; own Hit@1 0.60 → 0.80).
- **`vocabulary`:** BM25 cannot help (no shared word) and can hurt through fusion (q05, section 9).
- **`versions`:** Hit@1 0.00 → 0.50 with BM25; Hit@3 is already 1.00 with dense.
- **`two_hop`:** Hit@10 is 1.00 but Recall@10 only 0.72: every chain is "found" but not every hop. In the V2 check, q08 alone has
  Recall@10 = 0.50 with dense: one of its two hops is not in the top 10. The graph raises Recall@3 and @5 to 0.72 (q11's second hop,
  rank 6 → 2) but not Recall@10 (section 8).
- **Graph:** it changes only the 5 questions that name an entity of the graph; types without one (`vocabulary`, `versions`, most of
  `split_answer`) are identical to dense+bm25. `exact_identifier` Hit@1 0.50 → 1.00: q01's chunk goes rank 4 → 1.
- **`not_in_corpus`:** excluded from the metrics: q09 and q10 on Kestrel, OWN-13 and OWN-14 on the own corpus (D-34). OWN-04
  (vinegar, answerable) and OWN-05 (one evidence quote) are now typed `other`. The own corpus has no `versions` question.
- **Own `two_hop` (OWN-15, n = 1):** neither hop is in the top 10 with dense or with dense+bm25, so the own corpus's Hit@10 is 0.92,
  not 1.00. There is no graph for the own corpus to try (section 6, optional features).

### Cost per question (no answer generation)

| Corpus | Configuration | Retrieval ms | LLM calls | Index build |
|---|---|---:|---:|---|
| Kestrel | dense | 35.8 | 0 | dense 22.0 s (model load; embeddings cached on disk after the first run) |
| Kestrel | dense+bm25 | 27.6 | 0 | + BM25 0.03 s when built, 0.00 s when loaded from disk |
| Kestrel | dense+bm25+graph | 33.4 | 0 | + graph: 82 LLM calls once (section 7); 0 calls and 0.03 s when unchanged |
| Own | dense | 29.0 | 0 | dense 0.02 s (cached) |
| Own | dense+bm25 | 19.2 | 0 | + BM25 0.04 s when built, 0.00 s when loaded |

Latency differences between configurations are within run-to-run noise: earlier runs measured 31 and 40 ms, then 16 and 15 ms, for
Kestrel dense and dense+bm25; this one 36 and 28 ms. The honest statement is: BM25 and the graph add no LLM call and no measurable
per-query cost; the graph's cost is its build.
The measured costs of generation (1 LLM call per question) are the same in every configuration.

## 4. Default configuration and why

- **Kestrel: `dense+bm25`, plain RRF, `candidates` = 10, top_k = 5.** Against dense alone: Hit@3 0.80 → 0.90, Hit@5 0.90 → 1.00,
  Recall@5 0.77 → 0.79, and `exact_identifier` Hit@5 0.50 → 1.00, which matters because the Kestrel runbook is organised by KST-xxxx codes.
  Against the other fusion settings it is best at k = 3 and 5 (30 candidates: Hit@5 0.90, Recall@5 0.69; weighted 2:1 with 30 candidates: Hit@3 0.70).
  Price, stated openly: Recall@3 is 0.64 against 0.72 for dense alone (`split_answer` Recall@3 0.38 against 1.00: q03 loses a hop to chunks
  that BM25 promotes), and q05 and q03 sit at rank 2 instead of rank 1 (Hit@1 stays 0.50).
- **Own corpus: `dense+bm25`, same settings.** Hit@1 0.62 → 0.69 (`exact_identifier` 0.60 → 0.80), no loss at any k ≥ 3, no cost.
- Set `DEFAULT_RETRIEVERS=dense,bm25` in `.env`; `candidates` defaults to 10 in code. Weights are an option (`--weights`), not the default,
  because they did not beat plain RRF with 10 candidates and need tuning data we do not have.
- **Caveat:** `candidates` = 10 was chosen after seeing q05, on the questions it is scored on.

## 5. BM25

- **Index:** own inverted index (`app/bm25.py`), k1 = 1.5, b = 0.75 (usual defaults, not tuned), persisted as JSON in `data/index/`
  (git-ignored), one file per corpus, chunking and tokenizer variant, written atomically (D-03, D-04).
- **Tokenization:** lowercase, accent folding, hyphenated identifiers indexed whole and split (`KST-5031` → `kst-5031`, `kst`, `5031`),
  English stop-word list. Stop-word ablation on BM25 alone (Hit@1): Kestrel 0.62 vs 0.38, own 0.75 vs 0.58; one Kestrel question loses at
  k = 3 (D-05). No stemming (not measured).
- **Updates:** a fingerprint of the chunk texts and tokenizer settings is stored in the index file. Upload, delete, selection or chunking
  changes rebuild the index on the next query; there is no incremental update (a rebuild takes 0.03 to 0.04 s at this size).
  `POST /api/index/{corpus}/rebuild` forces it. Tested in `tests/test_bm25.py` (build, reuse, reload from disk, update, removal, corrupt file).
- **Cost:** no LLM call; build 0.03 to 0.04 s; no measurable per-query cost.
- **Where it helps:** identifiers (section 3). **Where it fails:** vocabulary mismatch. q05 ("I lost my phone and I can't log in") shares no
  content word with the guide ("authenticator device", "sign in"): BM25 does not return the gold chunk at all.

## 6. Concept graph

**Status: done (v3T01 to v3T04).** The V3 submission had no graph (D-12). Since then: schema, LLM extraction, validation,
provenance, storage and build cost (section 7, DECISIONS D-20 to D-27); entity resolution with recorded merges (D-28); graph
retrieval (D-29, below); the graph path in the answer view (D-30). Measured in the matrix as `dense+bm25+graph` (section 3); what
happened to the graph's chunks after fusion is in section 8.

This section follows the brief's list: schema, entity resolution, retrieval, cost, where it fails, optional features. The numbers
behind it (counts, merges, build cost, graded triples) are in section 7.

### Schema

Fixed, versioned (`app/graph_schema.py`, D-20). **Entity types (8):** Product, Feature, Plan, Service, ErrorCode, Policy,
Procedure, Team. **Relations (10):** the PRD's eight (`has_feature`, `included_in`, `depends_on`, `causes`, `resolved_by`,
`governed_by`, `owned_by`, `replaces`) plus **`raised_by`** (ErrorCode → Service) and **`escalates_to`** (ErrorCode → Team | Service),
because the chains of q08, q11 and q12 start at an error code and no PRD relation fits "raised by postroom" or "escalate to Payments
Core". Each relation declares the types allowed at each end. Fixed rather than open: the corpus is small and known, a fixed list can be
checked, and an open one returns the same type under several names (`Team`, `Department`, `Squad`). Cost: what fits no type is dropped
(people, chat channels, prices).

### Extraction and validation

One LLM call per chunk, temperature 0 (D-21). Pydantic checks every reply against the schema and against the chunk: known types,
allowed ends, each entity named in the chunk, and **each relation's quote copied from the chunk and naming one of its ends** (D-27;
a bare "Yes" from a table cell no longer passes). Invalid reply → one retry with the errors; if the retry is invalid too, its valid
entities and relations are kept and the rest dropped and logged (D-26; `--strict` skips the chunk instead). Every node and edge keeps its
evidence `{doc, chunk_id, quote}` in SQLite (D-22). Each chunk's extraction is cached under (chunk text, schema version, prompt version),
so a changed document re-extracts only its own chunks. Prompt v1 → v2 after the first build: D-25.

### Entity resolution

Rules in order: **identifier** (error codes, IPs and plan names match exactly or not at all), **normalised name** (case, accents,
brackets, articles, the suffixes team/service), **alias** listed by the extraction (D-28). Constraints that always hold: never across
types, never two different identifiers, never the two ends of a `replaces` edge (postroom replaced mailman: q12 needs both). Every merge
is recorded with its method and evidence and can be switched off (`--rules none`). Rejected: embedding similarity and an LLM judge,
because their mistakes are false merges (KST-5031 / KST-5038, mailman / postroom) that silently send a chain to the wrong team; we
accept missed merges instead. Result on our graph: 171 → 166 entities, 5 merges, one correct cross-document merge and one costly missed
merge (section 7).

### Graph retrieval (ticket v3T03)

`app/graph_retriever.py` walks 2 hops from the entities named in the question and ranks the evidence chunks it reaches by hop
(D-29). No LLM call at query time. Graph alone on Kestrel (resolved graph of section 7, top 10; `@n` = rank of the hop's gold chunk,
`-` = not in the top 10; reproduce with `python -m app.graph search "<question>"`):

| Question | Entities found | Hits | Hop chunks |
|---|---|---:|---|
| q01 exact_identifier | KST-2044 | 2 | runbook:4 @1 (both quotes) |
| q02 exact_identifier | KST-4012 | 1 | runbook:6 @1, runbook:7 - |
| q04 split_answer | single sign-on | 10 | guide:1 @4, guide:5 @2 |
| **q08 two_hop** | KST-5031 | 6 | **runbook:8 @1, catalog:3 @5** |
| **q11 two_hop** | KST-6010 | 3 | **runbook:10 @1, catalog:4 @3** |
| **q12 two_hop** | mailman | 6 | catalog:2 @1 (two quotes), catalog:3 @4 |
| q10 not_in_corpus | Business | 5 | (none to find) |
| q03, q05, q06, q07, q09 | none | 0 | the graph adds nothing |

**q08 and q11: both hop chunks in the top 10.** The chains followed: `KST-5031 -raised_by/escalates_to-> postroom -owned_by->
Messaging` (hop 2) and `KST-6010 -escalates_to-> Payments Core` (hop 1, whose node evidence includes the team-table row with
`#oncall-payments`). **But q08's last hop is found for the wrong reason:** the team-table chunk `service-catalog_md:3` comes from the
node **Identity**, reached through the wrong edge `postroom owned_by Identity` (known failure 11); Messaging's own evidence is only
`service-catalog_md:2`, because its table row was never extracted (known failure 13). The same holds for q12's third hop. Recall@k
counts both as found; an answer built on that path could still name Identity.
**Why node evidence counts.** The ticket's literal ranking (start entities and followed edges only) finds the first hop of q08, q11 and
q12 and none of their last hops; counting the evidence of every node reached finds all of them (D-29).
**Hub cap** (15 edges) never fires: the highest degree in our graph is 5. **No entity, no graph:** 5 of the 12 questions name no
entity of the graph and get no graph hit; the unanswerable q10 gets 5 (known failure 18).
With RRF k = 60 a chunk found only by the graph scores about 1/61 and can be pushed out of the top 5 by chunks that dense and BM25 both
rank; section 8 measures it. The gap the graph was built for is in section 3: `two_hop` Recall@5 = 0.56 and Recall@10 = 0.72 for
dense and for dense+bm25, on q08, q11 (2 hops) and q12 (3 hops with an alias, mailman → postroom → Messaging; D-10).

**In the answer view** (D-30): the entities found in the question with the words that matched them, the edges followed by hop with
their chunks, and for each chunk "in the final list at rank N" or "not in the final list after fusion". A wrong match or a wrong edge
(such as `postroom owned_by Identity`) is visible to the user instead of hidden in the ranking.

### Cost

| | LLM calls | Time | When |
|---|---:|---|---|
| Build (extraction, prompt v2) | 82 | 286.0 s in LLM calls; 135,390 + 33,797 tokens | once per changed chunk |
| Rebuild with unchanged documents | 0 | 0.03 s | every build (cache) |
| Entity resolution | 0 | part of the rebuild | every build |
| Re-check after a validation change (D-27) | 0 | part of the rebuild | when `VALIDATION_VERSION` changes |
| Query | 0 | 33.4 ms per question for dense+bm25+graph (27.6 ms without the graph, same run: noise) | every question |
| Whole experiment (v1, v2, partial runs) | 164 | 504.6 s; 243,380 + 65,138 tokens | spent once |

The model is `nvidia/nemotron-3.5-lightning-30b-a3b` with hidden thinking off (D-24): thinking on cost about 16 times the completion
tokens and 9 times the time for the same JSON. BM25, for comparison, builds in 0.03 to 0.04 s with no call. The graph is the only
retriever whose index costs LLM calls, and a document change costs one or two calls per changed chunk.

### Where it fails

Measured on our graph (DECISIONS known failures 9 to 18; the 10 graded triples in section 7: 3 correct, 5 wrong, 2 unsupported):

- **Chunk boundaries.** A line read under the wrong heading gives a wrong edge whose quote is real, so validation cannot catch it:
  `postroom owned_by Identity` (heading cut at the bottom of the chunk, failure 11) and `KST-2044 escalates_to Data Integrations`
  (cut at the top, graded triple #1). A boundary also loses a subject (`ledger owned_by Payments Core`, failure 14) and cuts a name
  ("Analytics Eng", failure 16).
- **Tables.** The plans table gives no edge since D-27 and the team table is partly extracted, so q08's and q12's last hop (the
  Messaging row with its channel) is missing (failures 13, 15).
- **The schema forces a relation.** With no relation for "is printed on" or "a role can do X", the model falls back on `has_feature`,
  and roles and people become `Team` or `Product` nodes (failure 12; 4 of the 7 bad graded triples).
- **Missed merges.** One team, three nodes; walks from KST-3012 and KST-3020 stop after one hop (failure 16).
- **Right chunk, wrong reason.** q08's team-table chunk is reached through the wrong Identity edge; Recall@k counts it (failure 17).
- **No entity, no graph.** 5 of 12 questions name no entity and get nothing; the unanswerable q10 gets 5 hits (failure 18).

### Optional features

| Feature | Done? | What and why |
|---|---|---|
| **Aliases** | yes | An `aliases` table (7 aliases after resolution); the extraction lists them, the alias rule merges on them, merged names stay as aliases, and retrieval matches them when no node has that name (D-28, D-29). The `replaces` constraint keeps mailman and postroom apart even if the LLM calls one an alias of the other. |
| **Time (valid from / until)** | no | The natural use is the `versions` questions (billing policy 2025 vs 2026). Not done: the dates belong to the document, not to each fact, so they could come from the file name with no LLM call, but the retriever would then need a rule for "which version does the question mean", and the gain is small (`versions` Hit@3 is already 1.00 with dense, section 3). A schema change also bumps the prompt and re-extracts all 43 chunks (about 82 calls). |
| **Properties on nodes or edges** | no | Would hold what the fixed schema drops: a team's chat channel, a price, a plan's limit (failures 9, 15). The better fix is a deterministic parser for markdown tables (one row, one node or edge, the row as quote, no LLM), which would also restore the plans table and the Messaging row. Not done in the time we had. |
| **A graph of the own corpus** | no | `GRAPH_CORPORA` is Kestrel only; selecting the graph on the own corpus returns 503. The own corpus (a single manual, 78 chunks at 500/20) has one `two_hop` question since D-34 (OWN-15), so a build of roughly 80 to 160 calls would be measured on n = 1. |

## 7. The graph in numbers

*Ticket v3T01 (concept graph extraction).* Built with `python -m app.graph build --corpus kestrel` on Kestrel 800/10.
Numbers below are copied from `data/results/graph/build-kestrel.json`; every call is in `data/results/graph/llm_calls.jsonl`.

Model: `nvidia/nemotron-3.5-lightning-30b-a3b` on NVIDIA's endpoint, thinking off (D-24); prompt v2 (D-25); invalid items of a
twice-invalid reply dropped, the rest kept (D-26); a quote must name an end of its relation (D-27, validation v2).

| | Value |
|---|---:|
| Chunks | 43 |
| Fully valid: at first try / after one retry | 3 / 5 |
| Kept in part (invalid items dropped, D-26 and D-27) | 34 |
| Skipped (nothing valid, in `graph_errors.jsonl`) | 1 (`product-guide_md:11`) |
| Entities / relations / aliases | 166 / 49 / 7 (171 entities before resolution, below) |
| Raw triples (before identical ones are merged) | 50 |
| Relations dropped by the D-27 re-check (0 calls) | 62 (57 from the plans table) |
| LLM calls behind this graph | 82 |
| Prompt + completion tokens | 135,390 + 33,797 |
| Time in LLM calls | 286.0 s (about 3.3 s per call) |
| Rerun with unchanged documents | 0 LLM calls, 0.03 s |
| Integrity: nodes / edges without evidence, unknown chunk ids | 0 / 0 / 0 |
| Whole experiment (prompt v1 + v2, all runs) | 164 calls, 243,380 + 65,138 tokens, 504.6 s |

Chain edges (q08, q11, q12): `KST-5031 raised_by postroom`, `KST-5031 escalates_to postroom`, `postroom owned_by Messaging`,
`postroom replaces mailman` and `KST-6010 escalates_to Payments Core` are in the graph (unchanged by D-27); `ledger owned_by Payments Core` is not
(section cut by a chunk boundary). The last hop of q08 / q12 (the Messaging row of the team table, with its channel) is missing, and one
wrong edge, `postroom owned_by Identity`, competes with the right one. The plans table gives no edge since D-27. Details: DECISIONS known failures 10 to 15.
Relations by type: has_feature 18, owned_by 11, escalates_to 7, causes 5, raised_by 4, included_in 2, replaces 1, governed_by 1
(before D-27: 111 relations, of which has_feature 51 and included_in 27).

Schema, validation, storage, model choice: DECISIONS D-20 to D-24. Prompt v1 → v2 (v1 skipped 22 of 43 chunks): D-25. Keeping the
valid part of a failed reply: D-26. Retrieval: section 6 (v3T03). Extraction quality: 10 graded triples, end of this section (v3T06).

### Entity resolution (ticket v3T02)

Rules identifier → normalised → alias, never across types, identifiers or `replaces` edges (DECISIONS D-28). Run from the cache with
`python -m app.graph build --rules none` and with the default rules: 0 LLM calls each. Every merge is in
`data/results/graph/merges.jsonl`.

| | `--rules none` | default rules |
|---|---:|---:|
| Entities | 171 | **166** |
| Merges (identifier / normalised / alias) | 0 | 5 (3 / 2 / 0) |
| Refused merges | 0 | 0 |
| Aliases | 1 | 7 |
| Relations | 49 | 49 |
| Integrity (incl. every merge has a method and evidence) | OK | OK |

The five merges: `Enterprise plan` and `Enterprise plans` → `Enterprise`, `Free plan` → `Free` (identifier: plan names);
`Single sign-on (SAML)` → `single sign-on`, `Platform team` → `Platform` (normalised). No rule merged two different things that we could
find. The alias rule found nothing to merge on this graph and the mailman trap did not occur (the LLM did not list mailman as an alias of
postroom); both are covered by unit tests, and a `replaces` edge blocks that merge if it ever happens.

**A correct merge, across documents.** "Platform team" (runbook, `support-runbook-errors_md:0`) and "Platform" (service catalog) are one
`Team` node (method `normalised`). Its evidence now spans both documents: the catalog's header ("Owner: Platform team"), the `Owner:
**Platform**.` lines of gateway, auth and embed-svc (`service-catalog_md:2`), and the team-table row `| Platform | Samuel Okafor |
#oncall-platform | 24/7 rotation |` (`service-catalog_md:3`). Without the merge the runbook's "Platform team" is an isolated node with
no edge.

**A missed merge, and what it cost.** "Analytics Engine" is three nodes (DECISIONS known failure 16):

| Node | Edge | Why not merged |
|---|---|---|
| Analytics Engine (Team) | query-engine, scheduler `owned_by` it | the right node |
| Analytics Engine (Service) | KST-3020 `escalates_to` it | the LLM typed the team as a Service; resolution never crosses types |
| Analytics Eng (Team) | KST-3012 `escalates_to` it | the 800-character chunk boundary cuts the name in `support-runbook-errors_md:5` |

Cost: `python -m app.graph walk KST-3012` and `walk KST-3020` stop after one hop, at a node with no other edge, so a graph path for
either error never reaches the team that owns the query engine or its channel. The type constraint that causes the second miss is the one
that keeps the role "Owner" apart from "Owner: Messaging"; we keep it and accept this miss.

**The q08 chain after resolution** (`python -m app.graph walk KST-5031 --hops 2`, abridged):

```
KST-5031 (ErrorCode)   chunks: support-runbook-errors_md:7, support-runbook-errors_md:8
    [1] KST-5031 -escalates_to-> postroom   (edge from support-runbook-errors_md:8)
        postroom (Service)   chunks: service-catalog_md:1, service-catalog_md:2, support-runbook-errors_md:8
        [2] postroom -owned_by-> Identity   (edge from service-catalog_md:1)
            Identity (Team)   chunks: service-catalog_md:1, service-catalog_md:3, support-runbook-errors_md:3
        [2] postroom -owned_by-> Messaging   (edge from service-catalog_md:2)
            Messaging (Team)   chunks: service-catalog_md:2
        [2] postroom -replaces-> mailman   (edge from service-catalog_md:2)
```

The walk crosses from the runbook to the service catalog and reaches the Messaging team's evidence chunk there (`service-catalog_md:2`,
"Owner: **Messaging**."), the acceptance criterion of v3T02. It does not reach the Messaging row of the team table (with
`#oncall-messaging`): that row was never extracted (known failure 13), so there is nothing to merge it with. The walk does reach the team
table chunk, but through the wrong edge `postroom owned_by Identity` (known failure 11): a retriever must not trust a single path.

### Extraction quality: 10 triples graded by hand (ticket v3T06)

`python -m app.graph sample --n 10 --seed 42`: 10 of the 49 edges of the resolved graph (graph hash `a461fb5b66bf`), graded against
their evidence chunks with rules committed before any grade (`data/results/v3/graph_triples_grading_rules.md`). Graded by Muhammad
Saad JAVED (grades drafted with AI assistance, each checked against its chunk), reviewed by Sultan Amiour. Details:
`data/results/v3/graph_triples_graded.md`.

**Counts: correct 3 / wrong 5 / unsupported 2.** Ten triples are a sanity check, not an accuracy measure: one triple moves a rate by
10 points.

One disagreement, settled: #5 `Kestrel has_feature Forgot password` was graded correct and reviewed unsupported. The sentence never
names Kestrel; only the guide as a whole is about it. By our rules (only the evidence chunk counts; when unsure, the worse grade) it is
**unsupported**.

What the 7 bad triples have in common: four force a relation onto something that is not a feature or not a product (#2, #3: company
details are *printed on* invoices; #8, #10: `Owner` is a role, typed `Product`). The schema has no relation for "a role can do X" or
"is printed on", so the model falls back on `has_feature`. One takes the wrong entity from the same section (#4: dashboards are sent on
a schedule, not datasets). One is inferred (#5). One comes from a chunk boundary (#1, below).

**One bad triple and what it cost: #1 `KST-2044 escalates_to Data Integrations`, wrong.** Chunk `support-runbook-errors_md:4`
starts in the middle of KST-2040's section, so KST-2040's line "**Escalate:** to Data Integrations if several customers report it…"
sits just above the `### KST-2044` heading, and the model attached it to the only error code in sight. KST-2044's own section says
"**Escalate:** no." It is the mirror of `postroom owned_by Identity` (known failure 11): there the heading was cut at the bottom of
the chunk, here at the top. Cost: q01 asks exactly about KST-2044, and `python -m app.graph search` for q01 follows this edge at hop 1,
so the graph path shown in the UI tells support to escalate an error the runbook says not to escalate. The quote check (D-21, D-27)
cannot catch it: the quote is in the chunk and names an end of the relation.

### How we would evaluate a concept graph beyond 10 triples

Ten graded triples say little: 3 correct of 10 gives a 95% interval of about 0.11 to 0.60 (Wilson). What we would measure next, cheapest
first:

1. **Automatic checks, every build (0 calls).** Already: integrity (no node or edge without evidence, no unknown chunk id). Easy to add:
   rules the domain gives, such as "a service has one owner" (would flag `postroom owned_by Identity`), "an error code escalates to one
   place", and type sanity (a `Team` named like a person or a channel).
2. **Precision on a larger, stratified sample.** About 50 triples, sampled per relation type and per chunk outcome (fully valid vs
   kept in part, D-26), so the weak strata are visible instead of averaged away; two graders on the same triples, with their agreement
   reported (we had one disagreement in ten, #5).
3. **Recall on a small gold set.** Write by hand every triple of a few chunks (the team table, two runbook sections) and count how many
   the graph has. Precision alone hides what was never extracted, and our worst losses are of that kind (failures 13, 15).
4. **Entity resolution.** Label the true clusters of the Team and Service nodes and count wrong merges and missed merges
   (we know of 0 wrong and at least 2 missed, failure 16).
5. **The path, not only the chunk.** For each `two_hop` question, is every hop reached through a correct edge? Recall@k counts q08's
   last hop as found through a wrong edge (failure 17), and the answer view shows that path to the user.
6. **Downstream.** Per-hop Recall@k with and without the graph, and whether the graph's chunks reach the top k after fusion (section 3
   and section 8). This is the measure that matters for retrieval, but with 3 `two_hop` questions one question moves Recall by up to 0.33.

## 8. Fusion: what happened to chunks after RRF

The mechanism was first measured with BM25 (below); the graph's chunks are at the end of this section. RRF uses ranks only, so a chunk found by two retrievers beats a chunk
found by one, even when the second vote comes from deep in a list. Levers: `candidates`, `--rrf-k`, `--weights`. Measured on Kestrel:

| Setting | Hit@3 | Hit@5 | Recall@3 | Recall@5 | q05 |
|---|---:|---:|---:|---:|---|
| dense alone | 0.80 | 0.90 | 0.72 | 0.77 | rank 1 |
| RRF, 30 candidates | 0.80 | 0.90 | 0.54 | 0.69 | rank 6 |
| RRF, 30 candidates, weights dense=2 bm25=1 | 0.70 | 0.90 | 0.54 | 0.77 | not fixed |
| **RRF, 10 candidates (default)** | **0.90** | **1.00** | 0.64 | **0.79** | **rank 2** |

Not tried: `--rrf-k` below 60, a score threshold on dense, a cross-encoder re-ranker.

### The graph's chunks after fusion (`dense+bm25+graph`)

Final rank of each hop's gold chunk in the fused top 10, without → with the graph (`-` = not in the top 10; graph rank = rank in the
graph's own list; from `kestrel__dense+bm25.json` and `kestrel__dense+bm25+graph.json`):

| Question | Hop chunk | dense+bm25 | + graph | Graph rank | Why |
|---|---|---:|---:|---:|---|
| q08 | support-runbook-errors_md:8 | 1 | 1 | 1 | already first |
| q08 | service-catalog_md:3 | - | **-** | 5 | graph only: 1/65 ≈ 0.0154, below the 10th chunk (0.0156) |
| q11 | support-runbook-errors_md:10 | 2 | **1** | 1 | third vote |
| q11 | service-catalog_md:4 | 6 | **2** | 3 | third vote (dense 10, BM25 2) |
| q12 | service-catalog_md:2 (hops 1 and 2) | 1 | 1 | 1 | already first |
| q12 | service-catalog_md:3 | - | **-** | 4 | graph only: 1/64 ≈ 0.0156, below the 10th chunk (0.0288) |
| q01 | support-runbook-errors_md:4 | 4 | **1** | 1 | third vote (dense 7, BM25 1) |
| q04 | product-guide_md:1 | 4 | 5 | 4 | `product-guide_md:4` (graph rank 1) overtakes it; the top 6 all have three votes |

**The answer to the brief: the graph moves `two_hop` Recall@5 from 0.56 to 0.72, and Recall@10 stays at 0.72.** The graph's chunks
that the other retrievers also return reach the top k, often first, because a third vote outscores two. The one hop only the graph finds,
the team table for q08 and q12, does not: a chunk with one vote at rank 4 or 5 cannot beat chunks with two votes, which is what
section 6 predicted for RRF with k = 60. That chunk is also reached through the wrong edge `postroom owned_by Identity` (known
failure 17), so the hop the fusion drops is the one the graph found for the wrong reason.

Worse with the graph: no question at its first gold chunk (summary). One later hop moves down, q04's `product-guide_md:1`, rank 4 → 5,
still in the top 5. **Unanswerable q10** ("Business") is not scored, but 4 of the graph's chunks are in its final top 10, two of them at
ranks 1 and 2 with three votes; an answer generated from that list has more plausible context to wrongly answer from (known failure 18).
Caveat: 3 `two_hop` questions; the gain at k = 3 and 5 is one hop of one question (q11).

Not tried: weighting the graph above 1 in RRF (`--weights`), which would let a graph-only hop at rank 4 or 5 enter the top 10 but would
also promote the graph's noise (q10, q04), and has no held-out questions to be tuned on.

## 9. A question where adding a retriever made things worse

**Kestrel q05**, "I lost my phone and I can't log in anymore. What can I do?" (`vocabulary`). Gold chunk `product-guide_md:6`.
Dense ranks it **1st**; BM25 does not return it at all. With plain RRF (k = 60, 30 candidates) it falls to rank **6**: five chunks that
are weak in dense (ranks 2, 16, 22, 27, 29) but present in BM25's top 6 get two votes and outscore it.

| Chunk | Dense rank | BM25 rank | RRF score |
|---|---:|---:|---:|
| support-runbook-errors_md:2 | 2 | 1 | 0.0325 |
| service-catalog_md:3 | 16 | 6 | 0.0283 |
| support-runbook-errors_md:6 | 22 | 3 | 0.0281 |
| billing-policy-2025_md:2 | 27 | 2 | 0.0276 |
| product-guide_md:2 | 29 | 4 | 0.0269 |
| **product-guide_md:6 (gold)** | **1** | none | **0.0164** |

Why: RRF discards scores. Dense ranks 16 to 29 have cosine scores of 0.05 to 0.15 (noise); the gold chunk has 0.41; rank-only fusion
still treats "found by both" as strong evidence, and BM25 matched only generic words here. Weights dense=2 bm25=1 did not fix it.

**Fix, predicted then measured.** With 10 candidates, only chunks inside dense's top 10 keep their dense vote, so four of the five
displacers fall to a single BM25 vote (1/62 to 1/66, below the gold chunk's 1/61). Measured: q05 goes **rank 6 → rank 2** (Hit@3
0.00 → 1.00); Kestrel overall Hit@3 0.80 → 0.90, Hit@5 0.90 → 1.00. It does not change `split_answer` (Recall@3 0.38) or `two_hop`
(Recall@5 0.56), and q05 is still not rank 1. The setting was chosen after looking at this question (D-18).

In that experiment q05 is a real loss by the summary's rule (v3T21): a question counts as **worse** only when a gold chunk leaves the
top 5 the LLM receives, here recall@5 1.00 → 0.00 (rank 1 → 6). With the final 10 candidates it stays inside (rank 1 → 2) and is
listed as *moved within the top 5*, like own OWN-01 (4 → 5) and OWN-02 (2 → 3).

The final run has one real loss when BM25 is added: **Kestrel q03** (`split_answer`), "What happens, step by step, when a customer's
card payment fails?". Its first gold chunk only moves from rank 1 to 2, which the old rule reported as a small reshuffle, but recall@5
drops **1.00 → 0.25**: three of its four answer pieces leave the top 5, so the answer can no longer list every step. Adding the graph
does not bring them back. The full lists (worse, better, moved) are in `data/results/v3/summary.md`.

## 10. Decision log and known failures

`DECISIONS.md`, D-01 to D-18 (V3), D-20 to D-27 (concept graph extraction), D-28 (entity resolution), D-29 (graph retrieval), D-30
(graph path in the UI), D-31 (answer validation), D-32 (shared tokenizer), D-33 (own-corpus storage), D-34 (own question labels),
D-35 (answers with a reasoning model), and its "Known failures" section. In short: BM25 returns nothing for vocabulary-mismatch questions;
rank-only fusion lets double votes displace a correct single-retriever hit (q05), and the graph's only new hop (section 8);
`split_answer` Recall@3 drops with BM25; `two_hop` Recall@5 is 0.56 with or without BM25 and 0.72 with the graph, Recall@10 is 0.72 in
every configuration; the English-only stop-word list; no incremental BM25 update; gold chunks are found by quote matching
(a quote cut by a chunk boundary yields no gold chunk; none occurs in our question files, checked by `tests/test_questions.py`);
graph retrieval can reach the right chunk through a wrong edge, and adds nothing to a question that names no entity (known failures
8 to 18 for the graph).

## 11. How to reproduce

```bash
pip install -r requirements.txt
cp .env.example .env                                  # LLM_* only needed to generate answers
python -m unittest discover -s tests -t .             # all tests; no model download, no API key
python -m app.graph build --corpus kestrel            # concept graph extraction (LLM_* in .env), section 7
python -m app.evaluate matrix --required-only         # data/results/v3/ (summary.md = every table above)
python -m app.evaluate run --corpus kestrel --retrievers dense --ids q01,q02,q03,q04,q05,q06,q07,q08,q09,q10   # V2 check
# ablations with 30 candidates:
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --candidates 30 --out-dir data/results/v3-candidates30
python -m app.evaluate run --corpus kestrel --retrievers dense,bm25 --candidates 30 --weights dense=2,bm25=1 --out-dir data/results/v3-weighted
```

Dense runs in strict mode: the evaluation stops rather than silently measuring TF-IDF.

## 12. V4: Concept map explorer (ticket v4T07)

A **Concept map** tab (Kestrel): type a concept (an error code, a service, a team, a plan) and see what it is linked to, by which
relation, in which direction, and the passage each link was read from. Click a linked concept to move to it; **Documents** lists the
documents that mention it; **Ask about this** opens the Ask tab on Kestrel with the concept graph switched on and asks about the
concept. Endpoint `GET /api/graph/concept?name=`, built on the V3 graph search (`graph_search.neighborhood()`): the concept is found
with the same keys as retrieval (node names, then aliases, then the question matcher, D-29), so "SSO", "Platform team" or "the
Messaging team" work. Each lookup is one `concept.lookup` trace (spans `graph.load` and `concept.match`; outcome `found` or
`not_found`); the chunk texts go to the tab, never into the trace.

**Hub cap.** In a walk (V3) a hub is reached but not expanded. In a list of neighbours it means: at most 15 links are shown, the rest
are counted ("N more link(s) not shown"), and a neighbour with more than 15 links is tagged *hub*. On our graph it never fires: the
most connected node is postroom with 5 links. It is tested on a hand-made graph with 21 links.

### Cost and latency

| | LLM calls | Time |
|---|---:|---|
| First lookup after the app starts (loads the graph and the chunks) | 0 | 441 ms |
| Each lookup after that (9 cases × 20 runs, n = 180) | 0 | median **34.6 ms**, p95 47.6 ms |

Measured in-process (FastAPI test client, same laptop, real Kestrel graph of section 7). Most of the 35 ms is rebuilding the chunk
list to attach document names and passages; the graph itself is cached per build. No LLM call, no token, no cost.

### Cases tried

| Typed | Why | What the tab shows |
|---|---|---|
| `KST-5031` | error code, start of q08 | 2 links, both to postroom (`raised_by`, `escalates_to`), from `support-runbook-errors_md:8` |
| `postroom` | most connected node | 5 links: KST-5031 twice, `owned_by` **Identity** and `owned_by` **Messaging**, `replaces` mailman |
| `Platform team` | a merged name | the merged `Platform` node (found by its normalised name), owner of auth, gateway and embed-svc |
| `SSO` | an alias | `single sign-on` (found by alias; also written "Single sign-on (SAML)"), 4 links |
| `mailman` | replaced service | its own node, linked to postroom by `replaces`: the alias trap of q12 does not occur here either |
| `the Messaging team` | a phrase | the `Messaging` node, 1 link (`postroom owned_by Messaging`) |
| `KST-2044` | the graded bad triple #1 | `escalates_to Data Integrations` (wrong, section 7) and `raised_by Kestrel` |
| `Team` | a generic word | two nodes, the plan and a team, with 0 links each |
| `KST-9999` | unknown | "No concept called … Try an error code (KST-5031), a service (postroom) or a team (Messaging)." |

**What we found.**

- **The explorer shows the graph's mistakes to the user, with their evidence.** postroom has two owners, Identity (known failure 11,
  read from `service-catalog_md:1`, where the "Owner" line belongs to auth) and Messaging; KST-2044 shows the escalation the runbook
  says not to do. In the Ask tab these edges were hidden inside a ranking; here a user sees two owners side by side and can open both
  passages. It is also the quickest way we have to inspect the graph.
- **Duplicate links between the same two concepts** (KST-5031 `raised_by` and `escalates_to` postroom) are two lines with the same
  passage. Correct, but a user may read them as two facts.
- **A generic word finds nodes but nothing useful:** "Team" is two nodes without links. And "Ask about this" on such a concept gets no
  graph entity, because the question matcher ignores generic words on purpose (D-29); dense and BM25 still answer.
- **What the graph never extracted cannot be explored:** the Messaging row of the team table (known failure 13) is not there, so
  Messaging shows no on-call channel.

Tests: `tests/test_concept_map.py` (hand-made graph: neighbours, relations, directions, chunks, alias, unknown, hub cap; the fixture
graph of `test_graph_search`: KST-5031 → postroom with its source chunk; the API: the trace, a 503 when the graph is missing, a 400 on
the own corpus, and the "Ask about this" question producing a `retrieve.graph` span).
