# RAG V3: Team Evaluation Report

## 1. Team and release

- Team members:
  - **Sultan AMIOUR**
  - **Muhammad Saad JAVED**
  - **Jaimie MATHANGI**
- Team of 3 (GitHub: `s-amiour`, `Msaadjaved`, `Chimaera-Emerald`); Jaimie joined for V4. The "teams of 3" extra piece applies:
  weighted RRF compared with plain RRF (ticket B7).

## 2. Summary

- V3 adds **BM25** (both corpora, persistent index), **RRF fusion** with every retriever switchable from the UI, the API and the
  evaluation command, and **Recall@k per hop** next to V2's HitRate@k.
- **Concept graph: extraction, entity resolution and graph retrieval done (tickets v3T01 to v3T03; sections 6 and 7); UI path pending
  (T4).** Graph alone puts both hop chunks of q08 and q11 in its top 10 (section 6). The third required configuration
  (`dense+bm25+graph`) now runs; its measurement is T5.
- **BM25 helps, but only once fusion is configured carefully.** On Kestrel, plain RRF with 30 candidates per retriever left Hit@k unchanged
  (0.50 / 0.80 / 0.90 / 1.00 at k = 1, 3, 5, 10) and lowered Recall@3 from 0.72 to 0.54. Limiting each retriever to its 10 best candidates
  (D-18) gives Hit@3 **0.90**, Hit@5 **1.00** and Recall@5 **0.79**, the best of all configurations at k = 3 and 5. It fixes
  `exact_identifier` (Hit@5 0.50 → 1.00). It costs a little on rank 1 (Hit@1 stays 0.50, q05 and q03 drop from rank 1 to 2) and on
  `split_answer` Recall@3 (1.00 → 0.38).
- On our own corpus: Hit@1 0.67 → 0.75, no loss at k ≥ 3, identical with 10 or 30 candidates.
- BM25 needs no LLM call, builds in 0.03 to 0.04 s and adds no measurable query latency (15 to 20 ms per query in every configuration).
- **Limit of this work:** `two_hop` Recall stays at 0.56 at k = 5 and 0.72 at k = 10 with or without BM25. Finding every hop of a chain is
  the unsolved part, and it is what the graph was meant to address.
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
| dense+bm25+graph | | | not run (section 6) | | | | | | | |

### Own corpus (500/20), identical with 10 and 30 candidates

| Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | **ALL** | 12 | 0.67 | 0.92 | 1.00 | 1.00 | 0.67 | 0.92 | 1.00 | 1.00 |
| dense | exact_identifier | 5 | 0.60 | 0.80 | 1.00 | 1.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| dense | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| **dense+bm25 (default)** | **ALL** | 12 | **0.75** | 0.92 | 1.00 | 1.00 | **0.75** | 0.92 | 1.00 | 1.00 |
| dense+bm25 | exact_identifier | 5 | **0.80** | 0.80 | 1.00 | 1.00 | **0.80** | 0.80 | 1.00 | 1.00 |
| dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | two_hop | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| dense+bm25 | not_in_corpus_trap | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

Reading the tables:

- **`exact_identifier`:** BM25 helps on both corpora (Kestrel Hit@5 0.50 → 1.00; own Hit@1 0.60 → 0.80).
- **`vocabulary`:** BM25 cannot help (no shared word) and can hurt through fusion (q05, section 9).
- **`versions`:** Hit@1 0.00 → 0.50 with BM25; Hit@3 is already 1.00 with dense.
- **`two_hop`:** Hit@10 is 1.00 but Recall@10 only 0.72: every chain is "found" but not every hop. In the V2 check, q08 alone has
  Recall@10 = 0.50 with dense: one of its two hops is not in the top 10.
- **`not_in_corpus`:** the two Kestrel questions are excluded from the metrics. The own corpus has none: its `not_in_corpus_trap`
  question (OWN-04, vinegar) is answerable and therefore scored (D-09). OWN-05 (`two_hops`) has a single evidence quote, so Hit = Recall there.
  The own corpus has no `versions` question. Type names in `own.jsonl` (`different_vocabulary`, `two_hops`) are mapped to the assignment's
  names when reporting; the file is untouched.

### Cost per question (no answer generation)

| Corpus | Configuration | Retrieval ms | LLM calls | Index build |
|---|---|---:|---:|---|
| Kestrel | dense | 16.4 | 0 | dense 22.6 s (model load; embeddings cached on disk after the first run) |
| Kestrel | dense+bm25 | 15.2 | 0 | + BM25 0.03 s when built, 0.00 s when loaded from disk |
| Own | dense | 20.0 | 0 | dense 0.02 s (cached) |
| Own | dense+bm25 | 13.0 | 0 | + BM25 0.04 s when built, 0.00 s when loaded |

Latency differences between configurations are within run-to-run noise (an earlier run measured 31 ms and 40 ms for Kestrel
dense and dense+bm25; this one 16 ms and 15 ms). The honest statement is: BM25 adds no LLM call and no measurable per-query cost.
The measured costs of generation (1 LLM call per question) are the same in every configuration.

## 4. Default configuration and why

- **Kestrel: `dense+bm25`, plain RRF, `candidates` = 10, top_k = 5.** Against dense alone: Hit@3 0.80 → 0.90, Hit@5 0.90 → 1.00,
  Recall@5 0.77 → 0.79, and `exact_identifier` Hit@5 0.50 → 1.00, which matters because the Kestrel runbook is organised by KST-xxxx codes.
  Against the other fusion settings it is best at k = 3 and 5 (30 candidates: Hit@5 0.90, Recall@5 0.69; weighted 2:1 with 30 candidates: Hit@3 0.70).
  Price, stated openly: Recall@3 is 0.64 against 0.72 for dense alone (`split_answer` Recall@3 0.38 against 1.00: q03 loses a hop to chunks
  that BM25 promotes), and q05 and q03 sit at rank 2 instead of rank 1 (Hit@1 stays 0.50).
- **Own corpus: `dense+bm25`, same settings.** Hit@1 0.67 → 0.75 (`exact_identifier` 0.60 → 0.80), no loss at any k ≥ 3, no cost.
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

**Status: extraction, entity resolution and graph retrieval done (v3T01 to v3T03); UI path pending (T4).** The V3 submission had
no graph. Since then:

- done: schema, LLM extraction, validation, provenance, storage, build cost (section 7, DECISIONS D-20 to D-27); entity resolution
  with recorded merges (D-28); graph retrieval (D-29, below);
- pending: the path display in the UI (T4); the `dense+bm25+graph` measurement and the "graph's chunks after fusion" analysis (T5).

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
rank; measuring that is T5.

What exists and works toward it: the retriever contract (`app/retrieval.py`, README), the switches in the UI, the API and the evaluation
command (`--retrievers dense,bm25,graph` is accepted and fails with a clear message while `app/graph_retriever.py` is absent), the graph-path
panel in the UI, per-hop Recall@k, and two new Kestrel `two_hop` questions with one evidence quote per hop (q11, 2 hops; q12, 3 hops with an
alias, mailman → postroom → Messaging). The measured gap the graph would have to close is in section 3: `two_hop` Recall@5 = 0.56 and
Recall@10 = 0.72 for dense and for dense+bm25. Open design decisions are listed in D-12.

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
valid part of a failed reply: D-26. Retrieval and extraction-quality grades: not done yet (next tickets).

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

## 8. Fusion: what happened to chunks after RRF

There is no graph, but the same mechanism was measured with BM25. RRF uses ranks only, so a chunk found by two retrievers beats a chunk
found by one, even when the second vote comes from deep in a list. Levers: `candidates`, `--rrf-k`, `--weights`. Measured on Kestrel:

| Setting | Hit@3 | Hit@5 | Recall@3 | Recall@5 | q05 |
|---|---:|---:|---:|---:|---|
| dense alone | 0.80 | 0.90 | 0.72 | 0.77 | rank 1 |
| RRF, 30 candidates | 0.80 | 0.90 | 0.54 | 0.69 | rank 6 |
| RRF, 30 candidates, weights dense=2 bm25=1 | 0.70 | 0.90 | 0.54 | 0.77 | not fixed |
| **RRF, 10 candidates (default)** | **0.90** | **1.00** | 0.64 | **0.79** | **rank 2** |

Not tried: `--rrf-k` below 60, a score threshold on dense, a cross-encoder re-ranker.

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

Other questions that get worse with BM25, all staying in the top 5: Kestrel q03 (`split_answer`) rank 1 → 2; own OWN-01 rank 4 → 5;
own OWN-02 rank 2 → 3. The full automatic list is in `data/results/v3/summary.md`.

## 10. Decision log and known failures

`DECISIONS.md`, D-01 to D-18 (V3), D-20 to D-27 (concept graph extraction), D-28 (entity resolution) and D-29 (graph retrieval), and its "Known failures" section. In short: BM25 returns nothing for vocabulary-mismatch questions;
rank-only fusion lets double votes displace a correct single-retriever hit (q05); `split_answer` Recall@3 drops with BM25; `two_hop`
Recall@5 and @10 are unchanged by BM25 (0.56, 0.72); the English-only stop-word list; no incremental BM25 update; gold chunks are found by quote matching
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
