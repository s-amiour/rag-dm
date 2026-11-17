# Decision log (V3)

Format: decision, options considered, choice and why. D-12 records that the concept graph was not implemented in V3;
D-20 to D-27 record its extraction step (V4, ticket v3T01). D-19 (the v1 tag) is added by branch `docs/d19-v1-tag`.
V2 decisions (chunk size, overlap, top-k, embedding model) are in `docs/REPORT_V2.md`.

## Fusion

**D-01. How to combine retrievers.** Options: (a) weighted sum of normalised scores (min-max / z-score), (b) RRF, (c) a learned
re-ranker / cross-encoder, (d) interleaving. **Chosen: RRF.** Cosine similarity, BM25 and graph scores are on different scales
and BM25 scores are unbounded; RRF uses ranks only, needs no tuning data (we have 22 questions) and is the standard baseline.
k = 60 (original paper); each retriever contributes its top `candidates` = 10 (D-18). A cross-encoder would add a model and per-query
cost we cannot justify on corpora of 43 and 78 chunks. Weighted RRF is supported (`--weights`) but plain RRF is the default.

**D-02. One retriever on = no fusion.** The ranking is returned as is, so the `dense` configuration reproduces V2 exactly.
Zero-score BM25 chunks are never returned: they carry no evidence and would enter the fusion as noise.

**D-18. How many candidates each retriever gives to the fusion (default 10, was 30).** With 30, a chunk that dense ranks 16th to
29th (cosine 0.05 to 0.15, i.e. noise) still earns an RRF vote, and if BM25 also returns it, "found by both" outscores a correct
chunk found by one retriever only. Kestrel q05 showed it: gold chunk dense #1, absent from BM25, pushed to rank 6 by five such chunks
(REPORT section 9). Options measured on Kestrel (10 answerable questions, k = 3 / 5, Hit and Recall):
dense alone 0.80 / 0.90 and 0.72 / 0.77; plain RRF with 30 candidates 0.80 / 0.90 and 0.54 / 0.69; weights dense=2 bm25=1 with 30
candidates 0.70 / 0.90 and 0.54 / 0.77; **10 candidates 0.90 / 1.00 and 0.64 / 0.79**. Caveat: the value was chosen after looking at q05,
on the same 10 questions it is evaluated on, so the gain is optimistic; we have no held-out set. Other levers not tried:
RRF k below 60, a score threshold on dense, a cross-encoder re-ranker.

## BM25

**D-03. Implementation.** Options: `rank_bm25` (in-memory, no persistence), SQLite FTS5 (persistent, built-in `bm25()`),
Whoosh / Tantivy / Pyserini, Elasticsearch/OpenSearch (a service to run). **Chosen: our own ~100-line inverted index**
(`app/bm25.py`): no new dependency, formula visible and unit-tested, and the same chunk ids as the other retrievers. FTS5 would
also have been valid; we rejected it because its tokenizer makes the identifier handling in D-05 harder to control.
Parameters k1 = 1.5, b = 0.75 (usual defaults, not tuned: 22 questions are too few to tune without overfitting).

**D-04. Where the index lives and how it is updated.** Persisted as JSON in `data/index/` (git-ignored: derived data), one file per
(corpus, chunk size, overlap, stop-word variant). The file carries a fingerprint of the chunk texts + tokenizer settings. If
documents are uploaded, deleted, (de)selected, or chunking changes, the fingerprint differs and the index is rebuilt on the next
query; written atomically (temp file + rename), so a crash never leaves a half index. **No incremental update**: rebuilding 43 or 78
chunks takes ~10 ms, and a full rebuild cannot drift from the documents. This would stop being true for a corpus of thousands of
chunks (then: per-document postings). `POST /api/index/{corpus}/rebuild` forces a rebuild.

**D-05. Tokenization.** (1) lowercase and fold accents (`Inès` = `ines`); (2) hyphenated identifiers are indexed whole *and* split:
`KST-5031` → `kst-5031`, `kst`, `5031`. The whole token is rare (high IDF) and separates KST-5031 from KST-5038; the parts keep
partial queries working. (3) **English stop-word list, on by default.** Measured on BM25 alone (original 10 Kestrel + 12 own
questions, before q11/q12; reproduce with `python -m app.evaluate matrix`):

| Corpus | Variant | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
|---|---|---:|---:|---:|---:|
| Kestrel | with stop words | 0.62 | 0.75 | 0.88 | 0.88 |
| Kestrel | without | 0.38 | 0.88 | 0.88 | 0.88 |
| Own | with stop words | 0.75 | 0.83 | 0.92 | 1.00 |
| Own | without | 0.58 | 0.75 | 0.92 | 1.00 |

Why: without it, a question such as "What is error E47 and how should it be fixed?" lets low-IDF words (`what`, `how`,
`should`, `and`) add up over a short chunk and outrank the chunk holding the single rare token `e47` (gold chunk ranked 4th instead of 1st).
Cost: one Kestrel question (q03) slips from rank 2 to 4 at k = 3. Limitation: English list only (both corpora are English).
No stemming/lemmatisation: it would help "refund/refunded" but also merge unrelated words; not measured, not done.

## Dense retriever

**D-06. Caching and strict mode.** V2 rebuilt the retriever (re-encoding every chunk) at every request. V3 caches the model and the
retriever in memory and the embeddings on disk (`data/index/dense-*.npz`, same fingerprint idea). Evaluation runs in **strict mode**:
V2's `Retriever` catches any error while loading sentence-transformers and silently continues with TF-IDF; the backend was recorded in
the V2 result files (the final ones say `sentence-transformers`, 8 of the 24 files say `tfidf`) but nothing in the UI warned about it. For a
comparison between retriever configurations the evaluation must fail loudly rather than measure the wrong retriever, and the
summary prints a warning if a fallback was allowed (`--allow-tfidf-fallback`).

## Evaluation

**D-07. Metrics.** Hit@k is V2's metric (kept so numbers stay comparable). Recall@k is per *hop* (one hop per evidence quote), so it
shows whether *all* hops of a chain were retrieved, which Hit@k hides (V2 report §8.9). A hop whose quote matches no chunk counts as
not retrieved and is listed (`questions_with_unmatched_hops`). `not_in_corpus` questions are excluded from both metrics (no gold
chunk), as in V2. Relevance = chunk contains the quote (V2 rule, unchanged).

**D-08. Chunking per corpus.** V3 keeps V2's final chunking (Kestrel 800/10, own 500/20) so the only thing that changes between the
V2 row and the V3 rows is the retriever set. The `dense` row is the re-measured V2 baseline on the extended question set;
`--ids q01..q10` reproduces it on the original ten.

**D-09. Question types and files.** Question files are append-only. `own.jsonl` uses `different_vocabulary` and `two_hops`, the
assignment uses `vocabulary` and `two_hop`: we map names at report time (`type_raw` keeps the original) instead of editing old
rows. `not_in_corpus_trap` (OWN-04) is **not** mapped to `not_in_corpus`: it is answerable (the manual says never to use vinegar),
so our own corpus has no truly unanswerable question; the report says so instead of pretending.

**D-10. New questions.** q11 (KST-6010 → Payments Core → #oncall-payments; cross-document, 2 hops) and q12 (mailman → postroom →
Messaging → channel; an alias plus 3 hops, which exercises entity resolution). Both: the question names neither intermediate entity,
and every hop has an evidence quote that matches exactly one chunk at 800/10 (checked by `tests/test_questions.py`).
Known weakness of q12: its first two hops sit in the same chunk, so a retriever can get 2/3 of the recall with one chunk.

## Concept graph

**D-11. Integration.** The graph is a retriever like the others: `search(...)` returns ranked chunk ids, a path for the UI and its
query-time cost (contract in `app/retrieval.py` and the README). Chunk ids depend on chunking, so the graph is built on 800/10.

**D-12. Graph not implemented.** Schema, validation, entity resolution, retrieval strategy and build cost were not decided or measured:
the graph was not finished in the time we had. We chose to submit a complete, measured BM25 + fusion + evaluation instead of an
unmeasured graph. The decisions that remain open, for whoever continues: fixed or open entity/relation types, what to do with LLM output
that fails validation, how to merge aliases (q12: mailman → postroom), and how to turn a question into ranked chunks
(entity-neighbourhood expansion, path search, GraphRAG community summaries, LightRAG-style dual-level retrieval).

*Update (V4, ticket v3T01):* extraction, schema, validation, provenance and build cost are now done (D-20 to D-27).
Entity resolution (T2), the graph retriever (T3) and the UI (T4) are still open; until T3, selecting the graph returns 503.

**D-20. Schema: fixed types, plus two relations.** File `app/graph_schema.py` (versioned: `SCHEMA_VERSION`). Options: (a) an
**open** schema, where the LLM names whatever types it finds (as in GraphRAG's default prompt); (b) a **fixed** schema, where the LLM may
only use our lists. **Chosen: fixed.** Our corpus is small and known (5 documents, 43 chunks), so we can list its types in advance. A
fixed list can be checked by Pydantic, so an invented type is caught instead of stored. With an open schema the same thing comes back
under several names (`Team`, `Department`, `Squad`), and T2 would have to merge types as well as entities. Cost: anything outside the
lists is dropped, e.g. people, chat channels, prices (known failure 9).
Entity types: `Product, Feature, Plan, Service, ErrorCode, Policy, Procedure, Team`. Relation types: the PRD's
`has_feature, included_in, depends_on, causes, resolved_by, governed_by, owned_by, replaces`, **plus `raised_by`
(ErrorCode → Service) and `escalates_to` (ErrorCode → Team | Service)**. Why we added two: the chains of q08, q11 and q12 start at
an error code. With the PRD list, "KST-5031 is raised by postroom" and "escalate KST-6010 to Payments Core" fit no relation (the closest,
`causes`, points the wrong way: a service does not *cause* its error code, it reports it). Each relation declares its allowed
source and target types (`owned_by`: Service → Team; `replaces`: both ends of the same type), and Pydantic checks them.

**D-21. Extraction and validation.** One LLM call per chunk, temperature 0, same client and `.env` as answer generation
(`GRAPH_LLM_MODEL` can override the model for the build only). The prompt holds the schema, the chunk (marked as data, not
instructions) and a JSON shape. Pydantic models `Entity`, `Relation`, `ChunkExtraction` check: known entity and relation types;
each entity's name or an alias appears in the chunk; both ends of a relation are in the chunk's entity list; their types are allowed for
the relation; **the relation's quote is in the chunk**. This is a cheap check against invented triples. Quote matching ignores whitespace
and markdown emphasis (`*`, backticks) and nothing else, so `Owner: Messaging.` matches `Owner: **Messaging**.` but a paraphrase does
not. A `` ```json `` fence around the reply is accepted; any other text around the JSON is an error.
On failure: one retry, with the previous reply and the list of errors appended. A second failure means the chunk is skipped and
`{chunk_id, error, raw_reply}` is written to `graph_errors`. The skip is cached: at temperature 0 asking again would most likely
return the same reply, so a rerun does not pay for it again (`--retry-skipped` forces it). Transport errors (429, timeout, 5xx) are
*not* cached. They are retried with back-off and, if they persist, the chunk is reported as missing and the next run tries again.
A wrong key or model stops the LLM calls at once (every call would fail the same way); the graph is still rebuilt from what was done.
Options rejected: provider "JSON mode" or function calling (not available on every OpenAI-compatible endpoint we use, and it checks the
shape, not the content against the chunk); asking for several chunks per call (fewer calls, but provenance becomes the LLM's guess).

**D-22. Provenance and storage.** SQLite (`data/index/graph-kestrel.sqlite`, git-ignored, derived data like the BM25 and dense
indexes; option used by the PRD, chosen over JSON because T2/T3 need joins). Tables `nodes`, `aliases`, `edges`, `evidence`. Every
node and edge has at least one evidence row `{doc, chunk_id, quote}`, and a node or edge seen in several chunks keeps all of them. Node
evidence is the line of the chunk that mentions it; edge evidence is the LLM's quote, checked in D-21. In this step two mentions are
the same node only if type and name are equal, ignoring case and spacing. Real merges (aliases, `mailman` → `postroom`) are T2's job.
`meta` stores a fingerprint of the chunk list + schema version + prompt version (`core.records_fingerprint`). If a document, the
selection, the chunking, the schema or the prompt changes, `graph_status()` says `stale`. The graph tables are rebuilt from scratch at
every build (cheap: no LLM call, see the cache below), so they cannot drift from the documents. Same choice as D-04.
Cache: each chunk's validated extraction is stored under a hash of (chunk text, schema version, prompt version). It survives
rebuilds, so a changed document re-extracts only its own chunks, and a rerun after a crash or a rate limit pays only for what is missing.
The integrity check (`python -m app.graph check`, also run by every build) counts nodes and edges without evidence, edges whose
ends are missing, and evidence chunk ids that do not exist in the current chunking; all must be 0.

**D-23. Cost logging and rate limits.** Every call, including a rate-limited try, is a row in `llm_calls`: model, prompt and completion
tokens, latency, attempt, outcome (`ok`, `invalid` then `retried` or `skipped`, `rate_limited`, `error`). The SDK's own retries are
switched off so none is hidden. A build exports `data/results/graph/llm_calls.jsonl`, `graph_errors.jsonl` and `build-kestrel.json`
(committed; no key, no whole document). The *graph build cost* in the summary and in REPORT §7 is the sum of the calls behind the
current cached extractions, so it stays correct when a rerun costs 0. Rate limits: `GRAPH_CALL_DELAY_S` (default 1 s) between
calls; on 429 the build waits for `Retry-After` or 2, 4, 8… s (max 60 s, 6 tries). If one model or key runs out, change
`GRAPH_LLM_MODEL` / the key and rerun: the cache keeps finished chunks, and the call log records which model did which chunk.
Personal data: the extraction prompt sends chunk text to the LLM provider, as answer generation already does. The Kestrel team table
holds (fictional) names of engineering leads. `Person` is not in the schema, so they do not become nodes, but a table-row quote can
contain one.

**D-24. Extraction model and hidden reasoning.** Measured on the NVIDIA endpoint (`integrate.api.nvidia.com`) with one short
extraction request (a 3-line chunk, same JSON shape as the build), 17 November 2026:

| Model, setting | Result |
|---|---|
| `mistralai/mistral-large-4-1t-a52b` (our `LLM_MODEL`) | no reply after 2 minutes on a real chunk (stopped by hand) |
| `meta/llama-3.3-70b-instruct` | 410 Gone: retired by the provider |
| `z-ai/glm-5.3-flash` | timeout after 90 s |
| `nvidia/nemotron-3.5-lightning-30b-a3b`, 400-token budget | 3 s, **empty answer**: the whole budget went into hidden reasoning (`finish_reason: length`) |
| same, `/no_think` in the system prompt | ignored, same empty answer |
| same, 4096-token budget | 10.1 s, 1,543 completion tokens, correct JSON after 5,452 characters of hidden reasoning |
| **same, thinking switched off** (`chat_template_kwargs`) | **1.1 s, 95 completion tokens, correct JSON** |

**Chosen: `nemotron-3.5-lightning-30b-a3b` with thinking off** (`GRAPH_LLM_MODEL`, `GRAPH_DISABLE_THINKING=1`) for the build only;
answers keep `LLM_MODEL`. Extraction is copying facts that are written in the chunk into a fixed format, which needs no long reasoning,
and the build makes about 43 calls: thinking on would cost about 16 times the completion tokens and 9 times the time for the same JSON.
Our validation (D-21) is the safety net, not the model's reasoning: a wrong type or an invented quote is caught either way.
Risk: with thinking off, the model may make more mistakes on hard chunks (tables, several relations); this shows up as `retried` /
`skipped` in the call log and in the extraction-quality grading. An empty reply cut by the token limit is logged with an explicit message
so the cause is visible. The choice is recorded per call (`model`) and per build (`llm_options` in `build-kestrel.json`).

**D-25. Prompt v1 → v2, after the first real build.** Prompt v1 (rules only, no example) on the full Kestrel build with the D-24 model:
43 chunks, **3 valid at the first try, 18 after one retry, 22 skipped (51%)**; 82 calls, 107,990 prompt + 31,341 completion tokens,
218.6 s in LLM calls; 49 entities, 34 relations; integrity OK; rerun 0 calls. Lost chunks included the team table
(`service-catalog_md:4`), which q08, q11 and q12 need. First validation error per failed reply (61 replies):

| Error | Replies | Cause |
|---|---:|---|
| entity "Kestrel Analytics" not in the chunk | 23 | our prompt said "Kestrel Analytics documentation" and gave it as the `Product` example: the model copied it |
| another name not written in the chunk ("Business plan" for "Business") | 10 | model rewrites names |
| `causes` / `governed_by` backwards (`KST-1001 causes single sign-on`) | 8 | the runbook's "**Cause:**" lines; v1 never said which way a relation reads |
| relation end missing from the entity list | 4 | rule stated, no example |
| other disallowed types (`raised_by` → a Team) | 4 | model error, correctly rejected |
| unknown relation | 1 | model error |

Validation did what it is for (no invented entity or backwards edge entered the graph), but the price was half the corpus. Options:
(a) keep v1 and accept the loss; (b) **fix the prompt (v2)**; (c) keep the valid part of a reply that fails twice (drop only the bad
entities and relations) instead of skipping the chunk; (d) a bigger model. **Chosen: (b) first**, because most errors came from our prompt:
v2 removes every mention of the product name, says to copy names exactly and never add the product, company or file, explains
direction with four examples ("an error code is never the source of `causes`"), adds one *invented* worked example (a fictional
error code and service, so it cannot leak answers about the real corpus; a test checks it passes our own validation), and makes the retry
message say what to delete. (c) is kept in reserve: it changes the ticket's "retry once, then skip" rule, so we only consider it if v2
still loses chunks the chain questions need. Bumping `PROMPT_VERSION` changes every cache key, so v2 re-extracts all chunks
(v1's calls stay in `llm_calls.jsonl`; `build-kestrel-prompt-v1.json` keeps v1's summary).
Result of v2, and of v1 replayed with D-26 on its stored replies (no new call), same 43 chunks:

| | v1 strict | v2 strict | v1 + keep valid part | **v2 + keep valid part (built)** |
|---|---:|---:|---:|---:|
| valid at first try / after retry | 4 / 17 | 4 / 9 | 4 / 17 | **4 / 9** |
| kept in part / skipped | 0 / 22 | 0 / 30 | 21 / 1 | **29 / 1** |
| relations kept | 34 | n/a | 59 | **111** (112 before merging identical ones) |
| chain edges present (of 6 checked) | n/a | n/a | 4 | **5** |

Errors in failed replies (all of them, not only the first), v1 → v2: invented "Kestrel Analytics" 26 → 2; relation end missing from
the entities 17 → 26; other disallowed types 4 → 25; other name not in chunk 9 → 8; `causes` backwards 4 → 5; bad JSON 0 → 2.
So **v2 alone is worse under the strict rule** (30 chunks skipped instead of 22): it removed the invented product name, but the
example (three relations) made the model write *more* relations, so more replies contain at least one bad one. **Combined with D-26 it
is clearly better**: twice as many valid relations as v1 and one more chain edge. The direction examples did not fix `causes`.
We keep v2 + D-26. Total cost of the whole experiment (v1, v2, both partial runs): 164 calls, 243,380 prompt + 65,138 completion tokens,
504.6 s in LLM calls (`llm_calls.jsonl`).

**D-26. Keep the valid part of a reply that fails twice (changes the ticket's "then skip" rule).** With prompt v2, the first 13 chunks
(both billing policies) were all skipped. Their text (retries, grace periods, refunds) barely fits the schema, the model still
produces relations, gets one wrong, and gets one wrong again on the retry, so the chunk's *valid* entities and relations were thrown
away with the bad one. Options: (a) skip the whole chunk (ticket rule); (b) more retries (more cost, same behaviour at
temperature 0); (c) a third, stricter prompt; (d) **keep the valid part**. **Chosen: (d), with (a) still available** (`--strict`).
After the retry, each entity and relation of the second reply is checked on its own with exactly the same rules (`salvage()` reuses
the validator's checks); invalid ones are dropped, a relation whose end was dropped goes too, and the result is validated again in full,
so **nothing that fails validation is ever stored**: the guarantee is the same as in (a), only the unit of rejection is smaller.
The chunk counts as `partial`; every dropped item is listed in `graph_errors` (kind `partial`) and the call log says `partial`. A
reply that is not JSON, or with no valid entity left, is still `skipped`. Chunks skipped by an earlier build are recovered from their
stored second reply **without an LLM call**. Risk: a reply that was wrong in one place may be wrong elsewhere in ways the rules cannot
see (a true quote read the wrong way, known failure 11); the extraction-quality grading must sample partial chunks too.
Measured result: D-25's table (skipped chunks 30 → 1, relations kept 111). This changes the v3T01 acceptance criterion "bad output → one retry, then a logged skip"; under `--strict` it holds exactly
(`tests/test_graph.py` checks both modes). To agree with the team before merging.

**D-27. A quote must name one end of its relation (validation v2).** Checking the plans table (ticket N.B. "check that table rows
produce triples") showed that it gave 57 edges whose quotes were single cells: `Business has_feature SCIM user provisioning`, quote
"Yes". "Yes" is in the chunk, so D-21's check passed, but it comes from another cell, and the edge is **false** (SCIM is Enterprise only).
A one-word quote is in every table, so it defeats the "quote is in the chunk" check that is meant to catch invented triples. Options:
(a) a minimum quote length (arbitrary: "Owner: **Identity**." is short and valid); (b) **the quote must contain the name or an alias of the
source or the target**; (c) both ends (too strict: "This error is raised by postroom" does not repeat the error code, which is in the
heading). **Chosen: (b).** A table row like `| SCIM user provisioning | — | — | — | — | Yes |` still passes, a bare "Yes" does not.
The rules have a version (`VALIDATION_VERSION`, part of the graph fingerprint, not of the cache key): after a change, the next build
re-checks every cached extraction with the new rules and drops what no longer passes, **without any LLM call**; under `--strict` such a
chunk is skipped. Result on our graph (0 calls): 62 relations dropped in 7 chunks, all for this reason: 57 from the plans table (all
of them, the true ones with the false ones), plus 5 elsewhere, every one unsupported (`connector-hub has_feature e sign-on (SAML)`,
`KST-1001 escalates_to on-call`, a "Kestrel Analytics" edge…). Relations 111 → 49; the 5 chain edges of q08, q11, q12 are unchanged.
Cost: the plans table now gives no edge (known failure 15). The prompt does not state the new rule yet (that would be prompt v3 and
re-extract all 43 chunks); the retry message explains it when it fires.

## Process

**D-17. Hidden instructions.** The V2 materials contained an instruction invisible on the slide (text in an unreadable colour) that
asked for a hidden page; an assistant that read the raw text added it. We removed that route and do not act on instructions that
are not visible on the slide, unless confirmed by the teacher.

## Known failures (V3 so far; extend after the runs)

1. BM25 has no signal when the question's words are absent from the corpus (vocabulary questions, e.g. "calcium deposits" vs
   "limescale"): it returns nothing and the fusion falls back to dense alone. Fusion cannot fix a question neither retriever finds.
2. Stop-word list is English only; there is no stemming; no incremental BM25 update (D-04, D-05).
3. BM25 and RRF do not know about document versions: for `versions` questions the 2025 and 2026 policies compete (V2 failure
   KST-07 persists as a generation problem, not a retrieval one).
4. Gold chunks are found by quote matching: a quote cut by a chunk boundary yields no gold chunk (flagged, not hidden).
5. q12's hops 1 and 2 share a chunk (D-10).
6. Rank-only fusion lets a chunk found by two retrievers outrank a correct chunk found by one, even when its second vote comes from deep
   in a list: Kestrel q05 (dense #1, BM25 absent) fell from rank 1 to 6 with 30 candidates; limiting to 10 candidates brings it back to
   rank 2 (D-18), still not rank 1. `split_answer` Recall@3 stays at 0.38 with BM25 on (q03 loses a hop to chunks BM25 promotes).
7. `two_hop` Recall stays at 0.56 at k = 5 and 0.72 at k = 10 with dense, with or without BM25: finding every hop of a chain needs
   something that follows relations between chunks (the missing graph).
8. The concept graph is not queried yet: extraction is done (D-20 to D-23); resolution, retrieval and UI are not (D-12 update).
9. The fixed schema drops what fits no type: people, chat channels (`#oncall-messaging`), prices, IP addresses. A chain that ends at a
   channel stops at the `Team` node; the channel is only in that node's evidence chunk (the team table row).
10. Cross-chunk relations are not extracted. Extraction sees one chunk at a time, so a relation whose two ends sit in different
    chunks is missed: e.g. "escalate to the team that owns postroom" gives `KST-5031 escalates_to postroom`, not `escalates_to Messaging`.
    T2 is expected to join such pieces through shared nodes (postroom); what it cannot join will be listed here.
11. Quote checking catches invented quotes, not wrong readings of a true quote: a triple whose quote is in the chunk but whose direction
    or relation is wrong passes validation (to be measured by the 10-triple extraction-quality grading).
    **Seen in the built graph: `postroom owned_by Identity`** (quote "Owner: **Identity**.", `service-catalog_md:1`). That chunk ends
    with the heading "### postroom"; the "Owner" line above it belongs to `auth`. The graph now gives postroom two owners, Identity
    (wrong) and Messaging (right), so a graph path for q08 / q12 can lead support to `#oncall-identity`. A service has one owner: T2 can
    flag services with several `owned_by` edges.
12. Wrong entity types pass when the name is in the chunk: the "Team" nodes include `Jonas Berg` (a person), `#oncall-payments` (a
    channel), roles (`Owner`, `Admin`, `Editor`) and the plan `Team`. The schema check sees the type name, not its meaning.
13. The team table is only partly extracted: `service-catalog_md:3` gave the Team nodes Payments Core, Identity and Platform, not
    Messaging, Data Integrations or Analytics Engine, although their rows are in the chunk. The `Messaging` node's only evidence is
    `service-catalog_md:2` ("Owner: **Messaging**."), so the graph reaches Messaging but not the row with `#oncall-messaging`, which is
    q08's and q12's last hop. q11's last hop is present (Payments Core has evidence in the table chunks 3 and 4).
14. A section cut by a chunk boundary loses its subject: `### ledger` is in `service-catalog_md:2` and its "Owner: **Payments Core**."
    line in `:3`, so `ledger owned_by Payments Core` is missing and the model invented a `Payments Core` *Service* owned by the team.
15. Tables give few or no triples. Plans table: its 57 edges quoted single cells ("Yes", "1") and were all dropped by D-27, so the graph
    does not know which plan has which feature. Team table: rows give Team nodes only (no relation in the schema links a team to its
    channel) and the Messaging row is missing (13). A deterministic parser of markdown tables (one edge or node per row, the row as quote,
    no LLM) would fix both; not done in this ticket.
