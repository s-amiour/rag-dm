# V3 evaluation summary

_Generated 2026-11-18 02:12:23. Chunking per corpus: kestrel 800/10, own 500/20. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. not_in_corpus questions are excluded from both._

## Configurations in this folder

| Corpus | Configuration | Run at | Chunking | Top-k | Candidates | RRF k | Dense backend |
|---|---|---|---|---:|---:|---:|---|
| kestrel | dense | 2026-11-18 02:12:21 | 800/10 | 10 | 10 | 60 | sentence-transformers |
| kestrel | dense+bm25 | 2026-11-18 02:12:22 | 800/10 | 10 | 10 | 60 | sentence-transformers |
| kestrel | dense+bm25+graph | 2026-11-18 02:12:22 | 800/10 | 10 | 10 | 60 | sentence-transformers |
| own | dense | 2026-11-18 02:12:23 | 500/20 | 10 | 10 | 60 | sentence-transformers |
| own | dense+bm25 | 2026-11-18 02:12:23 | 500/20 | 10 | 10 | 60 | sentence-transformers |

## Results: configurations × corpora × question types

| Corpus | Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kestrel | dense | **ALL** | 10 | 0.50 | 0.80 | 0.90 | 1.00 | 0.33 | 0.72 | 0.77 | 0.92 |
| kestrel | dense | exact_identifier | 2 | 0.50 | 0.50 | 0.50 | 1.00 | 0.25 | 0.50 | 0.50 | 1.00 |
| kestrel | dense | split_answer | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.62 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | versions | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense | two_hop | 3 | 0.33 | 0.67 | 1.00 | 1.00 | 0.17 | 0.39 | 0.56 | 0.72 |
| kestrel | dense+bm25 | **ALL** | 10 | 0.50 | 0.90 | 1.00 | 1.00 | 0.32 | 0.64 | 0.79 | 0.92 |
| kestrel | dense+bm25 | exact_identifier | 2 | 0.50 | 0.50 | 1.00 | 1.00 | 0.25 | 0.50 | 1.00 | 1.00 |
| kestrel | dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| kestrel | dense+bm25 | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25 | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25 | two_hop | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.39 | 0.56 | 0.56 | 0.72 |
| kestrel | dense+bm25+graph | **ALL** | 10 | 0.70 | 1.00 | 1.00 | 1.00 | 0.47 | 0.79 | 0.84 | 0.92 |
| kestrel | dense+bm25+graph | exact_identifier | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| kestrel | dense+bm25+graph | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | two_hop | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 | 0.72 | 0.72 | 0.72 |
| own | dense | **ALL** | 13 | 0.62 | 0.85 | 0.92 | 0.92 | 0.62 | 0.85 | 0.92 | 0.92 |
| own | dense | exact_identifier | 5 | 0.60 | 0.80 | 1.00 | 1.00 | 0.60 | 0.80 | 1.00 | 1.00 |
| own | dense | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| own | dense | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | **ALL** | 13 | 0.69 | 0.85 | 0.92 | 0.92 | 0.69 | 0.85 | 0.92 | 0.92 |
| own | dense+bm25 | exact_identifier | 5 | 0.80 | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | 1.00 | 1.00 |
| own | dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| own | dense+bm25 | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost per question and index build

| Corpus | Configuration | Retrieval ms/question | Total ms/question | LLM calls/question | Prompt tok/question | Completion tok/question | Dense backend | Index build |
|---|---|---:|---:|---:|---:|---:|---|---|
| kestrel | dense | 35.8 | 35.8 | 0.00 | 0 | 0 | sentence-transformers | dense: 21.99s (cached) |
| kestrel | dense+bm25 | 27.6 | 27.6 | 0.00 | 0 | 0 | sentence-transformers | dense: 21.99s (cached); bm25: 0.00s (cached) |
| kestrel | dense+bm25+graph | 33.4 | 33.4 | 0.00 | 0 | 0 | sentence-transformers | dense: 21.99s (cached); bm25: 0.00s (cached) |
| own | dense | 29.0 | 29.0 | 0.00 | 0 | 0 | sentence-transformers | dense: 0.02s (cached) |
| own | dense+bm25 | 19.2 | 19.2 | 0.00 | 0 | 0 | sentence-transformers | dense: 0.02s (cached); bm25: 0.00s (cached) |

## Where adding a retriever made things worse

**kestrel: `dense` → `dense+bm25`**: 2 question(s) where the first gold chunk ranks worse.
- `q03` (split_answer): rank 1 → 2 (recall@k 1.00 → 1.00) — What happens, step by step, when a customer's card payment fails?
- `q05` (vocabulary): rank 1 → 2 (recall@k 1.00 → 1.00) — I lost my phone and I can't log in anymore. What can I do?

**kestrel: `dense+bm25` → `dense+bm25+graph`**: 0 question(s) where the first gold chunk ranks worse.

**kestrel: `dense` → `dense+bm25+graph`**: 2 question(s) where the first gold chunk ranks worse.
- `q03` (split_answer): rank 1 → 2 (recall@k 1.00 → 1.00) — What happens, step by step, when a customer's card payment fails?
- `q05` (vocabulary): rank 1 → 2 (recall@k 1.00 → 1.00) — I lost my phone and I can't log in anymore. What can I do?

**own: `dense` → `dense+bm25`**: 2 question(s) where the first gold chunk ranks worse.
- `OWN-01` (exact_identifier): rank 4 → 5 (recall@k 1.00 → 1.00) — What is the capacity of the Vellora Duo water tank?
- `OWN-02` (split_answer): rank 2 → 3 (recall@k 1.00 → 1.00) — How long should the water filter be soaked before installation, and how long does it last?
