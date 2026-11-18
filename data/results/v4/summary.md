# V3 evaluation summary

_Generated 2026-11-18 11:15:53. Chunking per corpus: kestrel 800/10, own 500/20. Hit@k = at least one hop retrieved; Recall@k = share of hops retrieved. not_in_corpus questions are excluded from both._

## Configurations in this folder

| Corpus | Configuration | Run at | Chunking | Top-k | Candidates | RRF k | Dense backend | Fingerprint |
|---|---|---|---|---:|---:|---:|---|---|
| kestrel | dense+bm25+graph | 2026-11-18 11:13:52 | 800/10 | 10 | 10 | 60 | sentence-transformers | d535b5 |
| kestrel | dense+bm25+graph (rerank) | 2026-11-18 11:14:52 | 800/10 | 10 | 30 | 60 | sentence-transformers | 1b5b7a |
| own | dense+bm25 | 2026-11-18 11:15:13 | 500/20 | 10 | 10 | 60 | sentence-transformers | d59318 |
| own | dense+bm25 (rerank) | 2026-11-18 11:15:52 | 500/20 | 10 | 30 | 60 | sentence-transformers | 902d4e |

## Results: configurations × corpora × question types

| Corpus | Configuration | Type | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kestrel | dense+bm25+graph | **ALL** | 10 | 0.70 | 1.00 | 1.00 | 1.00 | 0.47 | 0.79 | 0.84 | 0.92 |
| kestrel | dense+bm25+graph | exact_identifier | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 0.38 | 0.62 | 1.00 |
| kestrel | dense+bm25+graph | vocabulary | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | versions | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph | two_hop | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 | 0.72 | 0.72 | 0.72 |
| kestrel | dense+bm25+graph (rerank) | **ALL** | 10 | 0.70 | 1.00 | 1.00 | 1.00 | 0.47 | 0.87 | 0.87 | 0.92 |
| kestrel | dense+bm25+graph (rerank) | exact_identifier | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph (rerank) | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.25 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph (rerank) | vocabulary | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph (rerank) | versions | 2 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 | 1.00 | 1.00 |
| kestrel | dense+bm25+graph (rerank) | two_hop | 3 | 1.00 | 1.00 | 1.00 | 1.00 | 0.56 | 0.56 | 0.56 | 0.72 |
| own | dense+bm25 | **ALL** | 13 | 0.69 | 0.85 | 0.92 | 0.92 | 0.69 | 0.85 | 0.92 | 0.92 |
| own | dense+bm25 | exact_identifier | 5 | 0.80 | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | 1.00 | 1.00 |
| own | dense+bm25 | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 | two_hop | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| own | dense+bm25 | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 (rerank) | **ALL** | 13 | 0.77 | 1.00 | 1.00 | 1.00 | 0.77 | 0.96 | 0.96 | 0.96 |
| own | dense+bm25 (rerank) | exact_identifier | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 (rerank) | split_answer | 2 | 0.50 | 1.00 | 1.00 | 1.00 | 0.50 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 (rerank) | vocabulary | 3 | 0.67 | 1.00 | 1.00 | 1.00 | 0.67 | 1.00 | 1.00 | 1.00 |
| own | dense+bm25 (rerank) | two_hop | 1 | 0.00 | 1.00 | 1.00 | 1.00 | 0.00 | 0.50 | 0.50 | 0.50 |
| own | dense+bm25 (rerank) | other | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Cost per question and index build

| Corpus | Configuration | Retrieval ms/question | Total ms/question | LLM calls/question | Prompt tok/question | Completion tok/question | Dense backend | Index build |
|---|---|---:|---:|---:|---:|---:|---|---|
| kestrel | dense+bm25+graph | 37.2 | 37.2 | 0.00 | 0 | 0 | sentence-transformers | dense: 19.94s (cached); bm25: 0.00s (cached) |
| kestrel | dense+bm25+graph (rerank) | 2318.3 | 2318.3 | 0.00 | 0 | 0 | sentence-transformers | dense: 19.59s (cached); bm25: 0.00s (cached); rerank: 0.00s |
| own | dense+bm25 | 16.1 | 16.1 | 0.00 | 0 | 0 | sentence-transformers | dense: 16.09s (cached); bm25: 0.00s (cached) |
| own | dense+bm25 (rerank) | 1114.1 | 1114.1 | 0.00 | 0 | 0 | sentence-transformers | dense: 16.96s (cached); bm25: 0.00s (cached); rerank: 0.00s |

## Where adding a retriever made things worse (top 5, the chunks the LLM receives)

(no comparable pairs were run)
