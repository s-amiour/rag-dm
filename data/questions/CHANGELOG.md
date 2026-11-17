# Question set changelog

Rule: questions are never edited silently. Every change to an existing row is listed here, and the row keeps its old
value in `type_original`. Question ids and texts are never changed; new questions are appended at the end.

## 2026-11-17 · ticket v3T13 · `own.jsonl`

Type labels changed to the assignment's names, so the per-type tables have one row per type
(before, `different_vocabulary`, `two_hops` and `not_in_corpus_trap` showed up as extra types):

| Id | Old type | New type | Why |
|---|---|---|---|
| OWN-03 | `different_vocabulary` | `vocabulary` | Same meaning, assignment's name. |
| OWN-07 | `different_vocabulary` | `vocabulary` | Same meaning, assignment's name. |
| OWN-11 | `different_vocabulary` | `vocabulary` | Same meaning, assignment's name. |
| OWN-04 | `not_in_corpus_trap` | `other` | The question *sounds* unanswerable but the manual answers it ("Never use vinegar…"), so it is answerable and must not count as `not_in_corpus`. None of the six types fits, so `other`. |
| OWN-05 | `two_hops` | `other` | It has one evidence quote, in one chunk: it is not a two-hop question, and counting it as one inflated the own corpus's two_hop scores (it was the only one, with Recall 1.00). |

Added:

| Id | Type | Why |
|---|---|---|
| OWN-13 | `not_in_corpus` | The own set had no unanswerable question, so `not_in_corpus` was never measured on it. The manual never gives the machine's weight, but it gives tempting numbers (tank capacity, ventilation space). |

Effect: the own corpus now has no `two_hop` question; its per-type numbers change, so the V3 matrix must be rerun
(rerun gate, ticket v3T05). OWN-13 is excluded from Hit@k / Recall@k like every unanswerable question; it still needs a
no-context run and a hand-graded answer (correct = "not in the documents").