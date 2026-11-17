# Grading rules: graph triples (ticket v3T06)

Written and committed **before** any triple is graded. The sample is drawn by
`python -m app.graph sample --n 10 --seed 42` from the resolved graph (after entity resolution) and saved in
`graph_triples_graded.json`; `python -m app.graph grades` checks the grades and renders `graph_triples_graded.md`.

## The three grades

| Grade | Rule |
|---|---|
| **correct** | The chunk states this relation between these two entities, in this direction. |
| **wrong** | The chunk says something else: wrong entity, wrong relation, or reversed direction. |
| **unsupported** | Nothing in the chunk says it: the LLM inferred or invented it. |

No other grade, no "partly correct". When unsure between two grades, pick the worse one and say why in the note.

## How to grade one triple

1. Read the relation's definition in `app/graph_schema.py` (for example `escalates_to`: "support escalates this error to this team,
   or to the team that owns this service"). Grade against that meaning, not against the everyday meaning of the word.
2. Open each evidence chunk in full: `python -m app.graph chunk <chunk id>`. The quote alone is not enough.
3. Ask: does **this chunk** say that *source* is related to *target* by *relation*, in this direction?
4. Write the grade, the chunk you graded on, and a one-line note.

## Edge cases (decided in advance)

- **Only the evidence chunks count.** If the triple is true elsewhere in the corpus but not in its evidence chunks, it is
  **unsupported**: the graph claims a source it does not have. Outside knowledge never makes a triple correct.
- **Several evidence chunks:** grade against each; the triple is **correct** if at least one chunk supports it. Record that
  chunk in `graded_chunk`. If none supports it, record the closest one.
- **Merged names and aliases:** if the chunk uses an alias of the node (e.g. "the Messaging team" for Messaging, "Platform team" for
  Platform), the alias counts as the entity. The aliases of both ends are listed with each triple.
- **Quote present but relation wrong:** extraction already checked that the quote is in the chunk and names one end; it does not
  check that the quote supports the relation. A real quote with the wrong relation, entity or direction is **wrong**, not correct.
- **Entity identity, not type label:** the type is not graded. "Analytics Engine" typed `Service` while the chunk means the team is
  still the Analytics Engine: grade the relation, and mention the type in the note. But an end that is a different thing altogether
  (a person, a chat channel, a role or a sentence typed as an entity, e.g. `Jonas Berg` as a Team) is a **wrong entity**: **wrong**.
- **Conditions:** a relation stated with a condition ("escalate to Payments Core only if more than three customers are affected")
  is **correct**; put the condition in the note.
- **Too vague to be an entity** (a whole sentence, "on-call", "service"): if the chunk does say it, **wrong** (wrong entity);
  otherwise **unsupported**.
- **Table rows:** a row supports a relation between the row's subject and a column's value only if the column header makes the
  relation clear. Reading a cell from the wrong row or column is **wrong**.

## Review

A second person reviews every grade with these rules. A disagreement is written in `review_note` (both grades and why), settled by
discussion, and the final grade is the one in `grade`. Who graded and who reviewed is recorded at the top of the file.

## Why only 10

Ten triples are a sanity check, not an accuracy measure: one triple moves a rate by 10 points. Report counts (e.g. 7 / 2 / 1),
not percentages. If the graph is rebuilt after grading (prompt, schema, validation or merges), the grades describe a graph that no
longer exists: `grades` warns when the graph's hash differs from the sampled one; re-sample with the same seed and grade again.
