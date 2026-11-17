"""Grading a random sample of graph triples (ticket v3T06).

    python -m app.graph sample --n 10 --seed 42     # draw the sample, write the file to grade
    python -m app.graph chunk support-runbook-errors_md:8
    python -m app.graph grades                      # check the grades, render the table with the counts

A person grades; this module only draws the sample, shows the evidence and checks
the result against data/results/v3/graph_triples_grading_rules.md.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from . import core

GRADES = ("correct", "wrong", "unsupported")
OUT_DIR = core.DATA / "results" / "v3"
OUT_NAME = "graph_triples_graded"
EXCERPT_CHARS = 220


def graph_hash(con) -> str:
    """Hash of what is graded: every edge (names, types, relation) with its evidence and the
    aliases of its ends. Same graph -> same hash; any rebuild that changes an edge -> new hash."""
    h = hashlib.sha256()
    aliases: dict[int, list[str]] = {}
    for node, alias in con.execute("SELECT node_id, alias FROM aliases ORDER BY alias"):
        aliases.setdefault(node, []).append(alias)
    rows = []
    for eid, s, st, rel, t, tt, sid, tid in con.execute(
            "SELECT e.id, n1.name, n1.type, e.relation, n2.name, n2.type, n1.id, n2.id FROM edges e "
            "JOIN nodes n1 ON n1.id = e.source_id JOIN nodes n2 ON n2.id = e.target_id"):
        ev = sorted(tuple(r) for r in
                    con.execute("SELECT chunk_id, quote FROM evidence WHERE kind='edge' AND owner_id=?", (eid,)))
        rows.append(json.dumps([s, st, rel, t, tt, ev, sorted(set(aliases.get(sid, []))),
                                sorted(set(aliases.get(tid, [])))], ensure_ascii=False))
    for row in sorted(rows):
        h.update(row.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _excerpt(chunk_text: str, quote: str) -> str:
    """The quote with some of the chunk around it, on one line."""
    i = chunk_text.find(quote)
    if i < 0:
        i = chunk_text.find(quote.replace("**", ""))
    if i < 0:
        start, end = 0, min(len(chunk_text), 2 * EXCERPT_CHARS)
    else:
        start, end = max(0, i - EXCERPT_CHARS), min(len(chunk_text), i + len(quote) + EXCERPT_CHARS)
    text = chunk_text[start:end].replace("\r", "").replace("\n", " / ")
    return ("..." if start > 0 else "") + text + ("..." if end < len(chunk_text) else "")


def sample(con, records: list[dict[str, Any]], n: int = 10, seed: int = 42) -> dict[str, Any]:
    """n edges drawn with random.Random(seed) from all edges sorted by id: the same graph and
    the same seed always give the same triples. Every edge is a stated triple (the graph has no
    inferred edges)."""
    meta = {row[0]: row[1] for row in con.execute("SELECT key, value FROM meta")}
    texts = {r["id"]: r["text"] for r in records}
    aliases: dict[int, list[str]] = {}
    for node, alias in con.execute("SELECT node_id, alias FROM aliases ORDER BY alias"):
        if alias not in aliases.setdefault(node, []):
            aliases[node].append(alias)

    edges = con.execute(
        "SELECT e.id, n1.id, n1.name, n1.type, e.relation, n2.id, n2.name, n2.type FROM edges e "
        "JOIN nodes n1 ON n1.id = e.source_id JOIN nodes n2 ON n2.id = e.target_id ORDER BY e.id").fetchall()
    picked = random.Random(seed).sample(list(edges), min(n, len(edges)))

    items = []
    for i, (eid, sid, s, st, rel, tid, t, tt) in enumerate(picked, start=1):
        evidence = [
            {"chunk_id": chunk, "document": doc, "quote": quote, "excerpt": _excerpt(texts.get(chunk, ""), quote)}
            for doc, chunk, quote in con.execute(
                "SELECT doc, chunk_id, quote FROM evidence WHERE kind='edge' AND owner_id=? ORDER BY chunk_id", (eid,))
        ]
        items.append({
            "n": i, "edge_id": eid,
            "source": s, "source_type": st, "source_aliases": aliases.get(sid, []),
            "relation": rel,
            "target": t, "target_type": tt, "target_aliases": aliases.get(tid, []),
            "evidence": evidence,
            "grade": "", "graded_chunk": "", "note": "", "review_note": "",
        })

    return {
        "meta": {
            "seed": seed, "n": len(items), "edges_in_graph": len(edges),
            "graph_hash": graph_hash(con), "graph_fingerprint": meta.get("fingerprint", ""),
            "build_id": meta.get("build_id", ""), "rules": meta.get("rules", ""),
            "chunk_size": meta.get("chunk_size", ""), "overlap": meta.get("overlap", ""),
            "grading_rules": "data/results/v3/graph_triples_grading_rules.md",
            "graded_by": "", "reviewed_by": "",
        },
        "triples": items,
    }


def out_paths(out_dir: Path | None = None) -> tuple[Path, Path]:
    folder = out_dir or OUT_DIR
    return folder / f"{OUT_NAME}.json", folder / f"{OUT_NAME}.md"


def already_graded(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return True  # unreadable: do not overwrite what may be someone's work
    return any(t.get("grade") or t.get("note") for t in data.get("triples", []))


def write_sample(data: dict[str, Any], out_dir: Path | None = None, force: bool = False) -> Path:
    json_path, _ = out_paths(out_dir)
    if already_graded(json_path) and not force:
        raise FileExistsError(f"{json_path} already has grades; use --force to replace it with a new sample")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return json_path


def check(data: dict[str, Any], current_hash: str | None = None) -> dict[str, Any]:
    """Problems block the result (an unknown grade, a grade without note or chunk);
    warnings do not (ungraded triples, no reviewer yet, a graph rebuilt since sampling)."""
    problems: list[str] = []
    warnings: list[str] = []
    counts = {g: 0 for g in GRADES}
    ungraded = 0

    for t in data.get("triples", []):
        label = f"#{t.get('n')} ({t.get('source')} -{t.get('relation')}-> {t.get('target')})"
        grade = (t.get("grade") or "").strip()
        if not grade:
            ungraded += 1
            continue
        if grade not in GRADES:
            problems.append(f"{label}: grade {grade!r} is not one of {', '.join(GRADES)}")
            continue
        counts[grade] += 1
        if not (t.get("note") or "").strip():
            problems.append(f"{label}: a graded triple needs a one-line note")
        chunks = [e["chunk_id"] for e in t.get("evidence", [])]
        if t.get("graded_chunk") not in chunks:
            problems.append(f"{label}: graded_chunk must be one of its evidence chunks {chunks}")

    meta = data.get("meta", {})
    if ungraded:
        warnings.append(f"{ungraded} triple(s) not graded yet")
    if not meta.get("graded_by"):
        warnings.append("meta.graded_by is empty")
    if not meta.get("reviewed_by"):
        warnings.append("meta.reviewed_by is empty: a second person reviews every grade")
    if current_hash and current_hash != meta.get("graph_hash"):
        warnings.append("the graph changed since this sample was drawn: these grades describe an older graph "
                        "(re-sample with the same seed and grade again, or say which graph was graded)")
    return {"counts": counts, "graded": sum(counts.values()), "ungraded": ungraded,
            "problems": problems, "warnings": warnings}


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(data: dict[str, Any], result: dict[str, Any]) -> str:
    m, c = data["meta"], result["counts"]
    lines = [
        "# Extraction quality: 10 graph triples graded by hand",
        "",
        f"**Counts: correct {c['correct']} / wrong {c['wrong']} / unsupported {c['unsupported']}**"
        + (f" ({result['ungraded']} not graded yet)" if result["ungraded"] else ""),
        "",
        f"Sample: `python -m app.graph sample --n {m['n']} --seed {m['seed']}`, {m['n']} of {m['edges_in_graph']} edges of the "
        f"resolved graph (rules: {m.get('rules') or '?'}; chunks {m.get('chunk_size')}/{m.get('overlap')}). "
        f"Graph hash `{m['graph_hash'][:16]}`, fingerprint `{m.get('graph_fingerprint', '')[:16]}`, build `{m.get('build_id')}`.",
        f"Rules: `{m['grading_rules']}`. Graded by: {m.get('graded_by') or '?'}. Reviewed by: {m.get('reviewed_by') or '?'}.",
        "Ten triples are a sanity check, not an accuracy measure: one triple moves a rate by 10 points.",
        "",
    ]
    for w in result["warnings"]:
        lines.append(f"> Warning: {w}")
    if result["warnings"]:
        lines.append("")
    lines += ["| # | Triple | Evidence chunk(s) | Quote | Grade | Graded on | Note | Review |",
              "|--:|---|---|---|---|---|---|---|"]
    for t in data["triples"]:
        triple = f"{t['source']} ({t['source_type']}) -**{t['relation']}**-> {t['target']} ({t['target_type']})"
        chunks = ", ".join(f"`{e['chunk_id']}`" for e in t["evidence"])
        quotes = " / ".join(f"\"{e['quote']}\"" for e in t["evidence"])
        lines.append(f"| {t['n']} | {_cell(triple)} | {chunks} | {_cell(quotes)} | {t.get('grade') or ''} | "
                     f"{t.get('graded_chunk') or ''} | {_cell(t.get('note') or '')} | {_cell(t.get('review_note') or '')} |")
    lines += ["", "## Chunk excerpts", ""]
    for t in data["triples"]:
        lines.append(f"**{t['n']}. {t['source']} -{t['relation']}-> {t['target']}**"
                     + (f" (aliases: {', '.join(t['source_aliases'] + t['target_aliases'])})"
                        if t["source_aliases"] or t["target_aliases"] else ""))
        for e in t["evidence"]:
            lines.append(f"- `{e['chunk_id']}`: {e['excerpt']}")
        lines.append("")
    return "\n".join(lines)


def load(out_dir: Path | None = None) -> dict[str, Any]:
    json_path, _ = out_paths(out_dir)
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} does not exist: python -m app.graph sample --n 10 --seed 42")
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_render(data: dict[str, Any], result: dict[str, Any], out_dir: Path | None = None) -> Path:
    _, md_path = out_paths(out_dir)
    md_path.write_text(render(data, result) + "\n", encoding="utf-8")
    return md_path
