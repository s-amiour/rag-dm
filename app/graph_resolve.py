"""Entity resolution for the concept graph (ticket v3T02).

T1 gives one node per (type, name) as written. Here we decide which of those
nodes are the same thing, with simple rules that can be explained and undone.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from .bm25 import fold

RULES = ("identifier", "normalised", "alias")  # applied in this order

ARTICLES = {"the", "a", "an"}
GENERIC_SUFFIXES = {"team", "teams", "service", "services"}

PLAN_NAMES = {"free", "starter", "team", "business", "enterprise"}  # product guide, section 2

_ERROR_CODE = re.compile(r"\bKST-\d{4}\b", re.IGNORECASE)
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
_BRACKETS = re.compile(r"\([^)]*\)")
_PUNCT = re.compile(r"[^\w\s-]")


def normalise_name(name: str) -> str:
    """'the Messaging team' -> 'messaging'; 'Inès Moreau' -> 'ines moreau'.

    Lowercase and fold accents, drop text in brackets ("Messaging (owner of
    postroom)"), drop punctuation (hyphens inside words are kept: connector-hub),
    then leading articles and generic suffixes. A word is never stripped when it
    is the only one left: the plan "Team" stays "team" instead of becoming "".
    """
    text = _PUNCT.sub(" ", _BRACKETS.sub(" ", fold(name)))
    words = [w.strip("-") for w in text.split()]
    words = [w for w in words if w]

    while len(words) > 1 and words[0] in ARTICLES:
        words.pop(0)
    while len(words) > 1 and words[-1] in GENERIC_SUFFIXES:
        words.pop()

    return " ".join(words)


def identifiers(name: str, entity_type: str) -> frozenset[str]:
    """Identifiers in a name. They match exactly or not at all.

    Error codes (KST-5031) and IP addresses anywhere in the name; for a Plan, the
    plan name itself, with or without "plan(s)": "Enterprise", "Enterprise plan"
    and "Enterprise plans" all give plan:enterprise, "Annual plans" gives none.
    """
    ids = {code.upper() for code in _ERROR_CODE.findall(name)} | set(_IP.findall(name))

    if entity_type == "Plan":
        words = normalise_name(name).split()
        if len(words) > 1 and words[-1] in ("plan", "plans"):
            words = words[:-1]
        if len(words) == 1 and words[0] in PLAN_NAMES:
            ids.add(f"plan:{words[0]}")

    return frozenset(ids)


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

Ref = tuple[str, str]  # a T1 node: (type, key) where key = graph.norm_key(name)


@dataclass
class Surface:
    """One T1 node: every way it was written, and the chunks it comes from."""
    type: str
    key: str
    names: dict[str, int] = field(default_factory=dict)   # surface name -> mentions
    chunks: set[str] = field(default_factory=set)

    @property
    def ref(self) -> Ref:
        return (self.type, self.key)


@dataclass
class Resolution:
    root: dict[Ref, Ref]                 # every T1 node -> the representative of its cluster
    clusters: dict[Ref, list[Ref]]       # representative -> members (representative first)
    merges: list[dict]                   # one per union: method, into, absorbed, chunks
    refused: list[dict]                  # unions a rule wanted but a constraint blocked

    def canonical_name(self, rep: Ref, surfaces: dict[Ref, Surface]) -> str:
        """The name most chunks use; ties: the shortest, then alphabetical."""
        count: dict[str, set[str]] = defaultdict(set)
        for ref in self.clusters[rep]:
            for name in surfaces[ref].names:
                count[name] |= surfaces[ref].chunks
        return min(count, key=lambda n: (-len(count[n]), len(n), n))


def resolve(surfaces: dict[Ref, Surface], alias_links: list[tuple[Ref, str, str]] = (),
            rules: tuple[str, ...] = RULES, never_merge: set[frozenset[Ref]] = frozenset()) -> Resolution:
    """Cluster T1 nodes with the rules switched on, in RULES order.

    alias_links: (node, alias name, chunk id) for every alias the extraction listed.
    never_merge: pairs of nodes that must stay apart, e.g. both ends of a `replaces`
    edge: "postroom replaced mailman" is history between two entities, not a second
    name (q12 tests it), even if the LLM also listed mailman as an alias of postroom.
    Constraints that always hold, whatever the rules: never across types; never two
    clusters whose identifiers differ (KST-5031 / KST-5038); never_merge.
    """
    unknown = set(rules) - set(RULES)
    if unknown:
        raise ValueError(f"unknown rule(s) {sorted(unknown)}; choose from {list(RULES)}")

    parent = {ref: ref for ref in surfaces}
    members = {ref: [ref] for ref in surfaces}
    ids = {ref: set().union(*(identifiers(n, ref[0]) for n in s.names)) if s.names else set()
           for ref, s in surfaces.items()}
    merges: list[dict] = []
    refused: list[dict] = []

    def find(ref: Ref) -> Ref:
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def blocked(a: Ref, b: Ref) -> str | None:
        if a[0] != b[0]:
            return "different types"
        if ids[a] and ids[b] and ids[a] != ids[b]:
            return f"different identifiers {sorted(ids[a])} / {sorted(ids[b])}"
        if any(frozenset((x, y)) in never_merge for x in members[a] for y in members[b]):
            return "linked by a replaces edge"
        return None

    def union(a: Ref, b: Ref, method: str, chunks: set[str]):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        reason = blocked(ra, rb)
        if reason:
            refused.append({"method": method, "a": _names(members[ra], surfaces),
                            "b": _names(members[rb], surfaces), "reason": reason})
            return
        parent[rb] = ra
        merges.append({"method": method, "into": ra, "absorbed": list(members[rb]),
                       "chunks": sorted(chunks | set().union(*(surfaces[m].chunks for m in members[rb])))})
        members[ra] += members.pop(rb)
        ids[ra] |= ids.pop(rb)

    ordered = sorted(surfaces)  # deterministic

    for rule in RULES:
        if rule not in rules:
            continue

        if rule in ("identifier", "normalised"):
            groups: dict[tuple, list[Ref]] = defaultdict(list)
            for ref in ordered:
                if rule == "identifier":
                    found = set().union(*(identifiers(n, ref[0]) for n in surfaces[ref].names))
                    if found:
                        groups[(ref[0], frozenset(found))].append(ref)
                else:
                    for key in {normalise_name(n) for n in surfaces[ref].names}:
                        groups[(ref[0], key)].append(ref)
            for group in groups.values():
                for other in group[1:]:
                    union(group[0], other, rule, set())

        else:  # alias
            by_name: dict[tuple[str, str], list[Ref]] = defaultdict(list)
            for ref in ordered:
                for n in surfaces[ref].names:
                    by_name[(ref[0], _key(n))].append(ref)
                    if "normalised" in rules:
                        by_name[(ref[0], normalise_name(n))].append(ref)
            for ref, alias, chunk in alias_links:
                for other in by_name.get((ref[0], _key(alias)), []) + (
                        by_name.get((ref[0], normalise_name(alias)), []) if "normalised" in rules else []):
                    union(ref, other, "alias", {chunk})

    clusters: dict[Ref, list[Ref]] = {}
    for ref in ordered:
        clusters.setdefault(find(ref), []).append(ref)
    root = {ref: rep for rep, refs in clusters.items() for ref in refs}
    final_merges = [{**m, "into": find(m["into"])} for m in merges]
    return Resolution(root, clusters, final_merges, refused)


def _key(name: str) -> str:
    return " ".join(fold(name).replace("*", "").replace("`", "").split())


def _names(refs: list[Ref], surfaces: dict[Ref, Surface]) -> list[str]:
    return sorted({n for r in refs for n in surfaces[r].names})


def replaces_pairs(edges: list[tuple[Ref, str, Ref]]) -> set[frozenset[Ref]]:
    """Both ends of every `replaces` edge: an old and a new entity, never one."""
    return {frozenset((src, dst)) for src, rel, dst in edges if rel == "replaces" and src != dst}
