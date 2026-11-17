"""Entity resolution for the concept graph (ticket v3T02).

T1 gives one node per (type, name) as written. Here we decide which of those
nodes are the same thing, with simple rules that can be explained and undone.
"""
from __future__ import annotations

import re

from .bm25 import fold

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
