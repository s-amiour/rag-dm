"""Entity resolution for the concept graph (ticket v3T02).

T1 gives one node per (type, name) as written. Here we decide which of those
nodes are the same thing, with simple rules that can be explained and undone.
"""
from __future__ import annotations

import re

from .bm25 import fold

ARTICLES = {"the", "a", "an"}
GENERIC_SUFFIXES = {"team", "teams", "service", "services"}

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
