"""Injection scanner (V4 Part C, ticket v4T08): pattern rules run on every uploaded
document before it can be indexed. DECISIONS D-36.

A document with at least one finding is quarantined (data/quarantine/<corpus>/): it is
not in the corpus folder, so it is never chunked, embedded or indexed, until a person
approves it (or rejects it) in the Documents tab.

Rules, not an LLM: they are free, instant, deterministic and explainable (each finding
says which rule fired, on which line, with the text). They only catch known shapes of
attack: a paraphrased instruction ("before answering, the assistant must...") gets
through. That limit is in DECISIONS D-36 and in the report.

Command line:  python -m app.injection path/to/file.md [more files]
"""
from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

RULES_VERSION = "v1"

# (rule id, why it is suspicious, compiled pattern). Patterns run on each line, case-insensitive,
# after accents and full-width letters are folded (so "ｉｇｎｏｒｅ" or "ïgnore" are read as "ignore"). English only.
_I = re.IGNORECASE
RULES: list[tuple[str, str, re.Pattern]] = [
    ("override",
     "asks to ignore or replace earlier instructions",
     re.compile(r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any|your|the|system)\b"
                r"[^.\n]{0,20}\b(instructions?|prompts?|rules|directions|guidelines|context)\b", _I)),
    ("new-instructions",
     "announces new instructions or a new identity for the model",
     re.compile(r"\b(new|updated|real|actual|secret)\s+(system\s+)?(instructions?|prompt)\s*:|\byou are now\b|\bfrom now on,? you\b", _I)),
    ("addresses-ai",
     "speaks to an AI reading the document, not to a human reader",
     re.compile(r"\b(if|when)\s+you\s+are\s+an?\s+(ai|llm|language model|assistant|chatbot|agent|coding agent|ai (coding )?agent)\b"
                r"|\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(ai|llm|assistant|language model|model)\b"
                r"|\bdear\s+(ai|llm|assistant|chatgpt|claude|gemini|copilot)\b", _I)),
    ("role-marker",
     "chat-template role marker, used to fake a system or assistant turn",
     re.compile(r"^\s*(#+\s*)?(system|assistant)\s*:\s*\S|<\|\s*(im_start|im_end|system|assistant|user|endoftext)\s*\|>"
                r"|\[/?INST\]|<</?SYS>>|</?s>\s*$", _I | re.MULTILINE)),
    ("exfiltration",
     "asks to reveal the prompt or send secrets somewhere",
     re.compile(r"\b(reveal|print|repeat|show|output)\b[^.\n]{0,30}\b(system prompt|your (instructions|prompt)|hidden instructions)\b"
                r"|\b(send|email|post|upload|forward|paste)\s+(me\s+|us\s+)?(the\s+|your\s+|all\s+|any\s+)?(api[ _-]?keys?|passwords?|secrets?|access tokens?|credentials|\.env)\b", _I)),
]

# Characters a reader cannot see but a model reads: zero-width characters, word joiner, BOM in the
# middle of text, bidi overrides, and Unicode "tag" characters (U+E0000 block) that can spell out a
# whole hidden instruction.
_INVISIBLE = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
              0x2066, 0x2067, 0x2068, 0x2069}


@dataclass
class Finding:
    rule: str
    reason: str
    line: int
    excerpt: str


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def _excerpt(line: str, start: int = 0, width: int = 120) -> str:
    begin = max(0, start - 30)
    snippet = line[begin:begin + width].strip()
    return ("…" if begin else "") + snippet + ("…" if begin + width < len(line) else "")


def scan(text: str) -> list[Finding]:
    """All findings in a document, in line order. Empty list = clean."""
    findings: list[Finding] = []

    for no, raw in enumerate(text.splitlines(), start=1):
        hidden = [c for c in raw if ord(c) in _INVISIBLE or 0xE0000 <= ord(c) <= 0xE007F]
        if no == 1 and raw.startswith("\ufeff"):
            hidden = hidden[1:]  # a BOM at the very start of a file is normal
        if hidden:
            decoded = "".join(chr(ord(c) - 0xE0000) for c in hidden if 0xE0020 <= ord(c) <= 0xE007E)
            findings.append(Finding(
                "invisible-text",
                f"{len(hidden)} invisible character(s): text a reader cannot see but a model reads",
                no,
                ("hidden text: " + _excerpt(decoded)) if decoded else _excerpt(raw)))

        line = _fold(raw)
        for rule, reason, pattern in RULES:
            match = pattern.search(line)
            if match:
                findings.append(Finding(rule, reason, no, _excerpt(line, match.start())))

    return findings


def report(text: str) -> dict:
    """What the API stores and returns: a verdict, the findings, and the rules version."""
    findings = scan(text)
    return {
        "suspicious": bool(findings),
        "rules_version": RULES_VERSION,
        "findings": [asdict(f) for f in findings],
    }


def main(paths: list[str]) -> int:
    flagged = 0
    for path in paths:
        result = report(Path(path).read_text(encoding="utf-8"))
        status = "SUSPICIOUS" if result["suspicious"] else "clean"
        print(f"{status:10} {path}")
        for f in result["findings"]:
            print(f"    line {f['line']:>4}  [{f['rule']}] {f['excerpt']}")
        flagged += result["suspicious"]
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
