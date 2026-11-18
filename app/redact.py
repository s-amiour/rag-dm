"""Personal data redaction (V4 Part C, ticket v4T09). DECISIONS D-37.

Personal data is replaced by placeholders before text leaves the machine (the call to the
LLM provider) and before it is written to a trace:

    redacted, mapping = redact("Mail me at jo@example.com")
    # redacted == "Mail me at [EMAIL_1]",  mapping == {"[EMAIL_1]": "jo@example.com"}

The same value always gets the same placeholder within one mapping, so pass the mapping
on to redact several texts of one request (question + chunks) consistently. `restore()`
puts the real values back into the answer, which is shown only on this machine.

The mapping holds the personal data itself: keep it in memory for the request only.
Never write it to a trace, a log or a result file.

What is detected (rules, no model, no network): e-mail addresses, phone numbers
(international, French and North American formats), payment card numbers (Luhn check),
IBANs (mod-97 check), and names introduced by a cue ("my name is", "I'm", Mr/Ms/Dr...).
A name with no cue is not detected (a known limit, D-37).

Switch: PII_REDACTION=0 in .env turns it off (default: on).
"""
from __future__ import annotations

import os
import re
from collections import Counter

KINDS = ("EMAIL", "PHONE", "CARD", "IBAN", "NAME")

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
_PHONE = re.compile(
    r"(?<![\w.])(?:"
    r"\+\d{1,3}[\s.-]?(?:\(0\)[\s.-]?)?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}"   # +33 6 12 34 56 78, +44 20 7946 0958
    r"|0[1-9](?:[\s.-]?\d{2}){4}"                                        # 06 12 34 56 78, 06.12.34.56.78
    r"|\(\d{3}\)\s?\d{3}[\s.-]\d{4}"                                     # (555) 123-4567
    r"|\d{3}[.-]\d{3}[.-]\d{4}"                                          # 555-123-4567
    r")(?![\w.]*\d)"
)
_CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){2,7}(?:[ ]?[A-Z0-9]{1,3})?\b")
_NAME = re.compile(
    r"(?i:\b(?:my name is|i am|i'm|je m'appelle|je suis)\s+|\b(?:mr|mrs|ms|miss|dr|prof|mme|mlle|m)\.?\s+)"
    r"([A-ZÀ-Ý][a-zà-ÿ'-]+(?:[ -][A-ZÀ-Ý][a-zà-ÿ'-]+){0,2})"
)

# Capitalised words that follow "I am" in support questions but are roles or plans, not
# names. Without this, "I am Owner of the workspace" loses the role the question is about.
NOT_NAMES = {
    "owner", "admin", "admins", "administrator", "editor", "viewer", "member", "members",
    "guest", "free", "starter", "team", "business", "enterprise", "pro", "kestrel",
}


def _not_a_role(value: str) -> bool:
    return value.split()[0].lower() not in NOT_NAMES


def _luhn(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def _iban_ok(value: str) -> bool:
    s = value.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1
    except ValueError:
        return False


def enabled() -> bool:
    return os.getenv("PII_REDACTION", "1").strip().lower() not in ("0", "false", "no")


def redact(text: str, mapping: dict[str, str] | None = None) -> tuple[str, dict[str, str]]:
    """Replace personal data in `text` by placeholders. Returns (redacted_text, mapping),
    mapping = {placeholder: original value}. Pass an existing mapping to extend it."""
    mapping = {} if mapping is None else mapping
    if not text:
        return text, mapping
    reverse = {v: k for k, v in mapping.items()}

    def placeholder(kind: str, value: str) -> str:
        if value in reverse:
            return reverse[value]
        n = 1 + sum(1 for k in mapping if k.startswith(f"[{kind}_"))
        key = f"[{kind}_{n}]"
        mapping[key] = value
        reverse[value] = key
        return key

    def sub(kind, pattern, check=None, group=0):
        nonlocal text

        def repl(m):
            value = m.group(group)
            if check and not check(value):
                return m.group(0)
            if group == 0:
                return placeholder(kind, value)
            start, end = m.start(group) - m.start(0), m.end(group) - m.start(0)
            whole = m.group(0)
            return whole[:start] + placeholder(kind, value) + whole[end:]

        text = pattern.sub(repl, text)

    # order matters: e-mails first (they contain digits), IBAN and cards before phones
    sub("EMAIL", _EMAIL)
    sub("IBAN", _IBAN, _iban_ok)
    sub("CARD", _CARD, _luhn)
    sub("PHONE", _PHONE)
    sub("NAME", _NAME, _not_a_role, group=1)
    return text, mapping


def restore(text: str, mapping: dict[str, str]) -> str:
    """Put the original values back (for display on this machine only)."""
    for key in sorted(mapping, key=len, reverse=True):  # [EMAIL_10] before [EMAIL_1]
        text = text.replace(key, mapping[key])
    return text


def counts(mapping: dict[str, str]) -> dict[str, int]:
    """How many values of each kind were redacted: safe to log (no personal data)."""
    c = Counter(key[1:].rsplit("_", 1)[0] for key in mapping)
    return {kind: c[kind] for kind in KINDS if c[kind]}
