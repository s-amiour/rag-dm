"""Concept-graph schema for the Kestrel corpus (ticket v3T01).

The schema is FIXED: the LLM may only use the entity types and relation types
listed here, and every relation declares which entity types it may connect.
Why fixed and not open: see DECISIONS.md D-20.

Bump SCHEMA_VERSION whenever this file changes. The version is part of the
extraction cache key and of the graph fingerprint, so a schema change
re-extracts every chunk instead of mixing old and new triples.
"""
from __future__ import annotations

SCHEMA_VERSION = "1"

# name -> one-line definition (sent to the LLM in the prompt)
ENTITY_TYPES: dict[str, str] = {
    "Product": "a product or product area named in the chunk (e.g. the public API, embedded dashboards)",
    "Feature": "a capability a user can use or configure (e.g. single sign-on, scheduled reports, audit log)",
    "Plan": "a subscription plan (Free, Starter, Team, Business, Enterprise)",
    "Service": "an internal production service from the service catalog (e.g. gateway, auth, postroom, ledger, mailman)",
    "ErrorCode": "an error code starting with KST- (e.g. KST-5031)",
    "Policy": "a written policy or rule set (e.g. Billing and Refund Policy 2026, the refund rules)",
    "Procedure": "a support or admin procedure, a fix or an action (e.g. add both IP addresses to the allowlist)",
    "Team": "an engineering team that owns services (e.g. Platform, Identity, Messaging, Payments Core)",
}

# name -> allowed source types, allowed target types, definition
RELATIONS: dict[str, dict] = {
    "has_feature": {
        "source": {"Product", "Plan", "Service"},
        "target": {"Feature"},
        "doc": "the source offers this feature",
    },
    "included_in": {
        "source": {"Feature"},
        "target": {"Plan"},
        "doc": "the feature is available on this plan",
    },
    "depends_on": {
        "source": {"Service", "Feature", "Product"},
        "target": {"Service", "Feature"},
        "doc": "the source needs the target to work",
    },
    "causes": {
        "source": {"Service", "Feature", "Policy", "Procedure"},
        "target": {"ErrorCode"},
        "doc": "the source situation or component leads to this error",
    },
    "resolved_by": {
        "source": {"ErrorCode"},
        "target": {"Procedure", "Feature"},
        "doc": "the error is fixed by this action",
    },
    "governed_by": {
        "source": {"Plan", "Feature", "Service", "Procedure", "ErrorCode"},
        "target": {"Policy"},
        "doc": "the source follows the rules of this policy",
    },
    "owned_by": {
        "source": {"Service"},
        "target": {"Team"},
        "doc": "the service is owned by this team",
    },
    "replaces": {
        "source": {"Service", "Policy", "Product", "Feature", "Plan"},
        "target": {"Service", "Policy", "Product", "Feature", "Plan"},
        "same_type": True,
        "doc": "the source is the newer version of the target (both of the same type)",
    },
    # Added to the PRD default list (DECISIONS.md D-20): without them the
    # q08 / q11 / q12 chains (error code -> service -> team) cannot be written.
    "raised_by": {
        "source": {"ErrorCode"},
        "target": {"Service"},
        "doc": "the error code is raised by this service",
    },
    "escalates_to": {
        "source": {"ErrorCode"},
        "target": {"Team", "Service"},
        "doc": "support escalates this error to this team (or to the team that owns this service)",
    },
}


def relation_allowed(relation: str, source_type: str, target_type: str) -> bool:
    """True if `relation` may go from an entity of `source_type` to one of `target_type`."""
    rule = RELATIONS.get(relation)

    if rule is None:
        return False

    if source_type not in rule["source"] or target_type not in rule["target"]:
        return False

    if rule.get("same_type") and source_type != target_type:
        return False

    return True


def schema_prompt() -> str:
    """The schema as plain text, for the extraction prompt."""
    lines = ["ENTITY TYPES (use only these):"]

    for name, doc in ENTITY_TYPES.items():
        lines.append(f"- {name}: {doc}")

    lines.append("")
    lines.append("RELATION TYPES (use only these, only between the listed types):")

    for name, rule in RELATIONS.items():
        src = " | ".join(sorted(rule["source"]))
        tgt = " | ".join(sorted(rule["target"]))
        extra = " (source and target of the same type)" if rule.get("same_type") else ""
        lines.append(f"- {name}: {src} -> {tgt}{extra}. {rule['doc']}")

    return "\n".join(lines)
