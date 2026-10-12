"""VALIDATE_OUTPUT stage.

Two layers:
  * hard schema checks - a failing record is replaced by a safe fallback, so
    written output always conforms to the schema;
  * semantic safety checks - violations are repaired (flag + escalate, or a
    standard refusal) and the repair is recorded for the debug report.

`schema_errors` / `semantic_errors` are also used by validate.py to audit
results.json independently of the pipeline.
"""
from __future__ import annotations

import copy
import math
import re
from typing import Dict, Iterable, List, Tuple

from . import config, templates
from .grounding import extract_citations, find_unsupported_claims
from .models import Article

REFUSAL_RE = re.compile(
    r"\b(can(?:no|')?t|cannot|unable to|not able to|won'?t be able to|do not provide|"
    r"don'?t provide|must not provide)\b",
    re.IGNORECASE,
)
# Language that would amount to giving trading advice or predictions.
ADVICE_CONTENT_RE = re.compile(
    r"\b(you should|i recommend|we recommend|i suggest|we suggest|consider) "
    r"(buy|sell|invest|trad)\w*|\b(will|is going to|is likely to|likely to) "
    r"(go|rise|increase|move|climb|fall|drop) ?(up|down|higher|lower)?\b|\b(bullish|bearish)\b|"
    r"\bbest (asset|stock|coin|trade)s? (to|for)\b",
    re.IGNORECASE,
)
ESCALATING_FLAGS = frozenset(
    {
        config.FLAG_INSUFFICIENT_GROUNDING,
        config.FLAG_UNSUPPORTED_CLAIM,
        config.FLAG_POLICY_SENSITIVE,
        config.FLAG_ACCOUNT_SPECIFIC,
        config.FLAG_LOW_CONFIDENCE,
        config.FLAG_CONFLICTING_SIGNALS,
        config.FLAG_UNSUPPORTED_LANGUAGE,
        config.FLAG_EMPTY_MESSAGE,
        config.FLAG_VALIDATION_FAILED,
    }
)


def is_safe_refusal(reply: str) -> bool:
    return bool(REFUSAL_RE.search(reply)) and not ADVICE_CONTENT_RE.search(reply)


def schema_errors(record: object, known_article_ids: Iterable[str]) -> List[str]:
    """Hard checks against the required output schema."""
    if not isinstance(record, dict):
        return ["record is not a JSON object"]
    errors: List[str] = []
    known = set(known_article_ids)
    missing = [f for f in config.RESULT_SCHEMA if f not in record]
    extra = sorted(set(record) - set(config.RESULT_SCHEMA))
    if missing:
        errors.append(f"missing fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    for field, expected in config.RESULT_SCHEMA.items():
        if field not in record:
            continue
        value = record[field]
        # bool is a subclass of int: reject it where a number is required.
        if not isinstance(value, expected) or (expected is not bool and isinstance(value, bool)):
            errors.append(f"field '{field}' has type {type(value).__name__}")
    if errors:
        return errors

    if not record["ticket_id"].strip():
        errors.append("ticket_id is empty")
    if record["intent"] not in config.ALLOWED_INTENTS:
        errors.append(f"intent '{record['intent']}' is not an allowed label")
    articles = record["retrieved_articles"]
    if not all(isinstance(a, str) for a in articles):
        errors.append("retrieved_articles must contain strings")
    else:
        unknown = [a for a in articles if a not in known]
        if unknown:
            errors.append(f"unknown article ids: {unknown}")
        if len(set(articles)) != len(articles):
            errors.append("retrieved_articles contains duplicates")
    if len(articles) > config.MAX_RETRIEVED:
        errors.append(f"more than {config.MAX_RETRIEVED} retrieved articles")
    confidence = record["confidence"]
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        errors.append(f"confidence {confidence} is outside [0, 1]")
    if not record["reply_draft"].strip():
        errors.append("reply_draft is empty")
    if not all(isinstance(f, str) for f in record["safety_flags"]):
        errors.append("safety_flags must contain strings")
    return errors


def semantic_errors(record: dict, kb: Dict[str, Article]) -> List[str]:
    """Safety checks on a schema-valid record."""
    errors: List[str] = []
    reply = record["reply_draft"]
    retrieved = record["retrieved_articles"]
    flags = set(record["safety_flags"])
    escalated = record["needs_human_escalation"]

    bad_citations = [c for c in extract_citations(reply) if c not in retrieved]
    if bad_citations:
        errors.append(f"reply cites articles that were not retrieved: {bad_citations}")
    if record["grounded"] and (not retrieved or not extract_citations(reply)):
        errors.append("record claims grounded=true without retrieved and cited articles")

    advice = record["intent"] == config.ADVICE_INTENT or config.FLAG_ADVICE_REQUEST in flags
    if advice and not is_safe_refusal(reply):
        errors.append("trading-advice request is not safely refused")

    source = " ".join(kb[a].text for a in retrieved if a in kb)
    claims = find_unsupported_claims(reply, source)
    if claims and config.FLAG_UNSUPPORTED_CLAIM not in flags and not escalated:
        errors.append(f"unsupported claims without flag or escalation: {claims}")
    if flags & ESCALATING_FLAGS and not escalated:
        errors.append(f"risk flags {sorted(flags & ESCALATING_FLAGS)} present but not escalated")
    return errors


def safe_fallback(record: dict, known_article_ids: Iterable[str]) -> dict:
    """A conservative, always-valid record used when validation fails hard."""
    known = set(known_article_ids)
    intent = record.get("intent") if isinstance(record, dict) else None
    retrieved = record.get("retrieved_articles") if isinstance(record, dict) else None
    flags = record.get("safety_flags") if isinstance(record, dict) else None
    clean_flags = [f for f in (flags or []) if isinstance(f, str)]
    return {
        "ticket_id": str(record.get("ticket_id", "")) if isinstance(record, dict) else "",
        "intent": intent if intent in config.ALLOWED_INTENTS else config.OTHER_INTENT,
        "retrieved_articles": [
            a for a in (retrieved or []) if isinstance(a, str) and a in known
        ][: config.MAX_RETRIEVED],
        "reply_draft": templates.SAFE_FALLBACK_REPLY,
        "grounded": False,
        "confidence": 0.0,
        "needs_human_escalation": True,
        "safety_flags": _merge_flags(clean_flags, [config.FLAG_VALIDATION_FAILED]),
    }


def _merge_flags(flags: Iterable[str], extra: Iterable[str]) -> List[str]:
    merged = set(flags) | set(extra)
    return [f for f in config.FLAG_ORDER if f in merged] + sorted(merged - set(config.FLAG_ORDER))


def enforce(record: dict, kb: Dict[str, Article]) -> Tuple[dict, List[str], List[str]]:
    """Validate a record and repair it safely.

    Returns (record_to_write, errors_found, fixes_applied).
    """
    known = kb.keys()
    errors = schema_errors(record, known)
    if errors:
        return safe_fallback(record, known), errors, ["replaced with safe fallback record"]

    fixed = copy.deepcopy(record)
    fixes: List[str] = []
    errors = semantic_errors(fixed, kb)
    for error in errors:
        if error.startswith("trading-advice"):
            fixed["reply_draft"] = templates.REFUSAL_FALLBACK
            fixed["grounded"] = False
            fixed["safety_flags"] = _merge_flags(
                fixed["safety_flags"], [config.FLAG_ADVICE_REQUEST, config.FLAG_INSUFFICIENT_GROUNDING]
            )
            fixed["needs_human_escalation"] = True
            fixes.append("replaced unsafe reply with standard trading-advice refusal")
        elif error.startswith(("reply cites", "record claims grounded")):
            fixed["grounded"] = False
            fixed["safety_flags"] = _merge_flags(fixed["safety_flags"], [config.FLAG_INSUFFICIENT_GROUNDING])
            fixed["needs_human_escalation"] = True
            fixes.append("marked ungrounded and escalated")
        elif error.startswith("unsupported claims"):
            fixed["safety_flags"] = _merge_flags(fixed["safety_flags"], [config.FLAG_UNSUPPORTED_CLAIM])
            fixed["needs_human_escalation"] = True
            fixes.append("flagged unsupported claim and escalated")
        elif error.startswith("risk flags"):
            fixed["needs_human_escalation"] = True
            fixes.append("escalated because risk flags were present")

    remaining = schema_errors(fixed, known) + semantic_errors(fixed, kb)
    if remaining:
        return safe_fallback(fixed, known), errors + remaining, fixes + ["replaced with safe fallback record"]
    return fixed, errors, fixes
