"""Request signals, confidence scoring, safety flags and escalation decisions."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from . import config
from .models import (
    Article,
    Assessment,
    Classification,
    Draft,
    GroundingReport,
    RequestSignals,
    RetrievalHit,
    Ticket,
)

# Questions that can only be answered by looking at the customer's own account.
_ACCOUNT_SPECIFIC_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bwhy (was|is|has|did|were|are|have|does|do|hasn'?t|isn'?t|wasn'?t)\b[^?.!]*\bmy\b",
        r"\b(when|how long)\b[^?.!]*\bmy\b[^?.!]*\b(arriv|approv|process|credit|complet|releas|unlock)",
        r"\b(status of|check) my (balance|account|withdrawal|deposit|payment|transaction)",
        r"\b(what is|what's) my (balance|account|withdrawal|deposit)",
    )
]
# KB wording that marks an article's topic as needing agent review.
_POLICY_SENSITIVE_KB_RE = re.compile(
    r"\b(compliance|should not promise|must not promise)\b", re.IGNORECASE
)


def policy_sensitive_articles(article_ids: Sequence[str], kb: Dict[str, Article]) -> List[str]:
    """Articles whose own wording says outcomes need an agent/compliance check."""
    return [aid for aid in article_ids if aid in kb and _POLICY_SENSITIVE_KB_RE.search(kb[aid].body)]


def is_policy_sensitive(intent: str, article_ids: Sequence[str], kb: Dict[str, Article]) -> bool:
    return intent in config.POLICY_SENSITIVE_INTENTS or bool(policy_sensitive_articles(article_ids, kb))


def runner_up(classification: Classification) -> Optional[str]:
    ranked = sorted(
        (i for i in classification.scores if i != classification.intent),
        key=lambda i: (-classification.scores[i], config.ALLOWED_INTENTS.index(i)),
    )
    return ranked[0] if ranked else None


def analyze_request(ticket: Ticket, classification: Classification) -> RequestSignals:
    signals = RequestSignals()
    message = ticket.message.strip()
    if not message:
        signals.empty_message = True
        signals.evidence.append("ticket message is empty")
    if ticket.language not in config.SUPPORTED_LANGUAGES:
        signals.unsupported_language = True
        signals.evidence.append(f"language '{ticket.language}' is not supported by the KB")

    advice_score = classification.scores.get(config.ADVICE_INTENT, 0.0)
    if classification.intent == config.ADVICE_INTENT or advice_score >= config.ADVICE_SCORE_THRESHOLD:
        signals.advice_request = True
        signals.evidence.append(f"trading-advice rules scored {advice_score}")

    for pattern in _ACCOUNT_SPECIFIC_PATTERNS:
        match = pattern.search(message)
        if match:
            signals.account_specific = True
            signals.evidence.append(f"asks about own account: '{match.group(0)}'")
            break

    top = classification.scores.get(classification.intent, 0.0)
    second = runner_up(classification)
    if top > 0 and second and classification.scores[second] >= top * config.CLASSIFIER_CONFLICT_RATIO:
        signals.multi_intent = True
        signals.evidence.append(
            f"mixed intents: {classification.intent}={top}, {second}={classification.scores[second]}"
        )
    return signals


def _ordered(flags: Sequence[str]) -> List[str]:
    unique = set(flags)
    return [f for f in config.FLAG_ORDER if f in unique] + sorted(unique - set(config.FLAG_ORDER))


def assess(
    ticket: Ticket,
    classification: Classification,
    signals: RequestSignals,
    hits: Sequence[RetrievalHit],
    draft: Draft,
    grounding: GroundingReport,
    kb: Dict[str, Article],
    article_topics: Dict[str, str],
) -> Assessment:
    flags: List[str] = []
    reasons: List[str] = []
    intent = classification.intent

    # --- conflicting signals: classifier vs retrieval, or two strong intents
    conflict_notes = []
    if hits and intent != config.OTHER_INTENT:
        top_topic = article_topics.get(hits[0].article_id)
        if top_topic and top_topic != intent and not (
            signals.advice_request and top_topic == config.ADVICE_INTENT
        ):
            conflict_notes.append(
                f"classifier says '{intent}' but top article {hits[0].article_id} is about '{top_topic}'"
            )
    if signals.multi_intent:
        conflict_notes.append("ticket mixes more than one issue")
    conflicting = bool(conflict_notes)

    # --- confidence
    retrieval_strength = min(1.0, hits[0].score / config.RETRIEVAL_SCORE_REF) if hits else 0.0
    breakdown = {
        "classifier_certainty": round(classification.certainty, 4),
        "retrieval_strength": round(retrieval_strength, 4),
        "grounding_score": round(grounding.score, 4),
    }
    confidence = sum(config.CONFIDENCE_WEIGHTS[k] * v for k, v in breakdown.items())
    used_ids = draft.citations or [h.article_id for h in hits[:1]]
    policy_sensitive = is_policy_sensitive(intent, used_ids, kb)
    penalties = (
        ("conflict_penalty", conflicting, config.CONFLICT_PENALTY),
        ("policy_sensitive_penalty", policy_sensitive, config.POLICY_SENSITIVE_PENALTY),
        ("account_specific_penalty", signals.account_specific, config.ACCOUNT_SPECIFIC_PENALTY),
    )
    for name, applies, penalty in penalties:
        if applies:
            confidence -= penalty
            breakdown[name] = -penalty
    if signals.empty_message:
        confidence = 0.0
    confidence = round(max(0.0, min(1.0, confidence)), 2)

    # --- flags and escalation reasons
    if signals.advice_request:
        flags.append(config.FLAG_ADVICE_REQUEST)
    if signals.empty_message:
        flags.append(config.FLAG_EMPTY_MESSAGE)
        reasons.append("ticket message is empty; need more information from the customer")
    if signals.unsupported_language:
        flags.append(config.FLAG_UNSUPPORTED_LANGUAGE)
        reasons.append(f"language '{ticket.language}' is not covered by the knowledge base")
    if not hits and not signals.empty_message:
        flags.append(config.FLAG_INSUFFICIENT_GROUNDING)
        reasons.append("no relevant knowledge-base article found")
    elif not grounding.grounded and hits:
        flags.append(config.FLAG_INSUFFICIENT_GROUNDING)
        detail = grounding.weak_sentences + grounding.unsupported_framing + grounding.invalid_citations
        reasons.append("reply is not fully grounded in retrieved articles" + (f": {detail}" if detail else ""))
    if grounding.unsupported_claims:
        flags.append(config.FLAG_UNSUPPORTED_CLAIM)
        reasons.append(f"reply contains unsupported claims: {grounding.unsupported_claims}")

    sensitive_kb = policy_sensitive_articles(used_ids, kb)
    if policy_sensitive:
        flags.append(config.FLAG_POLICY_SENSITIVE)
        source = f" (KB {', '.join(sensitive_kb)} requires checking account status/compliance)" if sensitive_kb else ""
        reasons.append(f"policy-sensitive topic '{intent}'{source}; agent must review before any commitment")
    if signals.account_specific:
        flags.append(config.FLAG_ACCOUNT_SPECIFIC)
        reasons.append("answer depends on the customer's own account data, which the draft cannot see")
    if conflicting:
        flags.append(config.FLAG_CONFLICTING_SIGNALS)
        reasons.append("conflicting signals: " + "; ".join(conflict_notes))
    if confidence < config.LOW_CONFIDENCE_THRESHOLD:
        flags.append(config.FLAG_LOW_CONFIDENCE)
        reasons.append(f"confidence {confidence} is below {config.LOW_CONFIDENCE_THRESHOLD}")

    escalate = bool(reasons)
    if escalate:
        explanation = "Escalated: " + "; ".join(reasons) + "."
    elif signals.advice_request:
        explanation = (
            "Not escalated: trading-advice request was politely refused using the "
            f"policy article ({', '.join(draft.citations) or 'none'}); no human action needed."
        )
    else:
        top = hits[0]
        explanation = (
            f"Not escalated: intent '{intent}' (certainty {classification.certainty:.2f}), "
            f"relevant article {top.article_id} '{top.title}' (BM25 {top.score:.2f}), "
            f"reply grounded (score {grounding.score:.2f}), confidence {confidence:.2f} >= "
            f"{config.LOW_CONFIDENCE_THRESHOLD}, no policy-sensitive or account-specific signals."
        )
    return Assessment(
        confidence=confidence,
        confidence_breakdown=breakdown,
        safety_flags=_ordered(flags),
        needs_human_escalation=escalate,
        escalation_reasons=reasons,
        explanation=explanation,
    )
