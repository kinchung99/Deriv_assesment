"""CLASSIFY_INTENT stage.

RuleClassifier is the deterministic default. LLMClassifier is an optional
drop-in that constrains the model to the allowed labels and falls back to the
rules on any failure.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .models import Classification, Ticket

log = logging.getLogger(__name__)

Rule = Tuple[str, float]

# (regex, weight). Strong, specific phrases outweigh single keywords.
INTENT_RULES: Dict[str, Sequence[Rule]] = {
    "deposit_issue": (
        (r"\bdeposit\w*", 3.0),
        (r"\b(top(ped)?[\s-]?up|fund(ed|ing)? (my |the )?account)\b", 2.0),
        (r"\b(bank |credit |debit )?card\b", 1.0),
        (r"\bbalance\b", 1.0),
        (r"\b(credited|not reflect\w*|not show\w*|shows? zero)\b", 1.0),
    ),
    "password_reset": (
        (r"\bpass ?words?\b", 3.0),
        (r"\b(reset|recovery) (e-?mail|link|code)\b", 2.0),
        (r"\breset\b", 1.5),
        (r"\bforg[oe]t\w*\b", 1.5),
    ),
    "withdrawal_issue": (
        (r"\bwithdr[ae]w\w*", 3.0),
        (r"\b(cash[\s-]?out|pay[\s-]?out)s?\b", 2.0),
        (r"\b(declin|reject)\w*", 1.0),
        (r"\bverif\w*", 0.5),
    ),
    "account_lock": (
        (r"\block(ed|out|s)?\b|\blocked out\b", 3.0),
        (r"\b(blocked|suspended|disabled|frozen)\b", 2.0),
        (r"\b(too many|failed|multiple|several) (\w+ )?(attempts|tries)\b", 2.0),
        (r"\b(log[\s-]?in|sign[\s-]?in)\b", 1.0),
    ),
    "trading_advice_request": (
        (r"\b(which|what) (asset|stock|coin|crypto|currency|pair|share|market)s?\b", 3.0),
        (r"\b(what|which) (to|should i|do i) (buy|sell|trade|invest)\b", 3.0),
        (r"\b(make|making|earn|earning|guaranteed?)\s+(a\s+|some\s+)?(profit|money|returns?)\b", 3.0),
        (r"\b(go(es)?|going|rise|rising) (up|down)\b", 2.0),
        (r"\b(predict\w*|forecast\w*|signals?|tips?)\b", 2.0),
        (r"\bprofit\w*\b", 1.5),
        (r"\b(buy|sell|invest\w*)\b", 1.5),
        (r"\b(recommend\w*|advice|advise)\b", 1.0),
        (r"\b(trade|trading|market)\b", 1.0),
    ),
}

_COMPILED: Dict[str, List[Tuple[re.Pattern, float, str]]] = {
    intent: [(re.compile(p, re.IGNORECASE), w, p) for p, w in rules]
    for intent, rules in INTENT_RULES.items()
}


def certainty_from_scores(scores: Dict[str, float]) -> Tuple[str, float]:
    """Winner label + certainty = margin share x rule coverage, in [0, 1]."""
    ranked = sorted(
        ((s, config.ALLOWED_INTENTS.index(i), i) for i, s in scores.items()),
        key=lambda t: (-t[0], t[1]),
    )
    top_score, _, top_intent = ranked[0]
    if top_score <= 0:
        return config.OTHER_INTENT, 0.0
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin_share = top_score / (top_score + second)
    coverage = min(1.0, top_score / config.CLASSIFIER_STRONG_SCORE)
    return top_intent, round(margin_share * coverage, 4)


class RuleClassifier:
    name = "rules"

    def score(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        scores: Dict[str, float] = {}
        matched: List[str] = []
        for intent, rules in _COMPILED.items():
            total = 0.0
            for pattern, weight, raw in rules:
                if pattern.search(text):
                    total += weight
                    matched.append(f"{intent}:{raw}")
            scores[intent] = round(total, 4)
        return scores, matched

    def classify_text(self, text: str) -> Classification:
        scores, matched = self.score(text or "")
        intent, certainty = certainty_from_scores(scores)
        return Classification(
            intent=intent, certainty=certainty, scores=scores, matched_rules=matched, source=self.name
        )

    def classify(self, ticket: Ticket) -> Classification:
        return self.classify_text(ticket.message)


class LLMClassifier:
    """Claude-backed classifier constrained to ALLOWED_INTENTS.

    The rule classifier still runs: its scores feed safety signals (e.g. an
    advice request is never missed because the model labelled it otherwise)
    and it is the fallback whenever the model is unavailable or off-label.
    """

    name = "llm"

    def __init__(self, client=None, fallback: Optional[RuleClassifier] = None) -> None:
        from .llm_client import get_default_client

        self.client = client if client is not None else get_default_client()
        self.fallback = fallback or RuleClassifier()

    def classify(self, ticket: Ticket) -> Classification:
        from .prompts import build_classifier_prompt

        rules = self.fallback.classify(ticket)
        if self.client is None:
            rules.source = "rules_fallback"
            rules.notes.append("LLM unavailable (no API key or SDK); used rule classifier")
            return rules
        system, user, schema = build_classifier_prompt(ticket.message)
        try:
            data = self.client.complete_json(system=system, user=user, schema=schema)
            label = str(data.get("intent", "")).strip()
            llm_certainty = float(data.get("confidence", 0.0))
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning("LLM classifier failed for %s: %s", ticket.id, exc)
            rules.source = "rules_fallback"
            rules.notes.append(f"LLM call failed ({type(exc).__name__}); used rule classifier")
            return rules
        if label not in config.ALLOWED_INTENTS:
            rules.source = "rules_fallback"
            rules.notes.append(f"LLM returned off-label intent {label!r}; used rule classifier")
            return rules
        certainty = max(0.0, min(1.0, llm_certainty))
        notes = []
        if label != rules.intent:
            # Disagreement between two independent classifiers lowers certainty.
            certainty = round(certainty * 0.5, 4)
            notes.append(f"LLM label {label!r} disagrees with rules label {rules.intent!r}")
        return Classification(
            intent=label,
            certainty=round(certainty, 4),
            scores=rules.scores,
            matched_rules=rules.matched_rules,
            source=self.name,
            notes=notes,
        )
