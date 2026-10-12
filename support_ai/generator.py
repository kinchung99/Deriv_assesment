"""GENERATE_REPLY stage.

TemplateGenerator (default, offline) builds the reply *extractively*: every
substantive sentence is a KB sentence, lightly rewritten into second person
and followed by its citation. Nothing is hardcoded per ticket.

LLMGenerator (optional) asks a hosted model to write the reply from the same
retrieved snippets and falls back to TemplateGenerator on any failure.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import config, templates
from .grounding import analyze_grounding, extract_citations
from .models import (
    Article,
    Classification,
    Draft,
    RequestSignals,
    RetrievalHit,
    Ticket,
    UsedSentence,
)
from .policy import is_policy_sensitive, runner_up
from .text import split_sentences

log = logging.getLogger(__name__)

# KB sentences addressed to support staff, not customers. They are never
# copied into a reply; they become constraints the draft must honour.
_INTERNAL_GUIDANCE_RE = re.compile(
    r"^(support|support agents?|agents?|staff)\b|^such requests\b|^avoid\b", re.IGNORECASE
)
# Agent-facing phrasing -> customer-facing phrasing (applied in order).
_REWRITES: Tuple[Tuple[re.Pattern, str], ...] = tuple(
    (re.compile(p, re.IGNORECASE), r)
    for p, r in (
        (r"^(ask|advise|tell|remind) the client to ", "Please "),
        (r"^if the client does not ([^,]+), ", r"If you do not \1, please "),
        (r"\bthe client's\b", "your"),
        (r"\bthe client\b", "you"),
        (r"\bthe registered\b", "your registered"),
        (r"\bthe account\b", "your account"),
        (r"\bbefore escalating\b", "before contacting us again"),
    )
)
_RESTRICTION_RE = re.compile(
    r"\b(?:must|should) not (?:provide|give|offer|share) (?P<restricted>[^.]+)", re.IGNORECASE
)
_REDIRECT_RE = re.compile(r"\beducational resources\b", re.IGNORECASE)


def is_internal_guidance(sentence: str) -> bool:
    return bool(_INTERNAL_GUIDANCE_RE.search(sentence.strip()))


def to_customer_voice(sentence: str) -> str:
    out = sentence.strip()
    for pattern, replacement in _REWRITES:
        out = pattern.sub(replacement, out)
    return out[:1].upper() + out[1:]


def cite(sentence: str, article_id: str) -> str:
    """Insert ' [A1]' before the sentence's final punctuation."""
    sentence = sentence.rstrip()
    if sentence and sentence[-1] in ".!?":
        return f"{sentence[:-1]} [{article_id}]{sentence[-1]}"
    return f"{sentence} [{article_id}]."


class TemplateGenerator:
    name = "template"

    def generate(
        self,
        ticket: Ticket,
        classification: Classification,
        signals: RequestSignals,
        hits: Sequence[RetrievalHit],
        kb: Dict[str, Article],
        article_topics: Dict[str, str],
    ) -> Draft:
        if signals.empty_message:
            return Draft(reply=templates.EMPTY_MESSAGE_REPLY, citations=[], source=self.name)

        intent = classification.intent
        used: List[UsedSentence] = []
        constraints: List[str] = []
        parts: List[str] = [templates.OPENERS.get(intent, templates.OPENERS["other"])]

        # 1) Informational content for the non-advice part of the request.
        topics = [] if intent == config.ADVICE_INTENT else [intent]
        second = runner_up(classification)
        if signals.multi_intent and second and second != config.ADVICE_INTENT:
            topics.append(second)
        for article in self._select_articles(topics, hits, kb, article_topics, signals):
            for sentence in split_sentences(article.body):
                if is_internal_guidance(sentence):
                    constraints.append(f"[{article.article_id}] {sentence}")
                    continue
                rendered = to_customer_voice(sentence)
                used.append(UsedSentence(article.article_id, sentence, rendered))
                parts.append(cite(rendered, article.article_id))

        # 2) Polite refusal for any trading-advice component.
        if signals.advice_request:
            parts.extend(self._refusal(hits, kb, article_topics, used, constraints))

        # 3) Fixed closer, consistent with how policy will escalate.
        cited_ids = sorted({u.article_id for u in used})
        if not used:
            parts.append(templates.CLOSER_NO_CONTEXT)
        elif signals.account_specific or is_policy_sensitive(intent, cited_ids, kb):
            parts.append(templates.CLOSER_AGENT_REVIEW)
        elif intent != config.ADVICE_INTENT:
            parts.append(templates.CLOSER_SELF_SERVE)

        reply = " ".join(parts)
        return Draft(
            reply=reply,
            citations=extract_citations(reply),
            used_sentences=used,
            constraints=constraints,
            source=self.name,
        )

    @staticmethod
    def _select_articles(
        topics: List[str],
        hits: Sequence[RetrievalHit],
        kb: Dict[str, Article],
        article_topics: Dict[str, str],
        signals: RequestSignals,
    ) -> List[Article]:
        if not topics or not hits:
            return []
        on_topic = [kb[h.article_id] for h in hits if article_topics.get(h.article_id) in topics]
        if on_topic:
            return on_topic
        top = hits[0]
        # Off-topic top hit: still the best evidence we have, unless it is the
        # advice policy article (handled by the refusal path instead).
        if article_topics.get(top.article_id) == config.ADVICE_INTENT:
            return []
        return [kb[top.article_id]]

    @staticmethod
    def _refusal(
        hits: Sequence[RetrievalHit],
        kb: Dict[str, Article],
        article_topics: Dict[str, str],
        used: List[UsedSentence],
        constraints: List[str],
    ) -> List[str]:
        policy_articles = [
            kb[h.article_id] for h in hits if article_topics.get(h.article_id) == config.ADVICE_INTENT
        ]
        for article in policy_articles:
            match = _RESTRICTION_RE.search(article.body)
            if not match:
                continue
            restricted = match.group("restricted").strip().rstrip(".")
            refusal = templates.REFUSAL_TEMPLATE.format(restricted=restricted) + "."
            parts = [cite(refusal, article.article_id)]
            used.append(UsedSentence(article.article_id, match.group(0), refusal))
            if _REDIRECT_RE.search(article.body):
                redirect = templates.REDIRECT_TEMPLATE + "."
                parts.append(cite(redirect, article.article_id))
                used.append(UsedSentence(article.article_id, "educational resources", redirect))
            for sentence in split_sentences(article.body):
                constraints.append(f"[{article.article_id}] {sentence}")
            return parts
        # No policy article retrieved: refuse anyway, uncited (=> escalated).
        return [templates.REFUSAL_FALLBACK]


class LLMGenerator:
    """Hosted-model generator with a deterministic safety net.

    The model sees only the retrieved snippets. Its draft is accepted only if
    it cites retrieved articles, passes the grounding check and (for advice
    requests) is a safe refusal; otherwise TemplateGenerator's draft is used.
    """

    name = "llm"

    def __init__(self, client=None, fallback: Optional[TemplateGenerator] = None) -> None:
        from .llm_client import get_default_client

        self.client = client if client is not None else get_default_client()
        self.fallback = fallback or TemplateGenerator()

    def generate(
        self,
        ticket: Ticket,
        classification: Classification,
        signals: RequestSignals,
        hits: Sequence[RetrievalHit],
        kb: Dict[str, Article],
        article_topics: Dict[str, str],
    ) -> Draft:
        def fallback(reason: str) -> Draft:
            draft = self.fallback.generate(ticket, classification, signals, hits, kb, article_topics)
            draft.source = "template_fallback"
            draft.notes.append(f"llm_fallback: {reason}")
            return draft

        if self.client is None:
            return fallback("LLM unavailable (no credentials or SDK)")
        if signals.empty_message or not hits:
            return fallback("nothing to ground an LLM reply on")

        from .prompts import build_generator_prompt
        from .validator import is_safe_refusal

        system, user, schema = build_generator_prompt(
            ticket.message, classification.intent, signals, hits, kb
        )
        try:
            data = self.client.complete_json(system=system, user=user, schema=schema)
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            log.warning("LLM generator failed for %s: %s", ticket.id, exc)
            return fallback(f"LLM call failed ({type(exc).__name__})")

        reply = " ".join(str(data.get("reply", "")).split())
        retrieved = [h.article_id for h in hits]
        if not reply:
            return fallback("empty reply")
        if not data.get("answerable", True):
            return fallback("model reported the snippets do not answer the question")
        grounding = analyze_grounding(reply, retrieved, kb)
        if not grounding.grounded:
            problems = (
                grounding.invalid_citations
                + grounding.weak_sentences
                + grounding.unsupported_framing
                + grounding.unsupported_claims
            )
            return fallback(f"reply failed grounding check: {problems or 'no citations'}")
        if signals.advice_request and not is_safe_refusal(reply):
            return fallback("advice request was not safely refused")
        return Draft(reply=reply, citations=extract_citations(reply), source=self.name)
