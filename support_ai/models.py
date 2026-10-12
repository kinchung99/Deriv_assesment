"""Typed data passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Ticket:
    id: str
    message: str
    language: str = "en"


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.title}. {self.body}"


@dataclass
class Classification:
    intent: str
    certainty: float
    scores: Dict[str, float] = field(default_factory=dict)
    matched_rules: List[str] = field(default_factory=list)
    source: str = "rules"
    notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalHit:
    article_id: str
    title: str
    score: float


@dataclass
class RequestSignals:
    """Request-level risk signals derived from the ticket and its classification."""

    advice_request: bool = False
    account_specific: bool = False
    empty_message: bool = False
    unsupported_language: bool = False
    multi_intent: bool = False
    evidence: List[str] = field(default_factory=list)


@dataclass
class UsedSentence:
    article_id: str
    source: str  # sentence as it appears in the KB
    rendered: str  # sentence as it appears in the reply (without citation)


@dataclass
class Draft:
    reply: str
    citations: List[str]
    used_sentences: List[UsedSentence] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)  # internal guidance honoured
    source: str = "template"
    notes: List[str] = field(default_factory=list)


@dataclass
class GroundingReport:
    grounded: bool
    score: float
    citations: List[str]
    invalid_citations: List[str]
    weak_sentences: List[str]
    unsupported_framing: List[str]
    unsupported_claims: List[str]


@dataclass
class Assessment:
    confidence: float
    confidence_breakdown: Dict[str, float]
    safety_flags: List[str]
    needs_human_escalation: bool
    escalation_reasons: List[str]
    explanation: str


@dataclass
class TicketResult:
    ticket_id: str
    intent: str
    retrieved_articles: List[str]
    reply_draft: str
    grounded: bool
    confidence: float
    needs_human_escalation: bool
    safety_flags: List[str]

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "intent": self.intent,
            "retrieved_articles": list(self.retrieved_articles),
            "reply_draft": self.reply_draft,
            "grounded": self.grounded,
            "confidence": self.confidence,
            "needs_human_escalation": self.needs_human_escalation,
            "safety_flags": list(self.safety_flags),
        }


@dataclass
class TicketTrace:
    """Everything needed to explain one ticket's result (debug artifact)."""

    ticket: Ticket
    classification: Optional[Classification] = None
    signals: Optional[RequestSignals] = None
    hits: List[RetrievalHit] = field(default_factory=list)
    draft: Optional[Draft] = None
    grounding: Optional[GroundingReport] = None
    assessment: Optional[Assessment] = None
    validation_errors: List[str] = field(default_factory=list)
    validation_fixes: List[str] = field(default_factory=list)
    result: Optional[TicketResult] = None
