"""Swap boundaries.

The pipeline depends only on these protocols. Any implementation with the
same methods (rules or LLM classifier, BM25 or embedding retriever, template
or hosted-model generator) can be dropped in without touching other stages.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

try:  # Python 3.8+
    from typing import Protocol
except ImportError:  # pragma: no cover
    from typing_extensions import Protocol  # type: ignore

from .models import Article, Classification, Draft, RequestSignals, RetrievalHit, Ticket


class Classifier(Protocol):
    name: str

    def classify(self, ticket: Ticket) -> Classification:
        """Return one of config.ALLOWED_INTENTS plus a certainty in [0, 1]."""


class Retriever(Protocol):
    name: str

    def index(self, articles: Sequence[Article]) -> None:
        """Build the search index. Must be called before retrieve()."""

    def retrieve(self, query: str, k: int) -> List[RetrievalHit]:
        """Return 1..k hits, deterministically ordered; weak matches have relevant=False."""


class Generator(Protocol):
    name: str

    def generate(
        self,
        ticket: Ticket,
        classification: Classification,
        signals: RequestSignals,
        hits: Sequence[RetrievalHit],
        kb: Dict[str, Article],
        article_topics: Dict[str, str],
    ) -> Draft:
        """Draft a reply using only the retrieved articles."""
