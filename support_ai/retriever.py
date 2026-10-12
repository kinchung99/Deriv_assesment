"""INDEX_KB + RETRIEVE_CONTEXT stages: deterministic BM25 over the KB.

Retrieval only ever sees the ticket text, never the predicted intent, so the
two signals stay independent and can be cross-checked later.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Sequence

from . import config
from .models import Article, RetrievalHit
from .text import expand_query, tokenize


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        k1: float = config.BM25_K1,
        b: float = config.BM25_B,
        min_score: float = config.MIN_RELEVANCE_SCORE,
        relative_cutoff: float = config.RELATIVE_SCORE_CUTOFF,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.min_score = min_score
        self.relative_cutoff = relative_cutoff
        self._articles: List[Article] = []
        self._term_freqs: List[Counter] = []
        self._doc_lens: List[int] = []
        self._idf: Dict[str, float] = {}
        self._avgdl = 0.0

    def index(self, articles: Sequence[Article]) -> None:
        self._articles = list(articles)
        self._term_freqs = [Counter(tokenize(a.text)) for a in self._articles]
        self._doc_lens = [sum(tf.values()) for tf in self._term_freqs]
        self._avgdl = sum(self._doc_lens) / max(len(self._doc_lens), 1)
        n_docs = len(self._articles)
        doc_freq: Counter = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        # BM25+ style idf: always positive, even for terms in most documents.
        self._idf = {
            term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def score_all(self, query: str) -> List[RetrievalHit]:
        """Score every article (used for debugging and by retrieve())."""
        if not self._articles:
            raise RuntimeError("index() must be called before retrieval")
        terms = expand_query(tokenize(query))
        hits = []
        for article, tf, dl in zip(self._articles, self._term_freqs, self._doc_lens):
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = freq + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += self._idf[term] * freq * (self.k1 + 1) / norm
            hits.append(RetrievalHit(article.article_id, article.title, round(score, 4)))
        # Deterministic: score desc, then article id asc to break ties.
        hits.sort(key=lambda h: (-h.score, h.article_id))
        return hits

    def retrieve(self, query: str, k: int = config.MAX_RETRIEVED) -> List[RetrievalHit]:
        ranked = self.score_all(query)
        if ranked[0].score < self.min_score:
            # Always return at least the best match (spec: top 1-3), marked not
            # relevant so it is never used as grounding for the reply.
            top = ranked[0]
            return [RetrievalHit(top.article_id, top.title, top.score, relevant=False)]
        floor = max(self.min_score, ranked[0].score * self.relative_cutoff)
        return [h for h in ranked[: max(1, min(k, config.MAX_RETRIEVED))] if h.score >= floor]
