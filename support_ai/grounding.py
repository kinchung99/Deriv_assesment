"""Grounding and unsupported-claim analysis of a reply draft.

Shared by the policy step (to score confidence) and the validator (to enforce
flags/escalation), so both always agree on what "grounded" means.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Sequence, Set

from . import config
from .models import Article, GroundingReport
from .templates import FRAMING_VOCAB
from .text import split_sentences, tokenize

CITATION_RE = re.compile(r"\s*\[([A-Za-z0-9_-]+)\]")

# Phrases that promise outcomes or assert account facts. A match is allowed
# only when every content word of the match appears in the retrieved KB text.
CLAIM_PATTERNS = (
    (r"\bwill (definitely |certainly |surely )?be (approved|processed|credited|completed|"
     r"released|unlocked|refunded|resolved|reversed|restored)\b", "outcome promise"),
    (r"\bguarantee[sd]?\b", "guarantee"),
    (r"\b(within|in under|in less than) (\d+|a few|one|two|three|24|48) "
     r"(minutes?|hours?|days?|business days?|weeks?)\b", "specific timeframe"),
    (r"\b\d+\s*(minutes?|hours?|days?|business days?|weeks?)\b", "specific timeframe"),
    (r"\byour (balance|account|withdrawal|deposit|payment|transaction|card)( \w+)? "
     r"(is|was|has been|shows|appears|got)\b", "account-specific fact"),
    (r"\bwe(?: have|'ve)? (checked|verified|reviewed|confirmed|located|found) your\b",
     "claimed account check"),
    (r"\b(refund\w*|reimburs\w*|compensat\w*)\b", "refund or compensation"),
    (r"\b(definitely|certainly|100%|for sure|rest assured)\b", "certainty claim"),
)
_COMPILED_CLAIMS = [(re.compile(p, re.IGNORECASE), label) for p, label in CLAIM_PATTERNS]


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text)


def extract_citations(text: str) -> List[str]:
    seen: List[str] = []
    for cid in CITATION_RE.findall(text):
        if cid not in seen:
            seen.append(cid)
    return seen


def _article_tokens(article_ids: Iterable[str], kb: Dict[str, Article]) -> Set[str]:
    tokens: Set[str] = set()
    for aid in article_ids:
        if aid in kb:
            tokens.update(tokenize(kb[aid].text))
    return tokens


def find_unsupported_claims(reply: str, source_text: str) -> List[str]:
    """Return claim descriptions in `reply` not backed by `source_text`."""
    source_tokens = set(tokenize(source_text))
    found: List[str] = []
    for pattern, label in _COMPILED_CLAIMS:
        for match in pattern.finditer(strip_citations(reply)):
            span_tokens = set(tokenize(match.group(0)))
            if span_tokens and span_tokens <= source_tokens:
                continue  # e.g. "profit guarantees" quoted from a policy article
            found.append(f"{label}: '{match.group(0)}'")
    return sorted(set(found))


def analyze_grounding(
    reply: str, retrieved_ids: Sequence[str], kb: Dict[str, Article]
) -> GroundingReport:
    retrieved = [aid for aid in retrieved_ids if aid in kb]
    retrieved_tokens = _article_tokens(retrieved, kb)
    citations = extract_citations(reply)
    invalid = [c for c in citations if c not in retrieved]

    supports: List[float] = []
    weak: List[str] = []
    framing_issues: List[str] = []
    for sentence in split_sentences(reply):
        cited = extract_citations(sentence)
        tokens = set(tokenize(strip_citations(sentence)))
        if not tokens:
            continue
        if cited:
            source = _article_tokens([c for c in cited if c in retrieved], kb)
            support = len(tokens & source) / len(tokens)
            supports.append(support)
            if support < config.GROUNDING_THRESHOLD:
                weak.append(strip_citations(sentence))
        else:
            allowed = retrieved_tokens | FRAMING_VOCAB
            if len(tokens & allowed) / len(tokens) < config.FRAMING_THRESHOLD:
                framing_issues.append(sentence)

    source_text = " ".join(kb[aid].text for aid in retrieved)
    claims = find_unsupported_claims(reply, source_text)
    score = round(sum(supports) / len(supports), 4) if supports else 0.0
    grounded = bool(supports) and not invalid and not weak and not framing_issues and not claims
    return GroundingReport(
        grounded=grounded,
        score=score,
        citations=citations,
        invalid_citations=invalid,
        weak_sentences=weak,
        unsupported_framing=framing_issues,
        unsupported_claims=claims,
    )
