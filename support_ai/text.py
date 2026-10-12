"""Text normalisation shared by retrieval and grounding checks."""
from __future__ import annotations

import re
from typing import List

# Multi-word variants collapsed to one canonical token before tokenising,
# so "sign-in", "sign in", "log in" and "login" all match each other.
_PHRASE_NORMALISATION = (
    (re.compile(r"\bsign[\s-]?in\b"), "login"),
    (re.compile(r"\blog[\s-]?in\b"), "login"),
    (re.compile(r"\be-mail\b"), "email"),
    (re.compile(r"\bcash[\s-]?out\b"), "withdrawal"),
    (re.compile(r"\btop[\s-]?up\b"), "deposit"),
)

# Query-side expansion: user vocabulary -> KB vocabulary.
_QUERY_SYNONYMS = {
    "payout": ["withdrawal"],
    "stock": ["asset"],
    "share": ["asset"],
    "crypto": ["asset"],
    "coin": ["asset"],
    "invest": ["investment"],
    "lockout": ["lock"],
    "blocked": ["lock"],
    "fund": ["deposit"],
    "buy": ["trading", "investment"],
    "sell": ["trading", "investment"],
    "trade": ["trading"],
    "predict": ["prediction"],
    "forecast": ["prediction"],
    "recommend": ["recommendation"],
    "password": ["reset"],
    "forgot": ["reset"],
}

STOPWORDS = frozenset(
    """
    a about after again all also am an and any are as at be been before but by
    can could did do does doing done for from get got had has have having he her
    his how i if in into is it its just me might more most much my no nor not now
    of off on once only or other our out over own please same she should so some
    still such than that the their them then there these they this those through
    to too until up us very was we were what when where which while who why will
    with would you your yours yet today many may tell
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    text = text.lower()
    for pattern, replacement in _PHRASE_NORMALISATION:
        text = pattern.sub(replacement, text)
    return text


def stem(word: str) -> str:
    """Tiny suffix stripper: enough to match deposit/deposited/deposits."""
    if len(word) <= 3 or word.isdigit():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suffix in ("ing", "ed"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def tokenize(text: str) -> List[str]:
    """Lowercase, normalise phrases, drop stopwords, stem."""
    return [stem(tok) for tok in _TOKEN_RE.findall(normalise(text)) if tok not in STOPWORDS]


def expand_query(tokens: List[str]) -> List[str]:
    """Append KB-vocabulary synonyms for (already stemmed) query tokens."""
    expanded = list(tokens)
    for tok in tokens:
        expanded.extend(_STEMMED_SYNONYMS.get(tok, ()))
    return expanded


_STEMMED_SYNONYMS = {
    stem(word): [stem(s) for s in synonyms] for word, synonyms in _QUERY_SYNONYMS.items()
}

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.split(text.strip()) if s.strip()]
