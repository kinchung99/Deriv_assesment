"""Central configuration: labels, schema, thresholds and paths.

Every tunable number lives here so behaviour can be adjusted (or audited)
without touching stage logic.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_TICKETS_PATH = ROOT_DIR / "tickets.json"
DEFAULT_KB_PATH = ROOT_DIR / "kb_articles.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR
RESULTS_FILENAME = "results.json"
DEBUG_JSON_FILENAME = "debug_report.json"
DEBUG_MD_FILENAME = "debug_report.md"

# --------------------------------------------------------------------------- #
# Labels and output schema
# --------------------------------------------------------------------------- #
ALLOWED_INTENTS = (
    "deposit_issue",
    "password_reset",
    "withdrawal_issue",
    "account_lock",
    "trading_advice_request",
    "other",
)
OTHER_INTENT = "other"
ADVICE_INTENT = "trading_advice_request"

# Field name -> accepted Python type(s), in the order they are written.
RESULT_SCHEMA = {
    "ticket_id": str,
    "intent": str,
    "retrieved_articles": list,
    "reply_draft": str,
    "grounded": bool,
    "confidence": (int, float),
    "needs_human_escalation": bool,
    "safety_flags": list,
}

SUPPORTED_LANGUAGES = frozenset({"en"})

# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
# Rule score at which the classifier is considered fully "covered".
CLASSIFIER_STRONG_SCORE = 3.0
# A runner-up intent scoring at least this fraction of the winner is a
# conflicting signal (the ticket plausibly mixes two issues).
CLASSIFIER_CONFLICT_RATIO = 0.6
# Any ticket whose trading-advice score reaches this is treated as an advice
# request, even if another intent wins (safety takes precedence).
ADVICE_SCORE_THRESHOLD = 3.0

# --------------------------------------------------------------------------- #
# Retrieval (BM25)
# --------------------------------------------------------------------------- #
BM25_K1 = 1.5
BM25_B = 0.75
MAX_RETRIEVED = 3
# Keep secondary hits only if they score at least this fraction of the top hit.
RELATIVE_SCORE_CUTOFF = 0.5
# Below this absolute BM25 score nothing is considered relevant.
MIN_RELEVANCE_SCORE = 1.5
# BM25 score at which retrieval strength saturates to 1.0.
RETRIEVAL_SCORE_REF = 4.0

# --------------------------------------------------------------------------- #
# Grounding, confidence and escalation
# --------------------------------------------------------------------------- #
# Min share of a cited sentence's content tokens found in its cited article(s).
GROUNDING_THRESHOLD = 0.5
# Min share of an uncited (framing) sentence's tokens that must come from the
# retrieved articles or the fixed framing vocabulary.
FRAMING_THRESHOLD = 0.6

CONFIDENCE_WEIGHTS = {
    "classifier_certainty": 0.45,
    "retrieval_strength": 0.35,
    "grounding_score": 0.20,
}
CONFLICT_PENALTY = 0.15
# Risk penalties: the draft may be accurate, but it cannot fully resolve these.
POLICY_SENSITIVE_PENALTY = 0.15
ACCOUNT_SPECIFIC_PENALTY = 0.2
LOW_CONFIDENCE_THRESHOLD = 0.5

POLICY_SENSITIVE_INTENTS = frozenset({"withdrawal_issue"})

# --------------------------------------------------------------------------- #
# Safety flags (closed vocabulary)
# --------------------------------------------------------------------------- #
FLAG_ADVICE_REQUEST = "advice_request"
FLAG_INSUFFICIENT_GROUNDING = "insufficient_grounding"
FLAG_POLICY_SENSITIVE = "policy_sensitive"
FLAG_LOW_CONFIDENCE = "low_confidence"
FLAG_ACCOUNT_SPECIFIC = "account_specific_request"
FLAG_CONFLICTING_SIGNALS = "conflicting_signals"
FLAG_UNSUPPORTED_CLAIM = "unsupported_claim"
FLAG_UNSUPPORTED_LANGUAGE = "unsupported_language"
FLAG_EMPTY_MESSAGE = "empty_message"
FLAG_VALIDATION_FAILED = "validation_failed"

# Canonical ordering so safety_flags are always emitted deterministically.
FLAG_ORDER = (
    FLAG_ADVICE_REQUEST,
    FLAG_POLICY_SENSITIVE,
    FLAG_ACCOUNT_SPECIFIC,
    FLAG_CONFLICTING_SIGNALS,
    FLAG_INSUFFICIENT_GROUNDING,
    FLAG_UNSUPPORTED_CLAIM,
    FLAG_LOW_CONFIDENCE,
    FLAG_UNSUPPORTED_LANGUAGE,
    FLAG_EMPTY_MESSAGE,
    FLAG_VALIDATION_FAILED,
)

# --------------------------------------------------------------------------- #
# Optional LLM backend (Claude). Never required: offline components are the
# default and the automatic fallback.
# --------------------------------------------------------------------------- #
LLM_CREDENTIAL_ENVS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
LLM_MODEL_ENV = "SUPPORT_AI_MODEL"
LLM_DEFAULT_MODEL = "claude-opus-5-5"
LLM_EFFORT = "low"  # short classification / drafting tasks
LLM_MAX_TOKENS = 4096  # headroom for adaptive thinking + a short JSON answer
LLM_TIMEOUT_SECONDS = 60.0
LLM_MAX_RETRIES = 2
