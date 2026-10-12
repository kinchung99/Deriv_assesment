"""Fixed, fact-free framing text used around KB-derived content.

These strings never carry product facts; all substantive content in a reply
comes from retrieved KB sentences (with citations). Their vocabulary is also
the allow-list for uncited "framing" sentences in the grounding check.
"""
from __future__ import annotations

from typing import FrozenSet

from .text import tokenize

OPENERS = {
    "deposit_issue": "Thanks for reaching out about your deposit.",
    "password_reset": "Sorry to hear you're having trouble resetting your password.",
    "withdrawal_issue": "Thanks for reaching out about your withdrawal.",
    "account_lock": "Sorry for the trouble signing in to your account.",
    "trading_advice_request": "Thanks for your question.",
    "other": "Thanks for contacting us.",
}

CLOSER_SELF_SERVE = (
    "If the issue continues after these steps, just reply to this message and "
    "we'll be happy to help further."
)
CLOSER_AGENT_REVIEW = (
    "Because the specific reason depends on details of your account, a member of our "
    "support team will review your case and follow up with you. We can't confirm "
    "outcomes or timelines until that review is done."
)
CLOSER_NO_CONTEXT = (
    "A member of our support team will review your request and get back to you."
)
EMPTY_MESSAGE_REPLY = (
    "It looks like your message was empty. Could you tell us a bit more about the "
    "issue you're experiencing so we can help?"
)
# Used only when an advice request is detected but no policy article was
# retrieved; it is uncited, so the result is marked ungrounded and escalated.
REFUSAL_FALLBACK = (
    "I'm sorry, but we can't provide investment or trading advice, market "
    "predictions, or asset recommendations."
)
REFUSAL_TEMPLATE = "I'm sorry, but we're not able to provide {restricted}"
REDIRECT_TEMPLATE = "You may find general educational resources helpful, where available"
SAFE_FALLBACK_REPLY = (
    "Thank you for contacting us. A member of our support team will review your "
    "request and get back to you."
)

# Generic customer-service words any reply may use without a citation.
_EXTRA_FRAMING_WORDS = """
thanks thank hello hi sorry trouble help happy glad assist assistance reach reaching
contact contacting question request reply message team member support follow
review case understand appreciate patience frustrating inconvenience apologise
apologize issue problem steps details information account
"""


def _framing_vocabulary() -> FrozenSet[str]:
    texts = list(OPENERS.values()) + [
        CLOSER_SELF_SERVE,
        CLOSER_AGENT_REVIEW,
        CLOSER_NO_CONTEXT,
        EMPTY_MESSAGE_REPLY,
        REFUSAL_FALLBACK,
        SAFE_FALLBACK_REPLY,
        _EXTRA_FRAMING_WORDS,
    ]
    vocab = set()
    for text in texts:
        vocab.update(tokenize(text))
    return frozenset(vocab)


FRAMING_VOCAB = _framing_vocabulary()
