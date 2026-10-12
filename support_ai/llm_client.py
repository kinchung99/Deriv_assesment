"""Thin wrapper around the Anthropic SDK.

Optional: the import is lazy, and get_default_client() returns None when the
SDK or credentials are missing, which makes every LLM component fall back to
its offline counterpart.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

from . import config

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """The model call failed or returned something unusable."""


class ClaudeClient:
    def __init__(self, model: Optional[str] = None) -> None:
        import anthropic  # optional dependency

        self._client = anthropic.Anthropic(
            timeout=config.LLM_TIMEOUT_SECONDS, max_retries=config.LLM_MAX_RETRIES
        )
        self.model = model or os.environ.get(config.LLM_MODEL_ENV, config.LLM_DEFAULT_MODEL)

    def complete_json(self, system: str, user: str, schema: dict) -> dict:
        """One request constrained to `schema` via structured outputs."""
        response = self._client.beta.messages.create(
            model=self.model,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": config.LLM_EFFORT,
                "format": {"type": "json_schema", "schema": schema},
            },
            # Re-run on Anthropic's recommended model if a safety classifier declines.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise LLMError("model declined the request")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated at max_tokens")
        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("expected a JSON object")
        return data


def get_default_client() -> Optional[ClaudeClient]:
    if not any(os.environ.get(name) for name in config.LLM_CREDENTIAL_ENVS):
        log.info("no %s set; using offline components", " / ".join(config.LLM_CREDENTIAL_ENVS))
        return None
    try:
        return ClaudeClient()
    except ImportError:
        log.warning("anthropic SDK not installed (pip install -r requirements-optional.txt); using offline components")
        return None
