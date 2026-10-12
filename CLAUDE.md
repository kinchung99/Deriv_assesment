# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A take-home AI Engineer assessment: a pipeline that classifies support tickets, retrieves help-center articles with BM25, and drafts grounded replies with confidence, safety flags and escalation. The spec requires all of the following:
- Runnable from a clean checkout, without secrets.
- Retrieval kept separate from generation.
- No logic keyed on ticket IDs. Graders may swap in new tickets.
- No hardcoded per-ticket answers.
- Trading-advice requests are always refused.

## Commands

```bash
python3 main.py              # run pipeline -> results.json, debug_report.json, debug_report.md (repo root)
python3 main.py -q --tickets other.json --out-dir /tmp/out   # custom input/output
python3 main.py --classifier llm --generator llm             # optional Claude backend (falls back offline)
python3 validate.py          # audits outputs; non-zero exit on failure
python3 -m unittest discover -s tests                        # full suite (pytest also works)
python3 -m unittest discover -s tests -p 'test_validator.py' -k test_unsupported_promise   # single test
python3 -m pytest tests/test_validator.py::SemanticTest::test_unsupported_promise_triggers_flag_and_escalation
```

There is no build or lint step. The core uses only the standard library and must stay compatible with **Python 3.9**: use `from __future__ import annotations` and `typing.List`/`Dict`, and no `match` statements. `anthropic` is optional. It is imported lazily in `support_ai/llm_client.py`, and the current SDK needs Python 3.10+.

**After any behaviour change, re-run `python3 main.py`.** `results.json` and the debug reports are committed outputs. `validate.py` checks that the on-disk `results.json` is byte-identical to a fresh offline run.

## Architecture

`support_ai/pipeline.py` runs the stages in batch: LOAD_DATA → INDEX_KB → CLASSIFY_INTENT → RETRIEVE_CONTEXT → GENERATE_REPLY → VALIDATE_OUTPUT → WRITE_RESULTS. It depends only on the `Classifier` / `Retriever` / `Generator` protocols in `interfaces.py`. `build_pipeline()` maps CLI names to implementations, so a new backend is one class plus one factory entry.

Cross-file behaviour that isn't obvious from any single module:

- **Classifier and retriever are kept independent.** Retrieval sees only the ticket text, never the predicted intent.
- **Retrieval always returns 1–3 hits; the spec requires it.** A best match below `MIN_RELEVANCE_SCORE` comes back with `relevant=False`. The pipeline stores all hits in `TicketTrace.hits` (written to `retrieved_articles`) and passes only `TicketTrace.context`, the relevant hits, to the generator and policy.
- **Article topics are derived, not hardcoded.** INDEX_KB tags each KB article with an intent by running `RuleClassifier` on the article's own text. `policy.assess` compares that tag with the ticket's intent to raise `conflicting_signals`. As a result, editing `INTENT_RULES` in `classifier.py` also changes article topics and conflict detection.
- **Policy and grounding run inside VALIDATE_OUTPUT.** For each ticket, the steps are:
  1. `grounding.analyze_grounding(reply, retrieved, kb)`
  2. `policy.assess(...)`, which sets confidence, flags and escalation reasons
  3. `validator.enforce(record, kb)`

  `enforce` repairs problems rather than only reporting them. It replaces a record that fails hard schema checks with a safe escalated fallback. For semantic failures it adds flags, escalates, or swaps in the standard refusal. It decides which fix to apply by matching the **prefix of the error strings** from `semantic_errors`, so if you change those messages, update `enforce` too.
- **Generator and policy must agree on escalation.** `TemplateGenerator` picks its closing sentence ("a member of our team will review…") using the same helpers policy uses to escalate: `policy.analyze_request` signals and `policy.is_policy_sensitive`. Keep them in sync so the reply never contradicts `needs_human_escalation`.
- **Grounding is lexical and drives several components.**
  - Cited sentences (`[A1]`) must share at least `GROUNDING_THRESHOLD` of their tokens with the cited article.
  - Uncited sentences may only use retrieved-article tokens plus `templates.FRAMING_VOCAB`, which is built automatically by tokenizing the strings in `templates.py`.
  - New fixed reply text belongs in `templates.py`, or it will fail the framing check.
  - `grounding.CLAIM_PATTERNS` (promises, timeframes, refunds, account facts) is ignored only when the matched words appear in the retrieved KB text.
  - The policy step, the validator and `LLMGenerator` all use these same functions.
- **Template generation is extractive.** KB sentences addressed to staff (`^Support…`, `^Such requests…`, `^Avoid…`) are never copied into replies; they are recorded as constraints. Customer-facing sentences are reworded by the regexes in `generator._REWRITES`. A trading refusal is built from the policy article's "must not provide X" sentence.
- **LLM components always have a fallback.** `LLMClassifier` uses an enum-constrained structured output and falls back to `RuleClassifier` on errors or off-label output. Even in LLM mode, the rule scores feed the advice-request signal. `LLMGenerator` falls back to `TemplateGenerator` if:
  - there is no client,
  - the call raises an error,
  - the model says the snippets can't answer the question,
  - the reply fails grounding,
  - an advice request isn't refused.

  Each fallback is recorded in `Draft.notes` and `source`.
- **Determinism matters.**
  - Hits are sorted by `(-score, article_id)`.
  - Classifier ties break by `ALLOWED_INTENTS` order.
  - Flags are emitted in `config.FLAG_ORDER`.
  - Written files contain no timestamps.
- **Configuration lives in one place.** All thresholds, weights, flag names and LLM settings are in `support_ai/config.py`.

## Tests

- `tests/_support.run_on_messages([...], **build_pipeline_kwargs)` runs the full pipeline on ad-hoc tickets against the real KB without writing files. Use it for edge-case tests.
- LLM tests inject a `StubClient` with a `complete_json(system, user, schema)` method through `build_pipeline(..., llm_client=stub)`. No network or SDK is needed.
- `conftest.py` at the repo root exists only so that plain `pytest` can import `support_ai`.
