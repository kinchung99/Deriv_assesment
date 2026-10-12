# Support Ticket Triage: grounded reply drafts

A small, runnable AI service that takes customer-support tickets and, for each one:

1. **classifies the intent** into a fixed set of labels,
2. **retrieves** the most relevant help-center articles, and
3. **drafts a reply grounded in those articles**, with a confidence score, safety flags and an escalation signal.

The default path uses only the Python standard library and gives the same output on every run, with no API key needed. An optional Claude-backed classifier and generator use the same interfaces and fall back to the offline components automatically.

## Quick start

Requires Python 3.9 or later. No `pip install` is needed.

```bash
python main.py        # run the pipeline   -> results.json, debug_report.json, debug_report.md
python validate.py    # validate outputs   -> PASS/FAIL per check, non-zero exit on failure
python -m unittest discover -s tests -v   # 45 unit/integration tests (pytest also works)
```

`main.py` options:

| Flag | Default | Purpose |
|---|---|---|
| `--tickets PATH` | `tickets.json` | tickets to process (same schema; any ids) |
| `--kb PATH` | `kb_articles.json` | knowledge base |
| `--out-dir DIR` | repo root | where outputs are written |
| `--classifier rules\|llm` | `rules` | swap the intent classifier |
| `--generator template\|llm` | `template` | swap the reply generator |
| `-q` | off | hide the per-stage log |

### Optional: Claude backend

```bash
pip install -r requirements-optional.txt   # anthropic SDK (needs Python 3.10+), pytest
export ANTHROPIC_API_KEY=...
python main.py --classifier llm --generator llm
```

The default model is `claude-opus-5-5`; you can override it with `SUPPORT_AI_MODEL`. If no credentials are set, the SDK isn't installed, a call fails, or the model's output fails validation, each ticket falls back to the offline component. `debug_report.json` records when that happened. The LLM path has been tested with a stubbed client only; it has not been run against the live API.

## Output

`results.json` contains one record per ticket, in input order, with exactly the required fields:

```json
{
  "ticket_id": "T3",
  "intent": "withdrawal_issue",
  "retrieved_articles": ["A3"],
  "reply_draft": "Thanks for reaching out about your withdrawal. Withdrawals may be declined due to incomplete verification, mismatched payment method details, compliance review, or account restrictions [A3]. Because the specific reason depends on details of your account, a member of our support team will review your case and follow up with you. We can't confirm outcomes or timelines until that review is done.",
  "grounded": true,
  "confidence": 0.65,
  "needs_human_escalation": true,
  "safety_flags": ["policy_sensitive", "account_specific_request"]
}
```

Results on the sample data:

| Ticket | Intent | Articles | Confidence | Escalate | Flags |
|---|---|---|---|---|---|
| T1 | deposit_issue | A1 | 0.99 | no | — |
| T2 | password_reset | A2 | 1.00 | no | — |
| T3 | withdrawal_issue | A3 | 0.65 | **yes** | policy_sensitive, account_specific_request |
| T4 | trading_advice_request | A5 | 0.84 | no (politely refused) | advice_request |
| T5 | account_lock | A4 | 1.00 | no | — |

The explainability artifacts are `debug_report.md` (human-readable) and `debug_report.json` (machine-readable). For each ticket they show:
- the predicted intent, with per-label rule scores,
- the retrieved article titles and BM25 scores,
- the KB sentences used in the reply,
- the staff-only KB guidance that was applied,
- the confidence breakdown,
- why the ticket was or wasn't escalated.

## Architecture

```
LOAD_DATA → INDEX_KB → CLASSIFY_INTENT → RETRIEVE_CONTEXT → GENERATE_REPLY → VALIDATE_OUTPUT → WRITE_RESULTS
 loader      BM25        «Classifier»       «Retriever»        «Generator»      grounding +       writer
             + topic     rules | llm        bm25               template | llm   policy +
             tagging                                                            validator
```

Each stage is a separate module and passes typed dataclasses to the next. The pipeline ([support_ai/pipeline.py](support_ai/pipeline.py)) depends only on the `Classifier`, `Retriever` and `Generator` protocols in [support_ai/interfaces.py](support_ai/interfaces.py). To swap in an embedding retriever or a hosted model, you write one class and register it in `build_pipeline()`.

| Module | Stage | What it does |
|---|---|---|
| [loader.py](support_ai/loader.py) | LOAD_DATA | Reads both JSON files. Checks that required keys exist and IDs are unique, and normalises whitespace. |
| [retriever.py](support_ai/retriever.py) | INDEX_KB, RETRIEVE_CONTEXT | BM25 over title and body, in pure Python. The query is the **ticket text only**. Results are sorted by `(-score, article_id)` so ties always break the same way. It returns the top hit plus any hit scoring at least 50% of it (at most 3), or nothing if the top score is below the relevance threshold. |
| [text.py](support_ai/text.py) | shared | Tokeniser, stopwords, light stemmer, phrase normalisation (`sign-in`/`log in` → `login`) and a small query-synonym map. |
| [classifier.py](support_ai/classifier.py) | CLASSIFY_INTENT | `RuleClassifier`: weighted regex rules per intent. Certainty = (top score's share of top + runner-up) × coverage, where coverage is how close the top score is to a "strong" score. `LLMClassifier`: a structured-output enum restricted to the allowed labels, falling back to the rules. |
| [policy.py](support_ai/policy.py) | between classify and validate | Request signals (advice request, account-specific question, mixed intents, language, empty message), the confidence formula, safety flags and escalation reasons. |
| [generator.py](support_ai/generator.py) | GENERATE_REPLY | `TemplateGenerator` builds an extractive reply from KB sentences (details below). `LLMGenerator` sends a prompt containing only the retrieved snippets, then puts the model's draft through the same grounding and safety checks. |
| [grounding.py](support_ai/grounding.py) | VALIDATE_OUTPUT | Scores how much of each cited sentence appears in the article it cites. Uncited framing sentences may only use the retrieved articles' vocabulary or a fixed template vocabulary. Also detects unsupported claims (promises, timeframes, refunds, invented account facts). |
| [validator.py](support_ai/validator.py) | VALIDATE_OUTPUT | Hard schema checks: a failing record is replaced with a safe escalated fallback. Semantic checks: cited articles were actually retrieved, advice requests are refused, unsupported claims are flagged or escalated, and risk flags imply escalation. Violations are repaired, not just reported. |
| [writer.py](support_ai/writer.py) | WRITE_RESULTS | Writes `results.json` and the debug reports. Nothing run-specific (no timestamps), so output is byte-for-byte reproducible. |

### How replies stay grounded

`TemplateGenerator` works without an LLM and doesn't hardcode any answer:

1. Each retrieved article is split into sentences.
2. Sentences addressed to support staff are kept out of the reply. These are lines like "Support should not promise approval times…" or "Support agents must not provide…". They are applied as constraints instead, and listed in the debug report.
3. The customer-facing sentences are reworded into second person with a small set of fixed rewrites ("Ask the client to confirm…" → "Please confirm…"). Each one is followed by its citation, e.g. `[A1]`.
4. The reply opens and closes with fixed text that states no facts. The closer matches the escalation decision: it says "a member of our team will review your case" exactly when the ticket will be escalated.
5. For a trading-advice request, the refusal comes from the policy article's own wording: "we're not able to provide market predictions, profit guarantees, or asset recommendations [A5]".

Every factual sentence can therefore be traced to a KB sentence. The grounding check confirms this, and `validate.py` checks it again.

### Confidence and escalation

```
confidence = 0.45·classifier_certainty + 0.35·retrieval_strength + 0.20·grounding_score
             − 0.15 if conflicting signals  − 0.15 if policy-sensitive  − 0.20 if account-specific
```

`retrieval_strength` is the top BM25 score ÷ 4.0, capped at 1. The result is clamped to [0, 1]. All weights and thresholds are in [support_ai/config.py](support_ai/config.py).

| Flag | Raised when | Escalates? |
|---|---|---|
| `advice_request` | asks for predictions, what to buy, or how to profit (even inside another question) | no, because the polite refusal fully handles it |
| `policy_sensitive` | withdrawal topic, or the KB article says outcomes need a compliance/status check | yes |
| `account_specific_request` | e.g. "why was *my* withdrawal declined", "when will *my* X be approved" | yes |
| `conflicting_signals` | classifier and top article disagree, or two intents score highly | yes |
| `insufficient_grounding` | no relevant article, or a reply sentence isn't supported by what it cites | yes |
| `unsupported_claim` | the reply promises outcomes or timeframes, mentions refunds, or asserts account facts that aren't in the KB | yes |
| `low_confidence` | confidence < 0.5 | yes |
| `unsupported_language` / `empty_message` | input outside what the KB covers | yes |
| `validation_failed` | the record failed hard validation and was replaced | yes |

## Design choices and trade-offs

- **Rules + BM25 instead of embeddings.** With 5 articles and 6 labels, transparent rules and BM25 are deterministic, explainable in the debug report, and run anywhere. The protocols make an embedding retriever or an LLM classifier a drop-in replacement.
- **Extractive generation by default.** Grounding and traceability come from how the reply is built, not from hoping the model follows instructions. The LLM generator is held to the same checks.
- **Safety over completeness.** If anything is uncertain (no relevant article, mixed intents, account-specific questions), the ticket is escalated instead of answered with a confident guess. Trading advice is always refused, even when the request is buried in another question or phrased as a prompt injection.
- **`retrieved_articles` may be empty.** For gibberish or off-topic tickets, nothing clears the relevance threshold. Returning an unrelated article would be misleading, so the list is empty and the ticket is flagged `insufficient_grounding` and escalated.
- **No ticket-ID logic.** Nothing in the pipeline depends on IDs; tests run paraphrased tickets with new IDs. The only ID-specific checks are the T1 and T4 sample checks in `validate.py`, which the spec requires.

## Known limitations and next steps

- The rule classifier and synonym map are tuned for this domain's vocabulary. A labelled evaluation set, and calibrating the confidence score against it, would be the next step.
- English only. Other languages are flagged and escalated rather than translated.
- The grounding check measures word overlap, so it can miss a paraphrased hallucination that reuses KB vocabulary. An entailment model or LLM judge could replace it behind the same function.
- Replies are short and formulaic by design. The LLM generator gives more natural wording under the same checks.

## Repository layout

```
main.py, validate.py            entry points
tickets.json, kb_articles.json  inputs
results.json, debug_report.*    generated outputs (committed for review)
support_ai/                     pipeline package (one module per concern)
tests/                          unittest suite (pytest-compatible)
```
