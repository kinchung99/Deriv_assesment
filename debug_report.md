# Debug report

Components: classifier=`rules`, retriever=`bm25`, generator=`template`

| Ticket | Intent | Retrieved articles | Confidence | Escalate | Flags |
|---|---|---|---|---|---|
| T1 | deposit_issue | A1 Card deposit processing times | 0.99 | no | — |
| T2 | password_reset | A2 Password reset troubleshooting | 1.0 | no | — |
| T3 | withdrawal_issue | A3 Withdrawal review and declines | 0.65 | **yes** | policy_sensitive, account_specific_request |
| T4 | trading_advice_request | A5 No investment or trading advice | 0.84 | no | advice_request |
| T5 | account_lock | A4 Account lock after failed sign-in attempts | 1.0 | no | — |

## T1

> I deposited by bank card two hours ago but my balance still shows zero. What should I check?

- **Predicted intent:** `deposit_issue` (source: rules, certainty 1.0, scores {'deposit_issue': 6.0})
- **Retrieved articles:**
  - A1 — Card deposit processing times (BM25 4.5924)
- **Confidence:** 0.99 — {'classifier_certainty': 1.0, 'retrieval_strength': 1.0, 'grounding_score': 0.9545}
- **Decision:** Not escalated: intent 'deposit_issue' (certainty 1.00), relevant article A1 'Card deposit processing times' (BM25 4.59), reply grounded (score 0.95), confidence 0.99 >= 0.5, no policy-sensitive or account-specific signals.

## T2

> I forgot my password and I am not receiving the reset email.

- **Predicted intent:** `password_reset` (source: rules, certainty 1.0, scores {'password_reset': 8.0})
- **Retrieved articles:**
  - A2 — Password reset troubleshooting (BM25 10.784)
- **Confidence:** 1.0 — {'classifier_certainty': 1.0, 'retrieval_strength': 1.0, 'grounding_score': 1.0}
- **Decision:** Not escalated: intent 'password_reset' (certainty 1.00), relevant article A2 'Password reset troubleshooting' (BM25 10.78), reply grounded (score 1.00), confidence 1.00 >= 0.5, no policy-sensitive or account-specific signals.

## T3

> Why was my withdrawal declined after verification?

- **Predicted intent:** `withdrawal_issue` (source: rules, certainty 1.0, scores {'withdrawal_issue': 4.5})
- **Retrieved articles:**
  - A3 — Withdrawal review and declines (BM25 4.7237)
- **Confidence:** 0.65 — {'classifier_certainty': 1.0, 'retrieval_strength': 1.0, 'grounding_score': 1.0, 'policy_sensitive_penalty': -0.15, 'account_specific_penalty': -0.2}
- **Request signals:** asks about own account: 'Why was my'
- **KB guidance honoured:** [A3] Support should not promise approval times without checking account status.
- **Decision:** Escalated: policy-sensitive topic 'withdrawal_issue' (KB A3 requires checking account status/compliance); agent must review before any commitment; answer depends on the customer's own account data, which the draft cannot see.

## T4

> Can you tell me which asset will go up today so I can make profit?

- **Predicted intent:** `trading_advice_request` (source: rules, certainty 1.0, scores {'trading_advice_request': 9.5})
- **Retrieved articles:**
  - A5 — No investment or trading advice (BM25 3.0165)
- **Confidence:** 0.84 — {'classifier_certainty': 1.0, 'retrieval_strength': 0.7541, 'grounding_score': 0.6515}
- **Request signals:** trading-advice rules scored 9.5
- **KB guidance honoured:** [A5] Support agents must not provide market predictions, profit guarantees, or asset recommendations.; [A5] Such requests should be politely declined and redirected to general educational resources if available.
- **Decision:** Not escalated: trading-advice request was politely refused using the policy article (A5); no human action needed.

## T5

> My account was locked after too many login attempts. What do I do now?

- **Predicted intent:** `account_lock` (source: rules, certainty 1.0, scores {'account_lock': 6.0})
- **Retrieved articles:**
  - A4 — Account lock after failed sign-in attempts (BM25 6.3377)
- **Confidence:** 1.0 — {'classifier_certainty': 1.0, 'retrieval_strength': 1.0, 'grounding_score': 1.0}
- **KB guidance honoured:** [A4] Avoid sharing account-specific details in unsecured channels.
- **Decision:** Not escalated: intent 'account_lock' (certainty 1.00), relevant article A4 'Account lock after failed sign-in attempts' (BM25 6.34), reply grounded (score 1.00), confidence 1.00 >= 0.5, no policy-sensitive or account-specific signals.
