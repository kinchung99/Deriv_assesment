# How customers phrase support tickets: findings for the intent classifier

Scope: how real customers word deposit, password, withdrawal, lockout and trading-advice tickets, and which kinds of feedback reuse the same words but should be labelled `other`. All suggestions target the current design: weighted regex rules per intent plus BM25 with a synonym map, stdlib only.

Companion files:
- `intent_phrasings.json` holds the full keyword and phrase lists, 84 regex suggestions with weights and rationales, the feedback taxonomy and the synonym additions. Every regex compiles with `re.IGNORECASE`.
- `dev_examples.json` holds 175 labelled development tickets: 108 issue tickets and 67 `other`.

## 1. Measured effect (development numbers, optimistic)

I prototyped the recommended rules in a scratch copy of the `RuleClassifier` scoring loop. No repo code was changed.

| Check | Current rules | Proposed rules |
|---|---|---|
| `dev_examples.json` (175) | 119 (68.0%) | 174 (99.4%)* |
| `other` recall on dev (67) | 32 (48%) | 67 (100%)* |
| Second ad-hoc check set (48 fresh phrasings) | 31 / 48 | 48 / 48* |
| Sample tickets T1–T5 and KB topic tags A1–A5 | all correct | all correct (unchanged labels) |
| BM25: correct article ranked 1st, dev issue tickets (108) | 70 | 90 (synonyms only) |
| BM25: tickets with no article above threshold | 22 | 5 |

\* Both sets were used while tuning, so these numbers are optimistic. The held-out test set is the real measure. The ad-hoc set raised the score from 39 to 48 only after a second round of generalising fixes (topping up, "going to rise", "should I sell", fee questions, "please improve").

Where the current classifier goes wrong:
- 35 of its 56 dev errors are feedback labelled as an issue. Any single topic word wins, because `other` only wins when every score is 0.
- The rest are implicit phrasings with no rule ("double my money", "go long or short", "can't get past the 2fa screen", "can't take my money out").
- A smaller group is multi-intent ties broken by label order, e.g. a lock caused by a withdrawal goes to `withdrawal_issue`.

## 2. Labelling conventions used in the dev set

These are judgement calls. They are listed so a tester's labels can be compared against them.

- **Issue vs feedback.** A ticket is an issue when it reports *my* transaction, account or credential in a bad state: pending, failed, missing, locked, not received. Praise, complaints about price, policy or speed in general, feature requests, UI critique and retrospective comments about something already resolved are `other`. This follows the feedback taxonomies that separate bug, request, complaint, praise and question, with sentiment as an independent tag ([Liu et al. 2018](https://arxiv.org/abs/1806.01694); [getperspective](https://getperspective.ai/blog/27-customer-feedback-examples-and-how-to-act-on-each-one); [koji taxonomy](https://www.koji.so/docs/customer-feedback-categorization)).
- **Fees.** Fee questions and complaints are `other` even when they name a deposit or withdrawal. Banking77 also separates fee intents (`cash_withdrawal_charge`, `top_up_by_card_charge`) from failure intents (`declined_cash_withdrawal`, `pending_top_up`) ([Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)).
- **Root-cause chains go to the current blocker.**
  - "forgot password → too many tries → locked" is `account_lock`.
  - "account locked after I requested a withdrawal" is `account_lock`.
  - "reset email not arriving" is `password_reset`.
- **2FA, OTP and authenticator** problems that block sign-in are `account_lock`. Reset codes and links are `password_reset`.
- **Resolved plus new issue.** The new issue wins: "got my withdrawal, but my deposit is still not showing" is `deposit_issue`.
- **Product and market information is not advice.** "How do multipliers work?", "what's the price of gold?", "what time does the market open?" and "where are your tutorials?" are `other`. "is BTC a good buy?" and "rise or fall on V75?" are advice.
- **Deposit/withdraw how-to questions** ("How do I withdraw?") are ambiguous and were left out of the dev set. The proposed rules send them to the topical intent, which gives the right KB article. Decide one convention and align it with the tester.

## 3. Findings per intent

### deposit_issue
**Intent behind it:** "I paid and the platform doesn't show it." Customers rarely say "deposit issue". They describe the evidence: the bank was debited, the balance is unchanged, the status is pending, or the top-up was reverted.

- **Vocabulary.**
  - Nouns: deposit, top up / topup / topping up, fund(ed) my account, add money, payment, wallet, balance.
  - Funding channels: card, Visa, bank transfer, M-Pesa/Airtel Money, Skrill/Neteller, USDT/TRC20, payment agent.
  - Crypto words: tx, hash, "confirmed on chain".
  - Sources: [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (`pending_top_up`, `top_up_failed`, `top_up_reverted`, `balance_not_updated_after_bank_transfer`); [Deriv FAQ](https://deriv.com/help-centre/deposits-and-withdrawals): "How long does it take for funds to *reflect*…"; [Trustpilot](https://www.trustpilot.com/review/deriv.com).
- **Non-native phrasing.** "not reflecting", "money not came", "kindly assist", "up to now nothing", "deposite", "depost". M-Pesa users also say "reflect" for delayed credits ([Upwork M-Pesa thread](https://community.upwork.com/t5/Freelancers/MPESA-Delay/td-p/834086)).
- **Implicit phrasings.**
  - "where is my money?? sent it this morning…"
  - "it's been 3 days and still pending"
  - "balance still 0"
  - "the money left my bank but never arrived"
  - "my money disappeared" (Banking77 `top_up_reverted`)
  - "funds deducted but have not arrived" (wording from a broker FAQ title, [Vantage](https://global.vantagehelpcenter.com/hc/en-us/articles/11493241736207-What-should-I-do-if-my-funds-were-deducted-but-have-not-arrived))
- **Direction cue.** Money that never arrived *in my account, wallet or balance* is a deposit. Money that never reached *my bank, card or M-Pesa* is a withdrawal.
- **Traps:**
  - praise ("deposits via mpesa are instant")
  - fee complaints ("why do you charge for card deposits")
  - UI feedback ("deposit page is confusing")
  - feature requests ("please add USDT TRC20")
  - account admin ("remove an old card")

### password_reset
**Intent behind it:** "I can't prove who I am because I don't know the secret, or the reset message isn't working."

- **Vocabulary.**
  - Password words: password, pass, pwd, passcode, access key, login details or credentials.
  - Reset words: reset / recovery link, email, code or SMS.
  - Memory words: forgot, forgotten, can't remember, "no idea what my password is", "expired link", "spam".
  - Bitext's `recover_password` intent has colloquial, typo and offensive variants ("pwd", "my user profile pass", "resety", "the damn password") ([Bitext dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset); flag meanings Q/Z/W in its [README](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/blob/main/README.md)).
- **Implicit phrasings.**
  - "I can't log in" and "I don't remember my login details" (Banking77 puts "I can't log in" under `passcode_forgotten`).
  - "requested a new password 4 times, nothing."
  - "the code goes to a phone I gave away", which is a recovery-channel failure ([unstar.app review analysis](https://unstar.app/blog/robinhood-locked-out-account-restrictions-gold-charges-2026)).
- **Traps:**
  - "my password reset worked, thanks" (praise)
  - "add face ID login, typing my password is annoying" (feature request)
  - "your password rules are too strict" (policy complaint)
  - "wrong password too many times → locked" (`account_lock`)

### withdrawal_issue
**Intent behind it:** "I asked for my money out and it hasn't come." This is often mixed with distrust: "they let me deposit easily but…".

- **Vocabulary and typos.**
  - Forms: withdraw, withdrawal, withdrawl, withdrew, witdraw, cash out / cashout, payout.
  - Status words: pending, processing, in review, on hold, declined, rejected, cancelled, reversed, "insufficient balance".
  - KYC words: verify again, proof of address, documents.
  - Sources: [LCX Trustpilot](https://ie.trustpilot.com/review/lcx.com); [Sikayetvar "stuck in review"](https://www.sikayetvar.com/en/tango-us/tango-keeps-25-usdt-withdrawal-stuck-in-review); Forex Peace Army Deriv threads (via search snippets, page returned 403) ([FPA](https://www.forexpeacearmy.com/community/threads/deriv-com-cant-verify-to-withdraw.72013/)).
- **Implicit phrasings.**
  - "I can't take/get my money out"
  - "money not came to my bank after withdraw"
  - "how long do withdrawals take? I sent a request 48 hours ago"
  - "Getting money in took a minute; getting it out is a fight" ([unstar.app](https://unstar.app/blog/robinhood-locked-out-account-restrictions-gold-charges-2026)).
- **Deriv-specific trap: "payout".** On Deriv, "payout" also means a *contract* payout: Rise/Fall pays out if the exit spot is above or below entry ([Deriv API docs](https://developers.deriv.com/docs/risefall)). "My rise/fall payout looks wrong" is a trade dispute (`other`), not a withdrawal. Deriv's terms also use "payouts" for withdrawals ([Funds and transfers](https://deriv.com/terms-and-conditions/funds-and-transfers)), so treat bare "payout" as weak (1.5) and guard on contract/trade context.
- **Other traps:**
  - "withdrawal fees too high" and "are there any fees for withdrawals?"
  - "withdrawals are super fast, best broker"
  - "please add PayPal withdrawals"
  - "took three agents to sort out my withdrawal last month"

### account_lock
**Intent behind it:** "The system won't let me in, even if I know my password."

- **Vocabulary.**
  - State words: locked / locked out / lockout / unlock, blocked, suspended, disabled, deactivated, frozen / froze, restricted, banned, on hold.
  - Cause words: too many / failed / wrong attempts or tries, cooldown, for security reasons.
  - 2FA words: 2FA, OTP, authenticator, "verification code".
  - Sources: CLINC150 `account_blocked` ("who froze my bank account", "why would I have been locked out") ([CLINC150](https://github.com/clinc/oos-eval)); Banking77 `pin_blocked` ("entered the PIN wrong too many times and now I am blocked").
- **Implicit phrasings.**
  - "lost my phone, now shut out"
  - "can't get past the 2fa screen"
  - "my assets are frozen and nobody says why"
  - "infinite verification loop"
  - Sources: [unstar.app](https://unstar.app/blog/robinhood-locked-out-account-restrictions-gold-charges-2026), which reports that 22 of 202 reviews describe lockouts or verification loops. In broker forums, "account locked after withdrawal request" is common ([FPA](https://www.forexpeacearmy.com/community/threads/deriv-com-cant-verify-to-withdraw.72013/)).
- **Gaps in the current rule set:**
  - `failed (\w+ )?attempts` misses "failed sign-in attempts" because two tokens and a hyphen sit in between.
  - "unlock" and 2FA/OTP have no rule.
  - "who froze my account" has no rule.
- **Traps:**
  - "lock in my profit" (take-profit)
  - "set an app lock PIN" (feature request)
  - "thanks for unlocking my account" (praise)
  - "app keeps logging me out" (bug report)
  - "prices froze during news" (trade complaint)

### trading_advice_request
**Intent behind it:** the user wants someone else to make the trading decision or guarantee its result.

- **Explicit requests.** "which stock/coin/pair", "what should I buy", "signals", "tips", "predictions".
- **Implicit requests.**
  - Good buy or worth it: "is BTC a good buy right now?", "is it worth investing in X" (CLINC150 out-of-scope samples; [r/Bitcoin daily threads](https://reddit.sentinel-team.org/posts/1rqhe4k/snapshots/2026-03-13T17%3A40%3A57.30158Z) where "is it a good time to buy?" is the standard beginner question).
  - Binary choices: "long or short?", and Deriv contract choices "rise or fall on V75?" and "over or under?" ([Deriv multipliers](https://deriv.com/trade/options/multipliers), [Rise/Fall](https://developers.deriv.com/docs/risefall)).
  - Get-rich wording: "double my money", "how much can I earn per day with $50", "sure win", "strategy that never loses on boom and crash". Paid "signals" for Deriv synthetic indices are heavily marketed ([Deriv Boom/Crash](https://deriv.com/derived-indices/crash-boom-indices)).
  - Loss-chasing: "how do I recover my losses quickly?"
- **Traps.** These come from bare `trade|market|buy|sell|profit|recommend|tips`:
  - "I love trading on your app"
  - "made a nice profit, love it"
  - "lost money trading, scam"
  - "the buy and sell buttons are too small"
  - "how do I buy crypto on your platform?"
  - "can I sell my position before expiry?"
  - "what time does the forex market open?"
  - "I recommend this app to my friends"
  - Contract *names* such as "Rise/Fall" and "Higher/Lower". These are product nouns; only "rise **or** fall?" as a question is advice.

## 4. Feedback that must stay `other`

| Category | Intention | Distinguishing signal vs a real issue | Example (keyword trap) |
|---|---|---|---|
| Praise | Satisfaction | positive adjective on the topic (fast, instant, smooth), "thanks for <past action>", no status marker | "Withdrawals are super fast, best broker!" |
| Resolved / thanks | Close the loop | "finally came through", "worked", "all good now", "last week" | "Great support, my password reset worked" |
| Complaint (policy/price) | Dissatisfaction with the company | target is a fee, spread, rules or speed in general; no "my X is pending" | "Your withdrawal fees are too high" |
| Venting | Release frustration after losses | "scam", "rigged", "worst", with no question | "I lost money trading on your platform, scam" |
| Feature request | New capability | "please/can you add\|support\|allow" + new object (not "my"), "would be nice", "I wish", "dark mode", "please improve" | "Please add an option to withdraw to PayPal" |
| UI/UX | Usability critique | page/screen/button/setup + confusing/too small/hard to find | "The deposit page UI is confusing" |
| Bug report | Broken app | crash, lag, "keeps logging me out" | "The app keeps logging me out" (not a lockout) |
| Support experience | Channel complaint | support/agent/bot + slow/forever/repeats, "three agents" | "Took three agents to sort out my withdrawal" |
| Product question | How it works | "how do X work", "can I…", take-profit, demo, regulated | "How do I lock in my profit with a take profit order?" |
| Trade dispute | Execution or settlement | slippage, closed automatically, contract payout | "My rise/fall contract payout looks wrong" |
| Meaningless | Noise | nothing fires | "asdfgh", "hello?" |

The category list follows [Liu et al. 2018](https://arxiv.org/abs/1806.01694) (comment, request, bug, complaint, meaningless) and [getperspective](https://getperspective.ai/blog/27-customer-feedback-examples-and-how-to-act-on-each-one) (feature requests, bugs, praise, pricing objections, support complaints). App-store analyses of trading apps show the same mix: crashes at key moments, slow withdrawals, and requests for trendlines and price alerts ([Exness report](https://kimola.com/reports/unlock-insights-exness-trade-user-feedback-analysis-report-app-store-th-151462) and HotForex report, both via search snippets; the pages now return 410).

## 5. Tone markers (context only, never a label)

- **Urgency.** "asap", "urgent", "pls", "!!!", "2nd time this week", elapsed time ("3 days", "48 hours"). Mostly co-occurs with deposit and withdrawal issues. It is a reason to escalate, not a label.
- **Anger and profanity.** Bitext's "W" register ("the damn password") shows profanity attaches to ordinary intents. Weight venting words low (1.5) so that "Deposit not received!!! Fix it" stays `deposit_issue`.
- **Sarcasm.** "Great, my withdrawal is pending AGAIN. Amazing service." contains praise words around a real issue. Bots that score sentiment get this wrong ([YourStory on airline bots](https://yourstory.com/2018/04/company-bot-misses-sarcasm-customer-tweets); [Mukherjee & Bala 2017](https://research.aston.ac.uk/en/publications/detecting-sarcasm-in-customer-tweets-an-nlp-based-approach)). The rule design handles it: issue evidence (topic + status marker in the same clause) outweighs praise words (2.0), so sarcastic issues keep their label. Sarcasm with no live issue ("another maintenance right when the market moves, thanks a lot") correctly falls to `other`.
- **Politeness fillers.** "kindly", "pls", "thanks in advance" are common in non-native English and are **not** feedback. Do not use bare "thanks" as a praise signal; use "thanks for <past action>".

## 6. Multi-intent tickets

Benchmarks such as MixATIS join single intents with "and", and later work criticises that as unnatural ([survey](https://arxiv.org/html/2512.11258v1)). Real tickets chain intents in four ways:

1. **Causal chain.** "forgot password → tried → locked". Label the current blocker (`account_lock`).
2. **Resolved + new.** "Got my withdrawal, thanks! but my deposit is still not showing". The new issue wins. Use a resolved-outcome guard (−1.5) on the topic that was resolved.
3. **Issue + advice.** "My withdrawal finally came through, now what should I trade to win it back?" Policy already raises `advice_request` when the advice score is at least 3.0. The proposed advice patterns (3.0 each) make this reliable even when another intent wins.
4. **Issue + feedback.** "Love the app but my withdrawal has been stuck for 5 days" and "Your fees are high and my withdrawal was rejected". The issue wins because status markers add +2.0.

There is one known unresolved tie: "Locked out after failed logins and my withdrawal is still pending inside" scores withdrawal 6.5 vs lock 6.0. This is a genuinely mixed ticket, and the existing `conflicting_signals` escalation is the right outcome.

The clause-bounded gap matters here. Every "topic + status" pattern uses `(?:(?!\bbut\b|[.,;!?])[^\n]){0,N}?` so that "deposit went through fine, **but** my withdrawal is pending" does not credit the deposit with "pending".

## 7. Recommended classifier changes

All patterns are in `intent_phrasings.json` → `intents.<label>.regex_suggestions` and compile under `re.IGNORECASE`. The summary below is in rough order of impact.

1. **Add an `other` entry to `INTENT_RULES`.** It covers praise, resolved, feature request, fee, complaint, UI/bug, support, product and trade-dispute patterns (22 patterns, weights 1.5–4.0). No scoring code change is needed, because `ALLOWED_INTENTS` already includes `other` and ties still favour real intents. Representative patterns:
   ```python
   (r"\b(?:please|pls|plz|kindly|could you|can you|you should)\s+(?:add|support|allow|introduce|include|bring)\b(?!\s+(?:my|me|us|it)\b)", 4.0),
   (r"\b(?:would|it'?d) be (?:nice|great|good|cool|helpful)\b|\bi wish\b|\bsuggestion\b|\bdark mode\b", 4.0),
   (r"\b(?:fees?|charges?|commissions?|spreads?|pricing|costs?)\b", 2.0),
   (r"\bthank(?:s| you)(?: so much)? for\b|\bappreciate (?:it|the|your)\b", 2.0),
   ```
2. **Add a minimum winning score.** Use `config.CLASSIFIER_MIN_SCORE = 1.5`: in `certainty_from_scores`, return `other` when the top real-intent score is below it. A single weak keyword ("card", "login", "market", "verify") then no longer decides the label.
3. **Add issue-evidence co-occurrence patterns** for deposit and withdrawal: topic + status marker in the same clause (+2.0), and marker before topic (+1.5). Lower the bare deposit anchor from 3.0 to 2.0. This is what separates "my deposit is still pending" from "deposits are instant, love it".
   ```python
   GAP = r"(?:(?!\bbut\b|[.,;!?])[^\n]){0,40}?"
   PROB = r"(?:not|n'?t|never|no|nothing|pending|stuck|fail\w*|declin\w*|reject\w*|revers\w*|revert\w*|cancel+\w*|missing|delay\w*|still|yet|waiting|processing|disappear\w*|deducted|in review|on hold|issue|problem|error|wrong)"
   (r"\b(?:d[ei]p+o?s[io]?t\w*|top[\s-]?ups?|topp(?:ed|ing)[\s-]?up|topup\w*)\b" + GAP + r"\b" + PROB + r"\b", 2.0)
   ```
4. **Support negative-weight guards and clamp totals at 0.** In `RuleClassifier.score` use `scores[intent] = max(0.0, total)`. Without the clamp, `margin_share` can exceed 1. Guards to add:
   - resolved outcome: −1.5 on deposit and withdrawal, −2.0 on password
   - `lock(ing)? in` / `app lock`: −3.0 on account_lock
   - contract/trade payout: −2.5 on withdrawal
   - face-ID/biometric: −1.5 on password
5. **Broaden the implicit advice patterns** (3.0 each):
   - `good (time to buy|buy|entry)`, `worth investing`
   - `should I buy|sell|hold`
   - `long or short|rise or fall|over or under`
   - `double|grow my money`
   - `sure win|never loses|winning tips`
   - `how much can I earn`
   - `recover my losses|win it back`
   - `going to|will … rise|crash|pump`

   At the same time, lower the bare `trade|trading|market` weight from 1.0 to 0.5 and `buy|sell|invest` from 1.5 to 1.0.
6. **Widen the account_lock rules.**
   - Add `unlock` and 2FA/OTP/authenticator (2.0).
   - Add account-scoped restriction states (`account … disabled|suspended|frozen|restricted`, 2.5) instead of bare "frozen", which also matches "prices froze".
   - Widen the attempts regex to `(?:\s+[\w-]+){0,3}`, so it covers "failed sign-in attempts", "wrong password too many times" and "few tries".
   - Add `log[\s-]?ins?`.
7. **Cover withdrawal typos and implicit wording.**
   - Typos: `\b(?:wi?th?d?r[aeo]w\w*|width?draw\w*|cash[\s-]?outs?|cashout\w*)`.
   - Implicit: `(get|take|pull) (my )?(money|funds|profits) out` (3.0).
   - Lower bare `payout` to 1.5.
8. **Cover password slang and typos:** `\bpas+\s?w\w*|\bpwd\b|\bpasscode\b|\bmy pass\b|\blogin (?:details|credentials)\b` (2.5). Also allow one word in `reset (\w+ )?(email|mail|link|code|sms|otp)` and add `can't remember`.
9. **Add BM25 synonym-map entries** in `text._QUERY_SYNONYMS`. Every target is in the KB vocabulary, and none of them changes T1–T5 retrieval.
   - **Most important:** `withdraw/withdrew/withdrawl → withdrawal`. `stem("withdraw")` is `withdraw`, which does not match the KB token `withdrawal`, so "I can't withdraw" currently misses A3. Also `guaranteed → guarantee`, because the stems differ (`guarant` vs `guarante`).
   - Credentials: `pwd/passcode/pasword → password`.
   - Lock states: `frozen/suspended/disabled/unlock/2fa/otp → lock`.
   - Verification: `kyc/verify/id/selfie → verification`.
   - Declines: `rejected/refused → declined`.
   - Status: `stuck → pending, delayed`.
   - Funding channels: `mpesa/skrill/usdt → payment, method`.
   - Advice words: `signal/tip/outlook/pump → prediction, recommendation`.
   - Assets: `btc/eth/gold/forex/pair/v75 → asset`.

   Measured on dev issue tickets: top-1 article 70 → 90 of 108, and empty results 22 → 5.
10. **Exclude `other` from the mixed-intent check.** In `policy.runner_up`, skip `other`; otherwise a praise word in a real issue ticket would raise `conflicting_signals`. I found no such case on the dev set, but it is cheap insurance. `ADVICE_SCORE_THRESHOLD = 3.0` still works: no non-advice dev ticket reaches 3.0 under the proposed rules.

**Side effects to expect.**
- `INDEX_KB` tags articles with the same rules. I verified that A1–A5 keep their topics under the proposed rules.
- Scores and certainties change, so re-run `python3 main.py` and update the committed outputs and README tables.
- Adding `other` rules lowers `margin_share` for tickets that mix feedback with an issue. That is intended: they are less certain.

## 8. Sources

- **Datasets.**
  - Banking77 train CSV, sampled locally: https://github.com/PolyAI-LDN/task-specific-datasets
  - CLINC150 `data_full.json`, sampled locally: https://github.com/clinc/oos-eval
  - Bitext customer-support CSV and README, sampled locally: https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset
- **Platform help and docs.**
  - Deriv help centre: https://deriv.com/help-centre/deposits-and-withdrawals and https://deriv.com/help-centre/account
  - Deriv terms: https://deriv.com/terms-and-conditions/funds-and-transfers
  - Rise/Fall: https://developers.deriv.com/docs/risefall
  - Multipliers: https://deriv.com/trade/options/multipliers
  - Boom/Crash: https://deriv.com/derived-indices/crash-boom-indices
- **Reviews and complaints.**
  - Trustpilot Deriv (summary via fetch): https://www.trustpilot.com/review/deriv.com
  - LCX Trustpilot: https://ie.trustpilot.com/review/lcx.com
  - Sikayetvar: https://www.sikayetvar.com/en/tango-us/tango-keeps-25-usdt-withdrawal-stuck-in-review
  - unstar.app Robinhood review analysis: https://unstar.app/blog/robinhood-locked-out-account-restrictions-gold-charges-2026
  - Forex Peace Army (search snippet only; 403 on fetch): https://www.forexpeacearmy.com/community/threads/deriv-com-cant-verify-to-withdraw.72013/
  - Upwork M-Pesa thread: https://community.upwork.com/t5/Freelancers/MPESA-Delay/td-p/834086
  - r/Bitcoin daily thread snapshot: https://reddit.sentinel-team.org/posts/1rqhe4k/snapshots/2026-03-13T17%3A40%3A57.30158Z
- **Feedback taxonomy.**
  - Liu et al. 2018: https://arxiv.org/abs/1806.01694
  - IJCNLP-2017 Task 4: https://aclanthology.org/I17-4004/
  - getperspective: https://getperspective.ai/blog/27-customer-feedback-examples-and-how-to-act-on-each-one
  - koji: https://www.koji.so/docs/customer-feedback-categorization
- **Sarcasm and multi-intent.**
  - YourStory: https://yourstory.com/2018/04/company-bot-misses-sarcasm-customer-tweets
  - Mukherjee & Bala: https://research.aston.ac.uk/en/publications/detecting-sarcasm-in-customer-tweets-an-nlp-based-approach
  - Multi-intent SLU survey: https://arxiv.org/html/2512.11258v1
