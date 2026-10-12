#!/usr/bin/env python3
"""Adapt public intent datasets to this project's 6-label taxonomy.

Raw downloads live in eval/raw/ (untrusted data). Run this script with an
isolated interpreter and pass the raw files as arguments:

    python3 -I eval/build_public_datasets.py \
        --banking77 eval/raw/banking77_test.csv \
        --clinc eval/raw/clinc150_data_full.json \
        --bitext eval/raw/bitext_customer_support.csv \
        --out-dir eval/datasets

Sources (downloaded 2026-10-12):
  * Banking77 test split  (PolyAI, CC-BY-4.0)
    https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv
  * CLINC150 data_full.json (clinc/oos-eval, CC-BY-3.0)
    https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json
  * Bitext customer-support LLM chatbot training set v11 (CDLA-Sharing-1.0)
    https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset

Only categories whose mapping is defensible get a label. Categories that are
plausibly in-scope but not clearly one of our labels go to
public_ambiguous.json with expected_intent = null (reported as a prediction
distribution only, never scored). See eval/FINDINGS.md for the rationale.

Sampling is deterministic (fixed seeds); re-running gives identical files.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Banking77
# --------------------------------------------------------------------------- #
B77_MAPPED = {
    # A customer funding their account and the money not arriving / failing.
    "pending_top_up": "deposit_issue",
    "top_up_failed": "deposit_issue",
    "top_up_reverted": "deposit_issue",
    "balance_not_updated_after_bank_transfer": "deposit_issue",
    "balance_not_updated_after_cheque_or_cash_deposit": "deposit_issue",
    # App passcode / password forgotten or not accepted.
    "passcode_forgotten": "password_reset",
    # Withdrawal pending or declined (ATM cash, not a platform payout - caveat).
    "pending_cash_withdrawal": "withdrawal_issue",
    "declined_cash_withdrawal": "withdrawal_issue",
}
# In-scope-ish but not clearly one label: never scored.
B77_AMBIGUOUS = {
    "pin_blocked",  # card PIN blocked, not a login lock
    "cash_withdrawal_not_recognised",  # fraud/unauthorised transaction
    "wrong_amount_of_cash_received",  # ATM dispensing fault
    "cash_withdrawal_charge",  # fee question about withdrawals
    "wrong_exchange_rate_for_cash_withdrawal",
    "topping_up_by_card",  # mostly how-to, some problems
    "transfer_into_account",  # how-to deposit
    "top_up_by_cash_or_cheque",
    "top_up_limits",
    "automatic_top_up",
    "top_up_by_card_charge",
    "top_up_by_bank_transfer_charge",
    "verify_top_up",
    "declined_transfer",  # outgoing P2P transfer; could be read as withdrawal
    "failed_transfer",
    "pending_transfer",
    "transfer_not_received_by_recipient",
    "change_pin",
    "verify_source_of_funds",
    "unable_to_verify_identity",
}
B77_OTHER_PER_CATEGORY = 4
B77_AMBIGUOUS_PER_CATEGORY = 4

# --------------------------------------------------------------------------- #
# CLINC150
# --------------------------------------------------------------------------- #
CLINC_MAPPED = {"account_blocked": "account_lock"}
CLINC_AMBIGUOUS = {
    "freeze_account",  # customer *asks* for a lock
    "direct_deposit",  # set-up how-to
    "pin_change",  # card PIN forgotten/changed
    "transfer",
    "card_declined",  # card declined at a shop
}
CLINC_OTHER_PER_INTENT = 1
CLINC_OOS_OTHER = 100
# Out-of-scope queries that touch finance/markets/deposits are not safely
# "other": they go to the ambiguous file unless explicitly relabelled.
CLINC_OOS_FINANCE_RE = re.compile(
    r"stock|market|invest|bitcoin|\bdow\b|deposit|\bcalls\b|gone up or down|bank login|\bsell stocks",
    re.IGNORECASE,
)
CLINC_OOS_RELABEL = {
    "make a list of stocks that would be good to invest in to start making money": "trading_advice_request",
}

# --------------------------------------------------------------------------- #
# Bitext customer support
# --------------------------------------------------------------------------- #
BITEXT_MAPPED = {"recover_password": "password_reset"}
BITEXT_AMBIGUOUS = {"payment_issue"}  # e-commerce payment problem; maybe deposit-like
BITEXT_MAPPED_N = 80
BITEXT_OTHER_PER_INTENT = 6
BITEXT_AMBIGUOUS_N = 10
_PLACEHOLDERS = {
    "Order Number": "#48213",
    "Account Type": "premium",
    "Account Category": "premium",
    "Invoice Number": "INV-1093",
    "Person Name": "Alex",
    "Refund Amount": "$40",
    "Currency Symbol": "$",
    "Delivery City": "London",
    "Delivery Country": "UK",
}
_PLACEHOLDER_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


# The single KB article covering each mapped label (A1 is about card deposits
# but is still the only deposit article; A3 is the only withdrawal article).
EXPECTED_ARTICLE = {
    "deposit_issue": "A1",
    "password_reset": "A2",
    "withdrawal_issue": "A3",
    "account_lock": "A4",
    "trading_advice_request": "A5",
}


def _fill(text: str) -> str:
    return _PLACEHOLDER_RE.sub(lambda m: _PLACEHOLDERS.get(m.group(1), "item"), text)


def _case(cid: str, message: str, intent: Optional[str], category: str, source: str) -> dict:
    case = {
        "id": cid,
        "message": " ".join(message.split()),
        "language": "en",
        "expected_intent": intent,
        "category": category,
        "source": source,
    }
    if intent == "trading_advice_request":
        case["expect_flags"] = ["advice_request"]
    if intent in EXPECTED_ARTICLE:
        case["expect_article"] = EXPECTED_ARTICLE[intent]
    return case


def _sample(items: List, n: int, rng: random.Random) -> List:
    items = list(items)
    if len(items) <= n:
        return items
    return rng.sample(items, n)


def build_banking77(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_cat: Dict[str, List[str]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["text"])
    rng = random.Random(77)
    labelled, ambiguous = [], []
    for cat in sorted(by_cat):
        texts = by_cat[cat]
        if cat in B77_MAPPED:
            chosen, intent, bucket = texts, B77_MAPPED[cat], labelled
        elif cat in B77_AMBIGUOUS:
            chosen, intent, bucket = _sample(texts, B77_AMBIGUOUS_PER_CATEGORY, rng), None, ambiguous
        else:
            chosen, intent, bucket = _sample(texts, B77_OTHER_PER_CATEGORY, rng), "other", labelled
        for t in chosen:
            bucket.append((cat, t, intent))
    src = "banking77/test"
    lab = [_case(f"B77-{i:04d}", t, it, c, src) for i, (c, t, it) in enumerate(labelled, 1)]
    amb = [_case(f"B77A-{i:03d}", t, None, c, src) for i, (c, t, it) in enumerate(ambiguous, 1)]
    return lab, amb


def build_clinc(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(150)
    by_intent: Dict[str, List[str]] = {}
    for text, label in data["test"]:
        by_intent.setdefault(label, []).append(text)
    labelled, ambiguous = [], []
    for intent in sorted(by_intent):
        texts = by_intent[intent]
        if intent in CLINC_MAPPED:
            labelled += [(intent, t, CLINC_MAPPED[intent]) for t in texts]
        elif intent in CLINC_AMBIGUOUS:
            ambiguous += [(intent, t, None) for t in _sample(texts, 4, rng)]
        else:
            labelled += [(intent, t, "other") for t in _sample(texts, CLINC_OTHER_PER_INTENT, rng)]
    oos_plain = []
    for text, _ in data["oos_test"]:
        if text in CLINC_OOS_RELABEL:
            labelled.append(("oos(relabelled)", text, CLINC_OOS_RELABEL[text]))
        elif CLINC_OOS_FINANCE_RE.search(text):
            ambiguous.append(("oos_finance", text, None))
        else:
            oos_plain.append(text)
    labelled += [("oos", t, "other") for t in _sample(oos_plain, CLINC_OOS_OTHER, rng)]
    src = "clinc150/test"
    lab = [_case(f"CL-{i:04d}", t, it, c, src) for i, (c, t, it) in enumerate(labelled, 1)]
    amb = [_case(f"CLA-{i:03d}", t, None, c, src) for i, (c, t, it) in enumerate(ambiguous, 1)]
    return lab, amb


def build_bitext(path: Path):
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_intent: Dict[str, List[str]] = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r["instruction"])
    rng = random.Random(27)
    labelled, ambiguous = [], []
    for intent in sorted(by_intent):
        texts = sorted(set(by_intent[intent]))
        if intent in BITEXT_MAPPED:
            labelled += [(intent, t, BITEXT_MAPPED[intent]) for t in _sample(texts, BITEXT_MAPPED_N, rng)]
        elif intent in BITEXT_AMBIGUOUS:
            ambiguous += [(intent, t, None) for t in _sample(texts, BITEXT_AMBIGUOUS_N, rng)]
        else:
            labelled += [(intent, t, "other") for t in _sample(texts, BITEXT_OTHER_PER_INTENT, rng)]
    src = "bitext-customer-support-v11"
    lab = [_case(f"BX-{i:04d}", _fill(t), it, c, src) for i, (c, t, it) in enumerate(labelled, 1)]
    amb = [_case(f"BXA-{i:03d}", _fill(t), None, c, src) for i, (c, t, it) in enumerate(ambiguous, 1)]
    return lab, amb


def _write(cases: List[dict], path: Path) -> None:
    path.write_text(json.dumps(cases, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(cases):4d} cases -> {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--banking77", type=Path, required=True)
    ap.add_argument("--clinc", type=Path, required=True)
    ap.add_argument("--bitext", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    b_lab, b_amb = build_banking77(args.banking77)
    c_lab, c_amb = build_clinc(args.clinc)
    x_lab, x_amb = build_bitext(args.bitext)
    _write(b_lab, args.out_dir / "banking77.json")
    _write(c_lab, args.out_dir / "clinc150.json")
    _write(x_lab, args.out_dir / "bitext.json")
    _write(b_amb + c_amb + x_amb, args.out_dir / "public_ambiguous.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
