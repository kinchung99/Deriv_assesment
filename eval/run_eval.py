#!/usr/bin/env python3
"""Evaluation harness for the support-ticket pipeline (stdlib only).

    python3 eval/run_eval.py                       # every eval/datasets/*.json
    python3 eval/run_eval.py eval/datasets/handwritten.json
    python3 eval/run_eval.py --report eval/report_after.json --compare eval/report_baseline.json

Per dataset it reports: accuracy, macro-F1, per-intent P/R/F1, confusion
matrix, per-category accuracy, top-1 article accuracy, escalation agreement,
expected-flag recall, schema/semantic validity of every record, safety
(advice requests refused, no promises), determinism (same process twice and
across processes with different PYTHONHASHSEED) and the misclassified cases.

Dataset format: a JSON list of
  {"id", "message", "language", "expected_intent",
   optional "acceptable_intents", "expect_escalation", "expect_flags",
   "expect_article", "category", "source"}
expected_intent = null marks an unscored case (prediction distribution only).
If acceptable_intents is given and the prediction is in it, the case counts as
correct (the prediction is used as the gold label for the metrics).

The pipeline is used only through its public API (build_pipeline().run).
Exit status is 0 unless --strict is given and a hard check fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_ai import config, templates  # noqa: E402
from support_ai.grounding import extract_citations  # noqa: E402
from support_ai.grounding import find_unsupported_claims  # noqa: E402
from support_ai.pipeline import build_pipeline  # noqa: E402
from support_ai.validator import is_safe_refusal, schema_errors, semantic_errors  # noqa: E402

LABELS = list(config.ALLOWED_INTENTS)
ADVICE = config.ADVICE_INTENT
DEFAULT_REPORT = EVAL_DIR / "report_baseline.json"

# Independent promise detector and spec-compliance checks live in check_requirements.py.
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import check_requirements as reqs  # noqa: E402
from check_requirements import independent_promises  # noqa: E402


# --------------------------------------------------------------------------- #
# Running the pipeline
# --------------------------------------------------------------------------- #
def load_dataset(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path}: dataset must be a JSON list")
    ids = [c["id"] for c in data]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise SystemExit(f"{path}: duplicate case ids {dup[:5]}")
    for c in data:
        exp = c.get("expected_intent")
        if exp is not None and exp not in LABELS:
            raise SystemExit(f"{path}: case {c['id']} has unknown expected_intent {exp!r}")
    return data


def write_tickets(cases: Sequence[dict], directory: Path) -> Path:
    tickets = [{"id": c["id"], "message": c["message"], "language": c.get("language", "en")} for c in cases]
    path = directory / "tickets.json"
    path.write_text(json.dumps(tickets, ensure_ascii=False), encoding="utf-8")
    return path


def run_pipeline(tickets_path: Path, kb_path: Path, classifier: str, generator: str):
    pipeline = build_pipeline(classifier=classifier, generator=generator)
    start = time.perf_counter()
    run = pipeline.run(tickets_path, kb_path, out_dir=None)
    return run, time.perf_counter() - start


_SUBPROCESS_SNIPPET = (
    "import json,sys; sys.path.insert(0, sys.argv[1]);"
    "from support_ai.pipeline import build_pipeline;"
    "r = build_pipeline(classifier=sys.argv[4], generator=sys.argv[5]).run(sys.argv[2], sys.argv[3], out_dir=None);"
    "sys.stdout.write(json.dumps(r.records, ensure_ascii=False))"
)


def cross_process_records(tickets_path: Path, kb_path: Path, seed: str, classifier: str, generator: str) -> Optional[str]:
    env = dict(os.environ, PYTHONHASHSEED=seed)
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SNIPPET, str(ROOT), str(tickets_path), str(kb_path), classifier, generator],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def prf(gold: List[str], pred: List[str]) -> dict:
    labels_present = sorted(set(gold) | set(pred), key=LABELS.index)
    per = {}
    for lab in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per[lab] = {
            "support": tp + fn,
            "predicted": tp + fp,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    macro = sum(per[lab]["f1"] for lab in labels_present) / len(labels_present) if labels_present else 0.0
    gold_labels = [lab for lab in LABELS if per[lab]["support"]]
    macro_gold = sum(per[lab]["f1"] for lab in gold_labels) / len(gold_labels) if gold_labels else 0.0
    return {"per_intent": per, "macro_f1": round(macro, 4), "macro_f1_labels": labels_present,
            "macro_f1_gold_labels": round(macro_gold, 4)}


def confusion(gold: List[str], pred: List[str]) -> Dict[str, Dict[str, int]]:
    m = {g: {p: 0 for p in LABELS} for g in LABELS}
    for g, p in zip(gold, pred):
        m[g][p] += 1
    return m


def evaluate_dataset(path: Path, kb_path: Path, args) -> dict:
    cases = load_dataset(path)
    with tempfile.TemporaryDirectory() as tmp:
        tickets_path = write_tickets(cases, Path(tmp))
        run1, secs1 = run_pipeline(tickets_path, kb_path, args.classifier, args.generator)
        run2, secs2 = run_pipeline(tickets_path, kb_path, args.classifier, args.generator)
        dump1 = json.dumps(run1.records, ensure_ascii=False)
        dump2 = json.dumps(run2.records, ensure_ascii=False)
        determinism = {"same_process_identical": dump1 == dump2}
        if not args.no_cross_process:
            outs = {s: cross_process_records(tickets_path, kb_path, s, args.classifier, args.generator) for s in ("1", "4242")}
            determinism["cross_process_identical"] = all(o == dump1 for o in outs.values())
            determinism["cross_process_failed"] = [s for s, o in outs.items() if o is None]
        if dump1 != dump2:
            r2 = {r["ticket_id"]: r for r in run2.records}
            determinism["differing_ids"] = [r["ticket_id"] for r in run1.records if r != r2.get(r["ticket_id"])][:20]

    run = run1
    kb = run.kb
    records = {r["ticket_id"]: r for r in run.records}
    traces = {t.ticket.id: t for t in run.traces}

    # ---- schema / semantic validity on every record
    schema_bad, semantic_bad, repaired, fallback = [], [], [], []
    for cid, rec in records.items():
        se = schema_errors(rec, kb.keys())
        if se:
            schema_bad.append({"id": cid, "errors": se})
            continue
        me = semantic_errors(rec, kb)
        if me:
            semantic_bad.append({"id": cid, "errors": me})
        tr = traces[cid]
        if tr.validation_errors:
            repaired.append({"id": cid, "errors": tr.validation_errors, "fixes": tr.validation_fixes})
        if config.FLAG_VALIDATION_FAILED in rec["safety_flags"]:
            fallback.append(cid)

    # ---- intent metrics
    scored = [c for c in cases if c.get("expected_intent") is not None]
    gold, pred, wrong = [], [], []
    by_category = defaultdict(lambda: [0, 0])
    conf_correct, conf_wrong = [], []
    for c in scored:
        rec = records[c["id"]]
        p = rec["intent"]
        acceptable = c.get("acceptable_intents") or []
        g = p if p in acceptable else c["expected_intent"]
        gold.append(g)
        pred.append(p)
        ok = g == p
        cat = c.get("category", "-")
        by_category[cat][0] += ok
        by_category[cat][1] += 1
        (conf_correct if ok else conf_wrong).append(rec["confidence"])
        if not ok:
            tr = traces[c["id"]]
            wrong.append(
                {
                    "id": c["id"],
                    "category": cat,
                    "expected": c["expected_intent"],
                    "predicted": p,
                    "confidence": rec["confidence"],
                    "escalated": rec["needs_human_escalation"],
                    "articles": rec["retrieved_articles"],
                    "rule_scores": {k: v for k, v in tr.classification.scores.items() if v},
                    "message": c["message"],
                }
            )
    n = len(scored)
    accuracy = sum(g == p for g, p in zip(gold, pred)) / n if n else None
    metrics = prf(gold, pred) if n else {"per_intent": {}, "macro_f1": None, "macro_f1_labels": [],
                                          "macro_f1_gold_labels": None}

    other_gold = [(g, p) for g, p in zip(gold, pred) if g == config.OTHER_INTENT]
    other_fp_rate = (sum(p != config.OTHER_INTENT for _, p in other_gold) / len(other_gold)) if other_gold else None

    # confidence calibration: accuracy above / below the low-confidence threshold
    def bucket_acc(lo, hi):
        sel = [(g == p) for c, g, p in zip(scored, gold, pred) if lo <= records[c["id"]]["confidence"] < hi]
        return {"n": len(sel), "accuracy": round(sum(sel) / len(sel), 4) if sel else None}

    calibration = {
        "conf_<0.5": bucket_acc(0.0, 0.5),
        "conf_0.5-0.8": bucket_acc(0.5, 0.8),
        "conf_>=0.8": bucket_acc(0.8, 1.01),
        "mean_conf_correct": round(sum(conf_correct) / len(conf_correct), 4) if conf_correct else None,
        "mean_conf_wrong": round(sum(conf_wrong) / len(conf_wrong), 4) if conf_wrong else None,
        "wrong_but_not_escalated": sum(1 for w in wrong if not w["escalated"]),
    }

    # ---- article top-1
    art_cases = [c for c in cases if c.get("expect_article")]
    art_wrong = []
    for c in art_cases:
        got = records[c["id"]]["retrieved_articles"]
        if not got or got[0] != c["expect_article"]:
            art_wrong.append({"id": c["id"], "expected": c["expect_article"], "got": got, "message": c["message"][:120]})
    article = {
        "n": len(art_cases),
        "top1_accuracy": round(1 - len(art_wrong) / len(art_cases), 4) if art_cases else None,
        "misses": art_wrong,
    }

    # ---- escalation agreement
    esc_cases = [c for c in cases if "expect_escalation" in c]
    esc_dis = []
    for c in esc_cases:
        got = records[c["id"]]["needs_human_escalation"]
        if got != c["expect_escalation"]:
            esc_dis.append({"id": c["id"], "expected": c["expect_escalation"], "got": got,
                            "flags": records[c["id"]]["safety_flags"], "message": c["message"][:120]})
    escalation = {
        "n": len(esc_cases),
        "agreement": round(1 - len(esc_dis) / len(esc_cases), 4) if esc_cases else None,
        "disagreements": esc_dis,
        "overall_escalation_rate": round(sum(r["needs_human_escalation"] for r in records.values()) / len(records), 4),
    }

    # ---- expected flags
    flag_cases = [c for c in cases if c.get("expect_flags")]
    flag_miss = []
    for c in flag_cases:
        missing = [f for f in c["expect_flags"] if f not in records[c["id"]]["safety_flags"]]
        if missing:
            flag_miss.append({"id": c["id"], "missing": missing, "got": records[c["id"]]["safety_flags"],
                              "message": c["message"][:120]})
    flags = {"n": len(flag_cases), "all_present_rate": round(1 - len(flag_miss) / len(flag_cases), 4) if flag_cases else None,
             "misses": flag_miss}

    # ---- safety
    advice_cases = [c for c in cases if c.get("expected_intent") == ADVICE or "advice_request" in (c.get("expect_flags") or [])]
    not_refused, not_detected = [], []
    for c in advice_cases:
        rec = records[c["id"]]
        if not is_safe_refusal(rec["reply_draft"]):
            not_refused.append({"id": c["id"], "intent": rec["intent"], "flags": rec["safety_flags"],
                                "message": c["message"][:120], "reply": rec["reply_draft"][:200]})
        if rec["intent"] != ADVICE and config.FLAG_ADVICE_REQUEST not in rec["safety_flags"]:
            not_detected.append(c["id"])
    flagged_advice_unrefused = [cid for cid, r in records.items()
                                if (r["intent"] == ADVICE or config.FLAG_ADVICE_REQUEST in r["safety_flags"])
                                and not is_safe_refusal(r["reply_draft"])]
    false_refusals = [cid for cid, r in records.items()
                      if config.FLAG_ADVICE_REQUEST in r["safety_flags"]
                      and next((c for c in cases if c["id"] == cid), {}).get("expected_intent") == config.OTHER_INTENT]
    promises_pipeline, promises_independent = [], []
    for cid, rec in records.items():
        src = " ".join(kb[a].text for a in rec["retrieved_articles"] if a in kb)
        claims = find_unsupported_claims(rec["reply_draft"], src)
        if claims:
            promises_pipeline.append({"id": cid, "claims": claims})
        ind = independent_promises(rec["reply_draft"])
        if ind:
            promises_independent.append({"id": cid, "matches": ind, "reply": rec["reply_draft"][:200]})
    safety = {
        "advice_like_cases": len(advice_cases),
        "advice_refused": len(advice_cases) - len(not_refused),
        "advice_refusal_rate": round(1 - len(not_refused) / len(advice_cases), 4) if advice_cases else None,
        "advice_not_refused": not_refused,
        "advice_not_detected_ids": not_detected,
        "flagged_advice_without_refusal": flagged_advice_unrefused,
        "other_cases_flagged_as_advice": false_refusals,
        "replies_with_unsupported_claims": promises_pipeline,
        "replies_with_promise_language_independent": promises_independent,
    }

    # ---- coherence between reply text, escalation and intent
    default_kb = Path(kb_path).resolve() == Path(config.DEFAULT_KB_PATH).resolve()
    article_topic = {v: k for k, v in reqs.OPERATIONAL_ARTICLE.items()} if default_kb else {}
    esc_selfserve = [cid for cid, r in records.items()
                     if r["needs_human_escalation"] and templates.CLOSER_SELF_SERVE in r["reply_draft"]]
    noesc_review = [cid for cid, r in records.items() if not r["needs_human_escalation"]
                    and (templates.CLOSER_AGENT_REVIEW in r["reply_draft"] or templates.CLOSER_NO_CONTEXT in r["reply_draft"])]
    off_topic = []
    for cid, r in records.items():
        if r["intent"] in (ADVICE, config.OTHER_INTENT):
            continue
        bad = [a for a in extract_citations(r["reply_draft"])
               if a in article_topic and article_topic[a] not in (r["intent"], ADVICE)]
        if bad:
            off_topic.append({"id": cid, "intent": r["intent"], "off_topic_citations": bad,
                              "message": next(c["message"] for c in cases if c["id"] == cid)[:110]})
    coherence = {
        "escalated_but_self_serve_closer": esc_selfserve,
        "not_escalated_but_review_closer": noesc_review,
        "cites_article_of_another_intent": off_topic,
    }

    # ---- unscored distribution
    unscored = [c for c in cases if c.get("expected_intent") is None]
    dist = defaultdict(Counter)
    for c in unscored:
        dist[c.get("category", "-")][records[c["id"]]["intent"]] += 1

    return {
        "dataset": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "n_cases": len(cases),
        "n_scored": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "macro_f1": metrics["macro_f1"],
        "macro_f1_labels": metrics["macro_f1_labels"],
        "macro_f1_gold_labels": metrics["macro_f1_gold_labels"],
        "per_intent": metrics["per_intent"],
        "confusion_matrix": confusion(gold, pred) if n else {},
        "per_category_accuracy": {k: {"correct": v[0], "n": v[1], "accuracy": round(v[0] / v[1], 4)}
                                  for k, v in sorted(by_category.items())},
        "other_false_positive_rate": round(other_fp_rate, 4) if other_fp_rate is not None else None,
        "calibration": calibration,
        "article_top1": article,
        "escalation": escalation,
        "expected_flags": flags,
        "schema": {
            "records": len(records),
            "schema_invalid": schema_bad,
            "semantic_invalid": semantic_bad,
            "repaired_by_enforce": repaired,
            "safe_fallback_used": fallback,
        },
        "safety": safety,
        "coherence": coherence,
        "determinism": determinism,
        "timing_seconds": {"run1": round(secs1, 3), "run2": round(secs2, 3)},
        "unscored_prediction_distribution": {k: dict(v) for k, v in sorted(dist.items())},
        "misclassified": wrong,
    }


# --------------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------------- #
ABBR = {"deposit_issue": "dep", "password_reset": "pwd", "withdrawal_issue": "wdr",
        "account_lock": "lck", "trading_advice_request": "adv", "other": "oth"}


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_dataset(r: dict, show: int) -> None:
    print("=" * 78)
    print(f"{r['dataset']}  ({r['n_scored']} scored / {r['n_cases']} cases)")
    print("=" * 78)
    print(f"accuracy {_pct(r['accuracy'])}   macro-F1 {r['macro_f1']} (gold labels only: {r['macro_f1_gold_labels']})   "
          f"'other' false-positive rate {_pct(r['other_false_positive_rate'])}")
    print(f"\n{'intent':<24}{'support':>8}{'pred':>6}{'prec':>8}{'recall':>8}{'f1':>8}")
    for lab in LABELS:
        m = r["per_intent"].get(lab)
        if m and (m["support"] or m["predicted"]):
            print(f"{lab:<24}{m['support']:>8}{m['predicted']:>6}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}")
    if r["confusion_matrix"]:
        print("\nconfusion (rows = gold, cols = predicted)")
        print(" " * 6 + "".join(f"{ABBR[l]:>6}" for l in LABELS))
        for g in LABELS:
            row = r["confusion_matrix"][g]
            if sum(row.values()):
                print(f"{ABBR[g]:<6}" + "".join(f"{row[p]:>6}" for p in LABELS))
    cal = r["calibration"]
    print(f"\ncalibration: acc@conf<0.5 {_pct(cal['conf_<0.5']['accuracy'])} (n={cal['conf_<0.5']['n']}), "
          f"0.5-0.8 {_pct(cal['conf_0.5-0.8']['accuracy'])} (n={cal['conf_0.5-0.8']['n']}), "
          f">=0.8 {_pct(cal['conf_>=0.8']['accuracy'])} (n={cal['conf_>=0.8']['n']}); "
          f"misclassified & not escalated: {cal['wrong_but_not_escalated']}")
    a, e, f = r["article_top1"], r["escalation"], r["expected_flags"]
    print(f"article top-1: {_pct(a['top1_accuracy'])} (n={a['n']})   escalation agreement: "
          f"{_pct(e['agreement'])} (n={e['n']}, overall rate {_pct(e['overall_escalation_rate'])})   "
          f"expected flags present: {_pct(f['all_present_rate'])} (n={f['n']})")
    s, sc, d = r["safety"], r["schema"], r["determinism"]
    print(f"schema invalid: {len(sc['schema_invalid'])}  semantic invalid: {len(sc['semantic_invalid'])}  "
          f"repaired by enforce: {len(sc['repaired_by_enforce'])}  safe-fallback records: {len(sc['safe_fallback_used'])}")
    print(f"safety: advice refused {s['advice_refused']}/{s['advice_like_cases']}  "
          f"flagged-but-unrefused {len(s['flagged_advice_without_refusal'])}  "
          f"'other' flagged as advice {len(s['other_cases_flagged_as_advice'])}  "
          f"unsupported-claim replies {len(s['replies_with_unsupported_claims'])}  "
          f"promise-language replies (independent) {len(s['replies_with_promise_language_independent'])}")
    co = r["coherence"]
    print(f"coherence: escalated but self-serve closer {len(co['escalated_but_self_serve_closer'])}  "
          f"not escalated but review closer {len(co['not_escalated_but_review_closer'])}  "
          f"cites another intent's article {len(co['cites_article_of_another_intent'])}")
    print(f"determinism: {d}   time: {r['timing_seconds']}")
    for item in s["advice_not_refused"]:
        print(f"  ! advice NOT refused {item['id']} [{item['intent']}] {item['message']!r}")
    if r["unscored_prediction_distribution"]:
        print("\nunscored cases, predicted-intent distribution by category:")
        for cat, dist in r["unscored_prediction_distribution"].items():
            print(f"  {cat:<42}" + ", ".join(f"{ABBR[k]}={v}" for k, v in sorted(dist.items())))
    wrong = r["misclassified"]
    if wrong:
        print(f"\nmisclassified ({len(wrong)}; showing {min(show, len(wrong))}):")
        for w in wrong[:show]:
            msg = w["message"] if len(w["message"]) <= 90 else w["message"][:87] + "..."
            print(f"  {w['id']:<9} {ABBR[w['expected']]}->{ABBR[w['predicted']]}  c={w['confidence']:.2f} "
                  f"{'E' if w['escalated'] else ' '} {msg!r}")
    print()


def print_comparison(reports: List[dict], baseline_path: Path) -> None:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    by_ds = {d["dataset"]: d for d in base.get("datasets", [])}
    print("=" * 78)
    print(f"comparison against {baseline_path}")
    for r in reports:
        b = by_ds.get(r["dataset"])
        if not b or r["accuracy"] is None:
            continue
        old = {w["id"] for w in b["misclassified"]}
        new = {w["id"] for w in r["misclassified"]}
        print(f"  {r['dataset']:<38} acc {_pct(b['accuracy'])} -> {_pct(r['accuracy'])}   "
              f"macro-F1 {b['macro_f1']} -> {r['macro_f1']}   fixed {len(old - new)}  broken {len(new - old)}")
        if new - old:
            print(f"      newly wrong: {sorted(new - old)[:15]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", type=Path, help="dataset JSON files (default: eval/datasets/*.json)")
    ap.add_argument("--kb", type=Path, default=config.DEFAULT_KB_PATH)
    ap.add_argument("--classifier", default="rules", choices=["rules", "llm"])
    ap.add_argument("--generator", default="template", choices=["template", "llm"])
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="where to write the JSON report")
    ap.add_argument("--compare", type=Path, help="previous report to diff against")
    ap.add_argument("--show", type=int, default=40, help="misclassified examples to print per dataset")
    ap.add_argument("--no-cross-process", action="store_true", help="skip the PYTHONHASHSEED determinism check")
    ap.add_argument("--no-requirements", action="store_true", help="skip the spec-compliance checks")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 on schema/semantic errors, unrefused advice or non-determinism")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)

    paths = args.datasets or sorted((EVAL_DIR / "datasets").glob("*.json"))
    paths = [p.resolve() for p in paths]
    reports = [evaluate_dataset(p, args.kb, args) for p in paths]
    for r in reports:
        print_dataset(r, args.show)
    requirements = None
    if not args.no_requirements:
        per = {r["dataset"]: reqs.check_dataset(load_dataset(ROOT / r["dataset"]), args.kb) for r in reports}
        glob = reqs.check_static_and_global(args.kb)
        for r in reports:
            r["requirements"] = per[r["dataset"]]
        requirements = {"overall": reqs.overall(per, glob), "global": glob}
        reqs.print_table(per, glob)

    print("=" * 78)
    print(f"{'dataset':<38}{'n':>6}{'acc':>8}{'macroF1':>9}{'mF1gold':>9}{'oth-FP':>8}{'art@1':>8}{'esc':>8}{'adv-ref':>9}")
    for r in reports:
        s = r["safety"]
        adv = f"{s['advice_refused']}/{s['advice_like_cases']}" if s["advice_like_cases"] else "-"
        mf1 = f"{r['macro_f1']:.3f}" if r["macro_f1"] is not None else "n/a"
        mf1g = f"{r['macro_f1_gold_labels']:.3f}" if r["macro_f1_gold_labels"] is not None else "n/a"
        print(f"{r['dataset']:<38}{r['n_scored']:>6}{_pct(r['accuracy']):>8}{mf1:>9}{mf1g:>9}"
              f"{_pct(r['other_false_positive_rate']):>8}{_pct(r['article_top1']['top1_accuracy']):>8}"
              f"{_pct(r['escalation']['agreement']):>8}{adv:>9}")

    out = {
        "harness": "eval/run_eval.py",
        "pipeline": {"classifier": args.classifier, "generator": args.generator, "retriever": "bm25",
                     "kb": str(Path(args.kb).resolve().relative_to(ROOT)) if Path(args.kb).resolve().is_relative_to(ROOT) else str(args.kb)},
        "python": sys.version.split()[0],
        "requirements": requirements,
        "datasets": reports,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")
    if args.compare:
        print_comparison(reports, args.compare)

    hard_fail = any(
        r["schema"]["schema_invalid"] or r["schema"]["semantic_invalid"] or r["safety"]["advice_not_refused"]
        or not r["determinism"]["same_process_identical"] or r["determinism"].get("cross_process_identical") is False
        for r in reports
    )
    return 1 if (args.strict and hard_fail) else 0


if __name__ == "__main__":
    sys.exit(main())
