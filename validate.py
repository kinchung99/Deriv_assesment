#!/usr/bin/env python3
"""Validate the pipeline outputs.

    python validate.py              # audit results.json + re-run twice for reproducibility
    python validate.py --out-dir X  # audit outputs written elsewhere

Exits non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from support_ai import config  # noqa: E402
from support_ai.loader import load_articles, load_tickets  # noqa: E402
from support_ai.pipeline import build_pipeline  # noqa: E402
from support_ai.validator import is_safe_refusal, schema_errors, semantic_errors  # noqa: E402

CheckResult = Tuple[str, str]  # (status, detail); status in PASS / FAIL / INFO


class Context:
    def __init__(self, tickets_path: Path, kb_path: Path, out_dir: Path) -> None:
        self.tickets_path = tickets_path
        self.kb_path = kb_path
        self.out_dir = out_dir
        self.results_path = out_dir / config.RESULTS_FILENAME
        self.tickets: list = []
        self.kb: dict = {}
        self.results: list = []


def check_files_exist(ctx: Context) -> CheckResult:
    required = [ctx.tickets_path, ctx.kb_path, ctx.results_path]
    missing = [str(p) for p in required if not p.is_file()]
    debug = [ctx.out_dir / config.DEBUG_JSON_FILENAME, ctx.out_dir / config.DEBUG_MD_FILENAME]
    if not any(p.is_file() for p in debug):
        missing.append("debug_report.json or debug_report.md")
    if missing:
        return "FAIL", f"missing: {missing} (run `python main.py` first)"
    return "PASS", "inputs, results.json and debug report present"


def check_json_valid(ctx: Context) -> CheckResult:
    paths = [ctx.tickets_path, ctx.kb_path, ctx.results_path]
    debug_json = ctx.out_dir / config.DEBUG_JSON_FILENAME
    if debug_json.is_file():
        paths.append(debug_json)
    for path in paths:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return "FAIL", f"{path.name}: {exc}"
    ctx.tickets = load_tickets(ctx.tickets_path)
    ctx.kb = {a.article_id: a for a in load_articles(ctx.kb_path)}
    ctx.results = json.loads(ctx.results_path.read_text(encoding="utf-8"))
    if not isinstance(ctx.results, list):
        return "FAIL", "results.json must be a JSON array"
    return "PASS", f"{len(paths)} files parse as JSON"


def check_one_result_per_ticket(ctx: Context) -> CheckResult:
    ticket_ids = [t.id for t in ctx.tickets]
    result_ids = [r.get("ticket_id") for r in ctx.results if isinstance(r, dict)]
    duplicates = sorted({i for i in result_ids if result_ids.count(i) > 1})
    missing = sorted(set(ticket_ids) - set(result_ids))
    extra = sorted(set(result_ids) - set(ticket_ids))
    if duplicates or missing or extra or len(result_ids) != len(ctx.results):
        return "FAIL", f"duplicates={duplicates} missing={missing} extra={extra}"
    return "PASS", f"{len(ticket_ids)} tickets -> {len(result_ids)} results"


def check_schema(ctx: Context) -> CheckResult:
    problems = []
    for r in ctx.results:
        errors = schema_errors(r, ctx.kb.keys())
        if errors:
            problems.append(f"{r.get('ticket_id') if isinstance(r, dict) else r}: {errors}")
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", f"all {len(ctx.results)} records match the required schema"


def check_article_ids(ctx: Context) -> CheckResult:
    bad = {
        r["ticket_id"]: [a for a in r.get("retrieved_articles", []) if a not in ctx.kb]
        for r in ctx.results
    }
    bad = {k: v for k, v in bad.items() if v}
    sizes_ok = all(1 <= len(r.get("retrieved_articles", [])) <= config.MAX_RETRIEVED for r in ctx.results)
    if bad or not sizes_ok:
        return "FAIL", f"unknown ids {bad}" if bad else f"every result needs 1 to {config.MAX_RETRIEVED} article ids"
    return "PASS", f"1-{config.MAX_RETRIEVED} ids per ticket, all in kb_articles.json ({', '.join(sorted(ctx.kb))})"


def check_safety(ctx: Context) -> CheckResult:
    problems = [f"{r['ticket_id']}: {e}" for r in ctx.results for e in semantic_errors(r, ctx.kb)]
    if problems:
        return "FAIL", "; ".join(problems)
    return "PASS", "citations, refusals, unsupported-claim flags and escalation are consistent"


def check_advice_refused(ctx: Context) -> CheckResult:
    advice = [
        r
        for r in ctx.results
        if r["intent"] == config.ADVICE_INTENT or config.FLAG_ADVICE_REQUEST in r["safety_flags"]
    ]
    unsafe = [r["ticket_id"] for r in advice if not is_safe_refusal(r["reply_draft"])]
    if unsafe:
        return "FAIL", f"advice requests not refused: {unsafe}"
    t4 = next((r for r in ctx.results if r["ticket_id"] == "T4"), None)
    if t4 is not None and (t4["intent"] != config.ADVICE_INTENT or t4 not in advice):
        return "FAIL", f"T4 classified as {t4['intent']}, expected {config.ADVICE_INTENT}"
    note = " (incl. T4)" if t4 is not None else ""
    return "PASS", f"{len(advice)} trading-advice request(s) safely refused{note}"


def check_sample_retrieval(ctx: Context) -> CheckResult:
    t1 = next((r for r in ctx.results if r["ticket_id"] == "T1"), None)
    if t1 is None:
        return "INFO", "T1 not in this ticket set; skipped"
    if "A1" not in t1["retrieved_articles"]:
        return "FAIL", f"T1 retrieved {t1['retrieved_articles']}, expected the deposit article A1"
    return "PASS", "T1 retrieves the card-deposit article A1"


def check_reproducible(ctx: Context) -> CheckResult:
    outputs = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(2):
            out = Path(tmp) / f"run{i}"
            build_pipeline().run(ctx.tickets_path, ctx.kb_path, out)
            outputs.append((out / config.RESULTS_FILENAME).read_bytes())
    if outputs[0] != outputs[1]:
        return "FAIL", "two offline runs produced different results.json"
    on_disk = ctx.results_path.read_bytes()
    if on_disk != outputs[0]:
        return "PASS", (
            "two fresh offline runs are byte-identical (note: results.json on disk differs, "
            "e.g. it was produced with --generator llm or older inputs)"
        )
    return "PASS", "two fresh offline runs are byte-identical and match results.json on disk"


CHECKS: List[Tuple[str, Callable[[Context], CheckResult]]] = [
    ("files exist", check_files_exist),
    ("JSON is valid", check_json_valid),
    ("one result per ticket", check_one_result_per_ticket),
    ("schema respected", check_schema),
    ("article ids valid", check_article_ids),
    ("safety rules", check_safety),
    ("trading advice refused", check_advice_refused),
    ("T1 retrieves deposit article", check_sample_retrieval),
    ("reproducible", check_reproducible),
]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickets", default=str(config.DEFAULT_TICKETS_PATH))
    parser.add_argument("--kb", default=str(config.DEFAULT_KB_PATH))
    parser.add_argument("--out-dir", default=str(config.DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    ctx = Context(Path(args.tickets), Path(args.kb), Path(args.out_dir))
    failed = 0
    for name, check in CHECKS:
        try:
            status, detail = check(ctx)
        except Exception as exc:  # noqa: BLE001 - report, don't crash
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        failed += status == "FAIL"
        print(f"[{status}] {name}: {detail}")
        if status == "FAIL" and name in ("files exist", "JSON is valid"):
            break  # later checks depend on these
    print(f"\n{'ALL CHECKS PASSED' if not failed else f'{failed} CHECK(S) FAILED'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
