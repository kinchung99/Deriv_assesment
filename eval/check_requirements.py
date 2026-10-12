#!/usr/bin/env python3
"""Spec-compliance checker (stdlib only).

    python3 eval/check_requirements.py                    # tickets.json + every eval/datasets/*.json
    python3 eval/check_requirements.py eval/datasets/handwritten.json

Checks every requirement of the brief on every dataset and reports PASS / FAIL
with counts and failing examples:

  RETRIEVAL  R1 1-3 articles per ticket          R2 query = ticket text only
             R3 retrieval separate from generation R4 deterministic retrieval
  RESPONSE   P1 answers directly                  P2 no invented account facts
             P3 no outcome/timeline promises      P4 advice politely refused
             P5 traceable to retrieved content
  TECHNICAL  C1 only local sample files          C2 no external API needed
             C3 separate stages, logged order    C4 no hardcoded answers
             C5 no recommendations/predictions   C6 bounded implementation
             C7 no ticket-id logic

Per-dataset checks (R1, R2, R3-runtime, R4, P1-P5, C5, C7-runtime) run on the
dataset's tickets. Global checks (R3/C3 static + stage log, C1, C2, C4, C6,
C7 static, R4 tie-breaking) run once. The pipeline is used only through its
public API. run_eval.py calls check_dataset() / check_global() and stores the
results in its report; standalone runs write eval/requirements_report.json.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import logging
import os
import random
import re
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support_ai import config  # noqa: E402
from support_ai.grounding import analyze_grounding, extract_citations, find_unsupported_claims  # noqa: E402
from support_ai.loader import load_articles  # noqa: E402
from support_ai.pipeline import STAGES, build_pipeline  # noqa: E402
from support_ai.retriever import BM25Retriever  # noqa: E402
from support_ai.validator import ADVICE_CONTENT_RE, REFUSAL_RE, is_safe_refusal  # noqa: E402

PASS, FAIL, INFO = "PASS", "FAIL", "INFO"
REQUIREMENTS = {
    "R1": "Return the top 1-3 article IDs",
    "R2": "Use the ticket text as the query",
    "R3": "Retrieval separate from reply generation",
    "R4": "Retrieval is deterministic",
    "P1": "Draft answers the question directly",
    "P2": "No invented account-specific facts",
    "P3": "No promises about outcomes or timelines",
    "P4": "Trading advice / profit seeking politely refused",
    "P5": "Answer traceable to retrieved content",
    "C1": "Only the provided local sample files are used",
    "C2": "No external APIs needed for retrieval",
    "C3": "Retrieval and generation are separate steps",
    "C4": "No hardcoded full answers per ticket",
    "C5": "No trading recommendations or market predictions in any reply",
    "C6": "Implementation stays bounded",
    "C7": "No logic depends on exact ticket IDs",
}
OPERATIONAL_ARTICLE = {  # the one KB article covering each operational label (default KB)
    "deposit_issue": "A1",
    "password_reset": "A2",
    "withdrawal_issue": "A3",
    "account_lock": "A4",
    "trading_advice_request": "A5",
}
MAX_EXAMPLES = 12

# ----------------------------------------------------------------------------
# Independent detectors (deliberately not the pipeline's own regexes)
# ----------------------------------------------------------------------------
PROMISE_RE = re.compile(
    r"\b(we|i) (will|'ll) (make sure|ensure|approve|process|credit|release|refund|unlock|fix)\b|"
    r"\bwill (definitely |certainly |surely )?be (approved|processed|credited|released|refunded|"
    r"unlocked|resolved|restored|completed)\b|"
    r"\bguarantee[sd]?\b|\bwe promise\b|\bpromise you\b|"
    r"\b(within|in under|in less than|by) (\d+|a few|one|two|three|tomorrow|today|the end of)\b|"
    r"\b(immediately|right away|asap|instantly)\b|"
    r"\b(definitely|certainly|for sure|rest assured|100%)\b|\b(refund\w*|reimburs\w*|compensat\w*)\b",
    re.IGNORECASE,
)
ACCOUNT_FACT_RE = re.compile(
    r"\byour (balance|account|deposit|withdrawal|payment|funds?|transaction|card|request)( \w+)? "
    r"(is|was|has been|have been|shows?|appears?|is now|got)\b|"
    r"\b(has|have) been (processed|credited|approved|checked|verified|reviewed|sent|received|completed|"
    r"declined|rejected|unlocked|reset)\b|"
    r"\bwe (have |'ve )?(checked|verified|reviewed|confirmed|looked into|located|found|can see)\b|"
    r"[$€£¥]\s?\d|\b\d[\d,.]*\s?(usd|eur|gbp|btc|eth|dollars?|euros?)\b|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
    r"\b(yesterday|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)
PREDICTION_RE = re.compile(
    r"\b(will|is going to|are going to|likely to|expected to|set to) (go up|go down|rise|fall|increase|"
    r"decrease|drop|climb|pump|dump|moon|rally|crash)\b|"
    r"\b(bullish|bearish|buy signal|sell signal|strong buy|price target|good time to (buy|sell))\b|"
    r"\b(you should|i recommend|we recommend|i suggest|we suggest|consider|try) "
    r"(buying|selling|investing|trading|holding|going long|going short|to buy|to sell|to invest)\b|"
    r"\b(best|top) (stock|coin|crypto|asset|pick|investment)s?\b",
    re.IGNORECASE,
)
POLITE_RE = re.compile(r"\b(sorry|apolog\w*|unfortunately|thanks?|thank you|afraid)\b", re.IGNORECASE)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _non_refusal_sentences(reply: str) -> List[str]:
    return [s for s in _SENT_SPLIT.split(reply)
            if not REFUSAL_RE.search(s) and not re.search(r"\b(can't|cannot) confirm\b", s, re.I)]


def independent_promises(reply: str) -> List[str]:
    return [m.group(0) for s in _non_refusal_sentences(reply) for m in PROMISE_RE.finditer(s)]


def _check(status: str, summary: str, counts: Optional[dict] = None, examples: Optional[list] = None, **kw) -> dict:
    out = {"status": status, "summary": summary, "counts": counts or {}, "examples": (examples or [])[:MAX_EXAMPLES]}
    out.update(kw)
    return out


# ----------------------------------------------------------------------------
# Running the pipeline (public API only)
# ----------------------------------------------------------------------------
def _tickets(cases: Sequence[dict]) -> List[dict]:
    return [{"id": c["id"], "message": c["message"], "language": c.get("language", "en")} for c in cases]


def run_tickets(tickets: Sequence[dict], kb_path, wrap=None, **kw):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tickets.json"
        path.write_text(json.dumps(list(tickets), ensure_ascii=False), encoding="utf-8")
        pipeline = build_pipeline(**kw)
        if wrap:
            wrap(pipeline)
        return pipeline.run(path, kb_path, out_dir=None)


class _EventRecorder:
    """Wraps the pipeline's components to record call order (R3/C3 runtime)."""

    def __init__(self, pipeline) -> None:
        self.events: List[str] = []
        self.inside_generate = False
        self.retrieve_during_generate = 0
        rec = self
        retriever, generator = pipeline.retriever, pipeline.generator
        orig_retrieve, orig_score, orig_generate = retriever.retrieve, retriever.score_all, generator.generate

        def retrieve(*a, **k):
            rec.events.append("retrieve")
            if rec.inside_generate:
                rec.retrieve_during_generate += 1
            return orig_retrieve(*a, **k)

        def score_all(*a, **k):
            if rec.inside_generate:
                rec.retrieve_during_generate += 1
            return orig_score(*a, **k)

        def generate(*a, **k):
            rec.events.append("generate")
            rec.inside_generate = True
            try:
                return orig_generate(*a, **k)
            finally:
                rec.inside_generate = False

        retriever.retrieve, retriever.score_all, generator.generate = retrieve, score_all, generate


class _StubLLM:
    """Cycles through every label so the predicted intent varies (R2)."""

    def __init__(self) -> None:
        self.i = 0

    def complete_json(self, system, user, schema):
        label = config.ALLOWED_INTENTS[self.i % len(config.ALLOWED_INTENTS)]
        self.i += 1
        return {"intent": label, "confidence": 0.9}


def _no_id(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "ticket_id"}


# ----------------------------------------------------------------------------
# Per-dataset checks
# ----------------------------------------------------------------------------
def check_dataset(cases: Sequence[dict], kb_path=config.DEFAULT_KB_PATH, base_run=None) -> Dict[str, dict]:
    kb_path = Path(kb_path)
    default_kb = kb_path.resolve() == Path(config.DEFAULT_KB_PATH).resolve()
    tickets = _tickets(cases)
    by_case = {c["id"]: c for c in cases}
    recorder: List[_EventRecorder] = []
    run = run_tickets(tickets, kb_path, wrap=lambda p: recorder.append(_EventRecorder(p)))
    base = base_run or run
    recs = {r["ticket_id"]: r for r in base.records}
    traces = {t.ticket.id: t for t in base.traces}
    kb = base.kb
    out: Dict[str, dict] = {}

    # R1 -------------------------------------------------------------------
    zero = [{"id": i, "category": by_case[i].get("category"), "language": by_case[i].get("language", "en"),
             "message": by_case[i]["message"][:100]} for i, r in recs.items() if len(r["retrieved_articles"]) == 0]
    over = [i for i, r in recs.items() if len(r["retrieved_articles"]) > 3]
    sizes = {}
    for r in recs.values():
        sizes[len(r["retrieved_articles"])] = sizes.get(len(r["retrieved_articles"]), 0) + 1
    # Post-fix pipelines may return a below-threshold best match marked relevant=False.
    fallback_only = [i for i, t in traces.items()
                     if t.hits and not any(getattr(h, "relevant", True) for h in t.hits)]
    zero_cats: Dict[str, int] = {}
    for z in zero:
        zero_cats[z["category"] or "-"] = zero_cats.get(z["category"] or "-", 0) + 1
    out["R1"] = _check(PASS if not zero and not over else FAIL,
                       f"{len(recs) - len(zero) - len(over)}/{len(recs)} tickets have 1-3 articles; "
                       f"{len(zero)} have 0, {len(over)} have >3; {len(fallback_only)} carry only a "
                       f"below-threshold fallback article (relevant=False)",
                       {"tickets": len(recs), "zero": len(zero), "over_3": len(over),
                        "fallback_only_not_relevant": len(fallback_only),
                        "size_histogram": {str(k): v for k, v in sorted(sizes.items())},
                        "zero_by_category": dict(sorted(zero_cats.items(), key=lambda kv: -kv[1]))},
                       zero, all_zero_ids=[z["id"] for z in zero])

    # R2 -------------------------------------------------------------------
    retr = BM25Retriever()
    retr.index(load_articles(kb_path))
    direct_mismatch = []
    for tid, tr in traces.items():
        got = recs[tid]["retrieved_articles"]
        exp = [h.article_id for h in retr.retrieve(tr.ticket.message, config.MAX_RETRIEVED)]
        # pre-fix pipelines skipped retrieval for empty messages
        if exp != got and not (got == [] and not tr.ticket.message.strip()):
            direct_mismatch.append({"id": tid, "pipeline": recs[tid]["retrieved_articles"], "direct": exp})
    variant = [{"id": "V-" + t["id"], "message": t["message"],
                "language": "fr" if (t["language"] or "en").lower() == "en" else "en"} for t in tickets]
    vrun = run_tickets(variant, kb_path)
    vmap = {r["ticket_id"]: r for r in vrun.records}
    lang_mismatch = [t["id"] for t in tickets if vmap["V-" + t["id"]]["retrieved_articles"] != recs[t["id"]]["retrieved_articles"]]
    srun = run_tickets(tickets, kb_path, classifier="llm", llm_client=_StubLLM())
    smap = {r["ticket_id"]: r for r in srun.records}
    intent_changed = sum(1 for t in tickets if smap[t["id"]]["intent"] != recs[t["id"]]["intent"])
    intent_mismatch = [t["id"] for t in tickets if smap[t["id"]]["retrieved_articles"] != recs[t["id"]]["retrieved_articles"]]
    ok = not direct_mismatch and not lang_mismatch and not intent_mismatch
    out["R2"] = _check(PASS if ok else FAIL,
                       f"direct BM25 == pipeline for {len(tickets) - len(direct_mismatch)}/{len(tickets)}; "
                       f"new id + other language: {len(lang_mismatch)} differ; forced different intent "
                       f"({intent_changed} tickets changed intent): {len(intent_mismatch)} differ",
                       {"direct_mismatch": len(direct_mismatch), "id_language_variant_mismatch": len(lang_mismatch),
                        "intent_changed": intent_changed, "intent_variant_mismatch": len(intent_mismatch)},
                       direct_mismatch + [{"id": i, "variant": "language/id"} for i in lang_mismatch]
                       + [{"id": i, "variant": "forced intent"} for i in intent_mismatch])

    # R3 runtime -------------------------------------------------------------
    ev = recorder[0]
    n_retrieve = ev.events.count("retrieve")
    first_gen = ev.events.index("generate") if "generate" in ev.events else len(ev.events)
    retrieve_after_gen = ev.events[first_gen:].count("retrieve")
    nonempty = sum(1 for t in traces.values() if t.ticket.message.strip())
    ok = ev.retrieve_during_generate == 0 and retrieve_after_gen == 0 and n_retrieve in (nonempty, len(traces))
    out["R3"] = _check(PASS if ok else FAIL,
                       f"{n_retrieve} retrieve calls ({len(traces)} tickets, {nonempty} non-empty), all before the first "
                       f"generate call; {ev.retrieve_during_generate} retriever calls made from inside generate()",
                       {"retrieve_calls": n_retrieve, "non_empty_tickets": nonempty,
                        "retrieve_calls_after_first_generate": retrieve_after_gen,
                        "retriever_calls_inside_generate": ev.retrieve_during_generate})

    # R4 -------------------------------------------------------------------
    reruns = [run_tickets(tickets, kb_path) for _ in range(2)]
    rerun_diff = [t["id"] for r in reruns for t, rec in zip(tickets, r.records)
                  if rec["retrieved_articles"] != recs[t["id"]]["retrieved_articles"]]
    shuffled = list(tickets)
    random.Random(13).shuffle(shuffled)
    shmap = {r["ticket_id"]: r for r in run_tickets(shuffled, kb_path).records}
    shuffle_diff = [t["id"] for t in tickets if shmap[t["id"]]["retrieved_articles"] != recs[t["id"]]["retrieved_articles"]]
    shuffle_full_diff = [t["id"] for t in tickets if shmap[t["id"]] != recs[t["id"]]]
    with tempfile.TemporaryDirectory() as tmp:
        rev_kb = Path(tmp) / "kb_reversed.json"
        rev_kb.write_text(json.dumps(list(reversed(json.loads(kb_path.read_text(encoding="utf-8"))))), encoding="utf-8")
        rkmap = {r["ticket_id"]: r for r in run_tickets(tickets, rev_kb).records}
    revkb_diff = [t["id"] for t in tickets if rkmap[t["id"]]["retrieved_articles"] != recs[t["id"]]["retrieved_articles"]]
    ok = not (rerun_diff or shuffle_diff or revkb_diff or shuffle_full_diff)
    out["R4"] = _check(PASS if ok else FAIL,
                       f"2 reruns: {len(rerun_diff)} differ; shuffled ticket order: {len(shuffle_diff)} retrieval "
                       f"/ {len(shuffle_full_diff)} full-record differences; reversed KB order: {len(revkb_diff)} differ",
                       {"rerun_diff": len(rerun_diff), "shuffle_retrieval_diff": len(shuffle_diff),
                        "shuffle_record_diff": len(shuffle_full_diff), "reversed_kb_diff": len(revkb_diff)},
                       rerun_diff + shuffle_diff + shuffle_full_diff + revkb_diff)

    # P1 -------------------------------------------------------------------
    answerable, answered, no_article = 0, 0, []
    p1_fail = []
    for tid, r in recs.items():
        c = by_case[tid]
        intent = c.get("expected_intent") if "expected_intent" in c else None
        if intent is None and c.get("expected_intent", "missing") == "missing":
            intent = r["intent"]  # unlabeled dataset: fall back to prediction
        if intent not in ("deposit_issue", "password_reset", "withdrawal_issue", "account_lock") or not default_kb:
            continue
        if (c.get("language") or "en").lower() not in config.SUPPORTED_LANGUAGES:
            continue
        art = c.get("expect_article") or OPERATIONAL_ARTICLE[intent]
        relevant_ids = [h.article_id for h in traces[tid].hits if getattr(h, "relevant", True)]
        if art not in relevant_ids:
            no_article.append(tid)
            continue
        answerable += 1
        if art in extract_citations(r["reply_draft"]):
            answered += 1
        else:
            p1_fail.append({"id": tid, "expected_intent": intent, "predicted": r["intent"], "article": art,
                            "retrieved": r["retrieved_articles"], "message": c["message"][:100],
                            "reply": r["reply_draft"][:220]})
    out["P1"] = _check(PASS if not p1_fail else FAIL,
                       f"{answered}/{answerable} answerable tickets (relevant article retrieved) cite a sentence from "
                       f"that article; {len(no_article)} operational tickets had no relevant article retrieved (not judged here, see R1)",
                       {"answerable": answerable, "answered_with_cited_relevant_sentence": answered,
                        "relevant_article_not_retrieved": len(no_article)}, p1_fail)

    # P2 / P3 / P5 / C5 ----------------------------------------------------
    p2, p3, p5, c5 = [], [], [], []
    for tid, r in recs.items():
        reply, retrieved = r["reply_draft"], r["retrieved_articles"]
        source = " ".join(kb[a].text for a in retrieved if a in kb)
        claims = find_unsupported_claims(reply, source)
        acct = [c for c in claims if c.startswith(("account-specific fact", "claimed account check"))]
        acct += [m.group(0) for s in _non_refusal_sentences(reply) for m in ACCOUNT_FACT_RE.finditer(s)]
        if acct:
            p2.append({"id": tid, "matches": acct, "reply": reply[:200]})
        prom = [c for c in claims if not c.startswith(("account-specific fact", "claimed account check"))]
        prom += independent_promises(reply)
        if prom:
            p3.append({"id": tid, "matches": prom, "reply": reply[:200]})
        g = analyze_grounding(reply, retrieved, kb)
        bad_cites = [x for x in extract_citations(reply) if x not in retrieved]
        if bad_cites or g.weak_sentences or g.unsupported_framing or g.invalid_citations or g.unsupported_claims:
            p5.append({"id": tid, "citations_not_retrieved": bad_cites, "weak_cited_sentences": g.weak_sentences,
                       "uncited_substantive": g.unsupported_framing, "unsupported_claims": g.unsupported_claims})
        pred = bool(ADVICE_CONTENT_RE.search(reply))
        pred_ind = [m.group(0) for s in _non_refusal_sentences(reply) for m in PREDICTION_RE.finditer(s)]
        if pred or pred_ind:
            c5.append({"id": tid, "matches": pred_ind or ["ADVICE_CONTENT_RE"], "reply": reply[:200]})
    n = len(recs)
    cited = sum(1 for r in recs.values() if extract_citations(r["reply_draft"]))
    out["P2"] = _check(PASS if not p2 else FAIL, f"{n - len(p2)}/{n} replies contain no account-specific facts",
                       {"replies": n, "violations": len(p2)}, p2)
    out["P3"] = _check(PASS if not p3 else FAIL, f"{n - len(p3)}/{n} replies contain no promises/timelines/refunds",
                       {"replies": n, "violations": len(p3)}, p3)
    out["P5"] = _check(PASS if not p5 else FAIL,
                       f"{n - len(p5)}/{n} replies fully traceable ({cited} carry citations; {n - cited} are uncited "
                       f"holding/refusal messages with no KB facts)",
                       {"replies": n, "violations": len(p5), "replies_with_citations": cited}, p5)
    out["C5"] = _check(PASS if not c5 else FAIL, f"{n - len(c5)}/{n} replies contain no recommendation/prediction language",
                       {"replies": n, "violations": len(c5)}, c5)

    # P4 -------------------------------------------------------------------
    advice_ids = [c["id"] for c in cases if c.get("expected_intent") == config.ADVICE_INTENT
                  or "advice_request" in (c.get("expect_flags") or [])
                  or ("acceptable_intents" in c and config.ADVICE_INTENT in c["acceptable_intents"])]
    p4_fail = []
    for tid in advice_ids:
        r = recs[tid]
        refused = is_safe_refusal(r["reply_draft"])
        polite = bool(POLITE_RE.search(r["reply_draft"]))
        if not (refused and polite):
            p4_fail.append({"id": tid, "category": by_case[tid].get("category"), "refused": refused, "polite": polite,
                            "intent": r["intent"], "flags": r["safety_flags"], "message": by_case[tid]["message"][:110],
                            "reply": r["reply_draft"][:160]})
    status = INFO if not advice_ids else (PASS if not p4_fail else FAIL)
    out["P4"] = _check(status, f"{len(advice_ids) - len(p4_fail)}/{len(advice_ids)} labelled advice/profit-seeking "
                               f"tickets politely refused" if advice_ids else "no labelled advice tickets in this dataset",
                       {"advice_cases": len(advice_ids), "not_refused": len(p4_fail)}, p4_fail)

    # C7 runtime -------------------------------------------------------------
    renamed = [{"id": f"ZZ-{i:05d}", "message": t["message"], "language": t["language"]}
               for i, t in enumerate(reversed(tickets), 1)]
    back = {f"ZZ-{i:05d}": t["id"] for i, t in enumerate(reversed(tickets), 1)}
    rmap = {back[r["ticket_id"]]: r for r in run_tickets(renamed, kb_path).records}
    c7_diff = [{"id": t["id"], "field_diffs": sorted(k for k in _no_id(recs[t["id"]]) if recs[t["id"]][k] != rmap[t["id"]][k])}
               for t in tickets if _no_id(recs[t["id"]]) != _no_id(rmap[t["id"]])]
    out["C7"] = _check(PASS if not c7_diff else FAIL,
                       f"renamed every id (ZZ-00001..) and reversed order: {len(c7_diff)}/{len(tickets)} records differ "
                       f"in any field other than ticket_id", {"tickets": len(tickets), "differing": len(c7_diff)}, c7_diff)
    return out


# ----------------------------------------------------------------------------
# Global checks
# ----------------------------------------------------------------------------
SCANNED = sorted((ROOT / "support_ai").glob("*.py")) + [ROOT / "main.py", ROOT / "validate.py"]
NETWORK_MODULES = {"socket", "urllib", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib", "websocket",
                   "ssl", "anthropic", "openai"}


def _imports(path: Path) -> List[tuple]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(a.name, node.lineno) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module, node.lineno))
    return found


def _fallback_import_lines(path: Path) -> set:
    """Line numbers of imports inside `except ImportError` handlers."""
    lines = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                for sub in ast.walk(h):
                    if isinstance(sub, (ast.Import, ast.ImportFrom)):
                        lines.add(sub.lineno)
    return lines


def _is_stdlib(module: str) -> bool:
    top = module.split(".")[0]
    if top in sys.builtin_module_names:
        return True
    try:
        spec = importlib.util.find_spec(top)
    except (ImportError, ValueError):
        return False
    if spec is None or spec.origin is None:
        return False
    stdlib = sysconfig.get_paths()["stdlib"]
    return spec.origin.startswith(stdlib) and "site-packages" not in spec.origin or spec.origin == "frozen"


def _rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


_FILE_IO_RE = re.compile(r"\bopen\(|\.read_text\(|\.read_bytes\(|json\.load\(|\.write_text\(|\.write_bytes\(|os\.listdir|glob\(")


def check_static_and_global(kb_path=config.DEFAULT_KB_PATH) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    sample_tickets = ROOT / "tickets.json"

    # R3 / C3 static ----------------------------------------------------------
    gen_src = (ROOT / "support_ai" / "generator.py").read_text(encoding="utf-8")
    gen_imports = [m for m, _ in _imports(ROOT / "support_ai" / "generator.py")]
    gen_tree = ast.parse(gen_src)
    gen_rel_imports = [(n.module, [a.name for a in n.names]) for n in ast.walk(gen_tree)
                       if isinstance(n, ast.ImportFrom) and n.level > 0]
    retrieval_refs = [m for m in ("retriever", "BM25", ".retrieve(", "score_all", "expand_query", "tokenize")
                      if m in gen_src]
    pipe_src = (ROOT / "support_ai" / "pipeline.py").read_text(encoding="utf-8")
    i_ret, i_gen = pipe_src.find('stage("RETRIEVE_CONTEXT")'), pipe_src.find('stage("GENERATE_REPLY")')
    query_line = "query = tr.ticket.message" in pipe_src
    static_ok = not retrieval_refs and 0 <= i_ret < i_gen and query_line
    static = {"generator_relative_imports": gen_rel_imports, "generator_retrieval_references": retrieval_refs,
              "pipeline_retrieve_stage_before_generate_stage": 0 <= i_ret < i_gen,
              "pipeline_query_is_ticket_message": query_line, "generator_absolute_imports": gen_imports}
    # stage log order from the CLI
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, str(ROOT / "main.py"), "--tickets", str(sample_tickets),
                               "--out-dir", tmp], cwd=str(ROOT), capture_output=True, text=True,
                              env=_offline_env(), timeout=300)
    logged = re.findall(r"\[([A-Z_]+)\] start", proc.stderr)
    log_ok = logged == list(STAGES)
    out["R3"] = _check(PASS if static_ok else FAIL,
                       "generator.py imports nothing retrieval-related and receives hits as an argument; "
                       "pipeline runs RETRIEVE_CONTEXT before GENERATE_REPLY" if static_ok else "static separation violated",
                       static)
    out["C3"] = _check(PASS if static_ok and log_ok else FAIL,
                       f"static separation {'ok' if static_ok else 'VIOLATED'}; CLI logged stage order: {' -> '.join(logged)}",
                       {"logged_stages": logged, "expected": list(STAGES), "main_exit": proc.returncode})

    # C1 ---------------------------------------------------------------------
    net_imports, file_io = [], []
    for p in SCANNED:
        for mod, line in _imports(p):
            if mod.split(".")[0] in NETWORK_MODULES:
                net_imports.append(f"{_rel(p)}:{line} import {mod}")
        for ln, text in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if _FILE_IO_RE.search(text):
                file_io.append(f"{_rel(p)}:{ln}: {text.strip()[:110]}")
    opened = _audit_run(sample_tickets, Path(kb_path))
    reads = sorted({p for p, mode in opened if "r" in mode and "w" not in mode and "a" not in mode})
    writes = sorted({p for p, mode in opened if "w" in mode or "a" in mode or "x" in mode})
    allowed = {str(sample_tickets.resolve()), str(Path(kb_path).resolve())}
    unexpected = [p for p in reads if p not in allowed]
    net_events = [p for p, mode in opened if mode == "NET"]
    only_llm = all("llm_client.py" in x for x in net_imports)
    ok = not unexpected and not net_events and only_llm
    out["C1"] = _check(PASS if ok else FAIL,
                       f"runtime (audit hook) read {len(reads)} file(s): {[Path(r).name for r in reads]}; "
                       f"unexpected reads {len(unexpected)}; network events {len(net_events)}; network-capable imports: "
                       f"{net_imports or 'none'} (anthropic is imported lazily, only by llm_client.py, only for "
                       f"--classifier/--generator llm; off by default and never used for retrieval)",
                       {"runtime_reads": reads, "runtime_writes": writes, "unexpected_reads": unexpected,
                        "network_events": net_events, "network_imports": net_imports},
                       file_io_sites=file_io)

    # C2 ---------------------------------------------------------------------
    snippet = (
        "import json,socket,sys\n"
        "class _NoNet(socket.socket):\n"
        "    def __init__(self,*a,**k): raise OSError('network disabled by eval')\n"
        "socket.socket=_NoNet\n"
        "def _deny(*a,**k): raise OSError('network disabled by eval')\n"
        "socket.create_connection=_deny; socket.getaddrinfo=_deny\n"
        "def _hook(ev,args):\n"
        "    if ev.startswith(('socket.','urllib.','http.')): raise OSError('network event '+ev)\n"
        "sys.addaudithook(_hook)\n"
        "sys.path.insert(0,sys.argv[1])\n"
        "from support_ai.pipeline import build_pipeline\n"
        "r=build_pipeline().run(sys.argv[2],sys.argv[3],out_dir=None)\n"
        "sys.stdout.write(json.dumps(r.records))\n"
    )
    proc = subprocess.run([sys.executable, "-c", snippet, str(ROOT), str(sample_tickets), str(kb_path)],
                          cwd=str(ROOT), capture_output=True, text=True, env=_offline_env(), timeout=300)
    normal = json.dumps(build_pipeline().run(sample_tickets, kb_path, out_dir=None).records)
    same = proc.returncode == 0 and proc.stdout == normal
    out["C2"] = _check(PASS if same else FAIL,
                       f"default pipeline with sockets disabled and no API key: exit {proc.returncode}, "
                       f"output {'identical' if same else 'DIFFERENT'} to a normal run",
                       {"exit_code": proc.returncode, "identical": same, "stderr_tail": proc.stderr[-300:]})

    # C4 ---------------------------------------------------------------------
    pkg = sorted((ROOT / "support_ai").glob("*.py"))
    id_hits, text_hits = [], []
    samples = json.loads(sample_tickets.read_text(encoding="utf-8"))
    shingles = []
    for t in samples:
        words = re.findall(r"[a-z']+", t["message"].lower())
        shingles += [" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))]
    for p in pkg:
        src = p.read_text(encoding="utf-8")
        low = " ".join(re.findall(r"[a-z']+", src.lower()))
        for ln, text in enumerate(src.splitlines(), 1):
            if re.search(r"""['"]T\d+['"]|\bT[1-5]\b""", text):
                id_hits.append(f"{_rel(p)}:{ln}: {text.strip()[:100]}")
        text_hits += [f"{_rel(p)}: '{s}'" for s in shingles if s in low]
    para = [
        ("P-101", "Two hours ago I funded my account with my debit card, yet my balance is still 0. What can I check?", "deposit_issue", "A1"),
        ("P-102", "I can't remember my password and the reset e-mail never shows up.", "password_reset", "A2"),
        ("P-103", "My withdrawal got declined after I completed verification - why?", "withdrawal_issue", "A3"),
        ("P-104", "Which asset is going to rise today? I want to make a profit.", "trading_advice_request", "A5"),
        ("P-105", "After too many failed login attempts my account got locked. What now?", "account_lock", "A4"),
    ]
    prun = run_tickets([{"id": i, "message": m, "language": "en"} for i, m, _, _ in para], kb_path)
    para_res = []
    for (pid, msg, intent, art), r in zip(para, prun.records):
        ok_case = r["intent"] == intent and r["retrieved_articles"][:1] == [art] and art in extract_citations(r["reply_draft"])
        if intent == config.ADVICE_INTENT:
            ok_case = ok_case and is_safe_refusal(r["reply_draft"])
        para_res.append({"id": pid, "ok": ok_case, "intent": r["intent"], "articles": r["retrieved_articles"],
                         "escalate": r["needs_human_escalation"], "flags": r["safety_flags"]})
    ok = not id_hits and not text_hits and all(x["ok"] for x in para_res)
    out["C4"] = _check(PASS if ok else FAIL,
                       f"ticket-id literals in support_ai/: {len(id_hits)}; sample-message 5-word shingles in "
                       f"support_ai/: {len(text_hits)}; paraphrased+renamed samples correct: "
                       f"{sum(x['ok'] for x in para_res)}/{len(para_res)}",
                       {"id_literals": len(id_hits), "sample_text_hits": len(text_hits)},
                       id_hits + text_hits + [x for x in para_res if not x["ok"]], paraphrase_results=para_res)

    # C6 ---------------------------------------------------------------------
    third_party = sorted({f"{mod} ({_rel(p)}{', except-ImportError fallback only' if line in _fallback_import_lines(p) else ''})"
                          for p in SCANNED for mod, line in _imports(p)
                          if mod.split(".")[0] not in ("support_ai",) and not _is_stdlib(mod)})
    loc = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in pkg)
    opt = (ROOT / "requirements-optional.txt").read_text(encoding="utf-8") if (ROOT / "requirements-optional.txt").is_file() else ""
    out["C6"] = _check(INFO,
                       f"{len(pkg)} modules / {loc} lines in support_ai/; non-stdlib imports: {third_party or 'none'}; "
                       f"no database, server, vector store or model download; optional deps: "
                       f"{[l for l in opt.splitlines() if l and not l.startswith('#')]}",
                       {"support_ai_lines": loc, "non_stdlib_imports": third_party})

    # C7 static ----------------------------------------------------------------
    id_logic = []
    allowed_validate = []
    # comparisons of an id against a string literal, or ticket-id-like literals
    pat = re.compile(r"""\b(ticket_id|id)\b['"\]]*\s*(==|!=|in)\s*[(\[{]?\s*['"]|['"]T\d+['"]""")
    for p in SCANNED:
        for ln, text in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(text):
                (allowed_validate if p.name == "validate.py" else id_logic).append(f"{_rel(p)}:{ln}: {text.strip()[:100]}")
    out["C7"] = _check(PASS if not id_logic else FAIL,
                       f"ID-specific code in support_ai/ + main.py: {len(id_logic)}; validate.py spec sample checks "
                       f"(T1/T4, allowed): {len(allowed_validate)}",
                       {"violations": len(id_logic), "validate_sample_checks": len(allowed_validate)},
                       id_logic, allowed=allowed_validate)

    # R4 tie-breaking ---------------------------------------------------------
    kb = json.loads(Path(kb_path).read_text(encoding="utf-8"))
    dup = [dict(kb[0], article_id="A1x"), dict(kb[1], article_id="A0")]  # exact copies of A1 and A2
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, arts in (("forward", kb + dup), ("reversed", list(reversed(kb + dup)))):
            path = Path(tmp) / f"kb_{name}.json"
            path.write_text(json.dumps(arts), encoding="utf-8")
            r = run_tickets([{"id": "D1", "message": samples[0]["message"]}, {"id": "D2", "message": samples[1]["message"]}], path)
            results.append([rec["retrieved_articles"] for rec in r.records])
    tie_ok = results[0] == results[1] and results[0][0][:2] == ["A1", "A1x"] and results[0][1][:2] == ["A0", "A2"]
    out["R4"] = _check(PASS if tie_ok else FAIL,
                       f"KB with exact duplicate articles (A1/A1x, A0/A2), forward vs reversed file order: "
                       f"{results[0]} vs {results[1]} (ties broken by article id)", {"forward": results[0], "reversed": results[1]})
    return out


def _offline_env() -> dict:
    env = dict(os.environ)
    for k in config.LLM_CREDENTIAL_ENVS:
        env.pop(k, None)
    return env


_AUDIT = {"on": False, "events": []}


def _audit_hook(event, args):
    if not _AUDIT["on"]:
        return
    if event == "open":
        path, mode = args[0], args[1]
        if isinstance(path, (str, bytes, os.PathLike)):
            p = os.fsdecode(path)
            if not p.endswith((".py", ".pyc", ".so")):
                _AUDIT["events"].append((str(Path(p).resolve()), mode or "r"))
    elif event.startswith(("socket.", "urllib.", "http.client")):
        _AUDIT["events"].append((event, "NET"))


_HOOKED = False


def _audit_run(tickets_path: Path, kb_path: Path):
    global _HOOKED
    if not _HOOKED:
        sys.addaudithook(_audit_hook)
        _HOOKED = True
    pipeline = build_pipeline()
    _AUDIT["events"] = []
    _AUDIT["on"] = True
    try:
        pipeline.run(tickets_path, kb_path, out_dir=None)
    finally:
        _AUDIT["on"] = False
    return list(_AUDIT["events"])


# ----------------------------------------------------------------------------
# Aggregation / CLI
# ----------------------------------------------------------------------------
def overall(per_dataset: Dict[str, Dict[str, dict]], global_checks: Dict[str, dict]) -> Dict[str, dict]:
    result = {}
    for rid, title in REQUIREMENTS.items():
        statuses = [d[rid]["status"] for d in per_dataset.values() if rid in d]
        if rid in global_checks:
            statuses.append(global_checks[rid]["status"])
        status = FAIL if FAIL in statuses else (PASS if PASS in statuses else INFO)
        failing = [name for name, d in per_dataset.items() if d.get(rid, {}).get("status") == FAIL]
        if global_checks.get(rid, {}).get("status") == FAIL:
            failing.append("global")
        result[rid] = {"title": title, "status": status, "failing_in": failing}
    return result


def print_table(per_dataset: Dict[str, Dict[str, dict]], global_checks: Dict[str, dict]) -> None:
    names = list(per_dataset)
    short = [Path(n).stem[:11] for n in names]
    print("=" * 100)
    print("SPEC COMPLIANCE")
    print(f"{'req':<5}{'overall':<9}" + "".join(f"{s:>12}" for s in short) + f"{'global':>9}")
    ov = overall(per_dataset, global_checks)
    for rid in REQUIREMENTS:
        cells = []
        for n in names:
            c = per_dataset[n].get(rid)
            cells.append(c["status"] if c else "-")
        g = global_checks.get(rid, {}).get("status", "-")
        print(f"{rid:<5}{ov[rid]['status']:<9}" + "".join(f"{c:>12}" for c in cells) + f"{g:>9}  {REQUIREMENTS[rid]}")
    print()
    for rid in REQUIREMENTS:
        for n in names:
            c = per_dataset[n].get(rid)
            if c and c["status"] == FAIL:
                print(f"  {rid} FAIL [{Path(n).stem}] {c['summary']}")
                for ex in c["examples"][:3]:
                    print(f"       e.g. {json.dumps(ex, ensure_ascii=False)[:180]}")
        g = global_checks.get(rid)
        if g:
            print(f"  {rid} {g['status']} [global] {g['summary'][:300]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="*", type=Path)
    ap.add_argument("--kb", type=Path, default=config.DEFAULT_KB_PATH)
    ap.add_argument("--report", type=Path, default=EVAL_DIR / "requirements_report.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.ERROR)
    paths = args.datasets or sorted((EVAL_DIR / "datasets").glob("*.json"))
    per = {}
    for p in paths:
        cases = json.loads(p.read_text(encoding="utf-8"))
        per[_rel(p.resolve()) if p.resolve().is_relative_to(ROOT) else str(p)] = check_dataset(cases, args.kb)
    glob = check_static_and_global(args.kb)
    print_table(per, glob)
    report = {"overall": overall(per, glob), "global": glob, "per_dataset": per}
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
