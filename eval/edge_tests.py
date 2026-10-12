#!/usr/bin/env python3
"""Edge-case and functional probes for the pipeline, CLI and validate.py.

    python3 eval/edge_tests.py            # all probes except the slow ReDoS one (~10 s)
    python3 eval/edge_tests.py --slow     # also the 6.4 KB ReDoS probe (~20 s extra)

Each probe prints OK / FINDING / CRASH with an exact repro. Results are also
written to eval/edge_results.json. Nothing outside a temporary directory is
written; the repo files are never touched.
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
from pathlib import Path
from typing import Callable, List, Optional

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent
sys.path.insert(0, str(ROOT))

from support_ai import config  # noqa: E402
from support_ai.pipeline import build_pipeline  # noqa: E402
from support_ai.validator import is_safe_refusal, schema_errors, semantic_errors  # noqa: E402

OK, FINDING, CRASH = "OK", "FINDING", "CRASH"
KB = json.loads(config.DEFAULT_KB_PATH.read_text(encoding="utf-8"))
SAMPLE = json.loads(config.DEFAULT_TICKETS_PATH.read_text(encoding="utf-8"))
RESULTS: List[dict] = []
TMP = Path(tempfile.mkdtemp(prefix="support_ai_edge_"))


def env_offline() -> dict:
    env = dict(os.environ)
    for k in config.LLM_CREDENTIAL_ENVS:
        env.pop(k, None)
    return env


def write(name: str, data, raw: Optional[bytes] = None) -> Path:
    path = TMP / name
    if raw is not None:
        path.write_bytes(raw)
    else:
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def cli(args: List[str], env: Optional[dict] = None, script: str = "main.py", timeout: int = 600):
    start = time.perf_counter()
    proc = subprocess.run([sys.executable, str(ROOT / script)] + args, cwd=str(ROOT), capture_output=True,
                          text=True, env=env or env_offline(), timeout=timeout)
    return proc, time.perf_counter() - start


def tail(text: str, n: int = 1) -> str:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    return " | ".join(lines[-n:])[:300]


def probe(pid: str, area: str, desc: str):
    def deco(fn: Callable[[], tuple]):
        def run():
            try:
                status, detail, repro = fn()
            except Exception as exc:  # noqa: BLE001
                status, detail, repro = CRASH, f"probe raised {type(exc).__name__}: {exc}", ""
            RESULTS.append({"id": pid, "area": area, "probe": desc, "status": status, "detail": detail, "repro": repro})
            print(f"[{status:<7}] {pid:<4} {area:<10} {desc}\n           {detail}")
        run.pid = pid
        return run
    return deco


def ticket_cli_probe(name: str, data=None, raw: Optional[bytes] = None, expect_exit=(0, 2)):
    path = write(name, data, raw)
    out = TMP / ("out_" + name.replace(".", "_"))
    proc, secs = cli(["-q", "--tickets", str(path), "--out-dir", str(out)])
    clean = proc.returncode in expect_exit and "Traceback" not in proc.stderr
    detail = f"exit {proc.returncode} in {secs:.2f}s; stderr: {tail(proc.stderr) or '-'}"
    repro = f"python3 main.py -q --tickets {name} --out-dir /tmp/out   # {name} = {raw[:80]!r}" if raw is not None else \
        f"python3 main.py -q --tickets {name} --out-dir /tmp/out   # {name} = {json.dumps(data, ensure_ascii=True)[:160]}"
    return proc, out, clean, detail, repro


# ----------------------------------------------------------------------------
# Input files
# ----------------------------------------------------------------------------
PROBES = []


def P(pid, area, desc):
    def deco(fn):
        PROBES.append(probe(pid, area, desc)(fn))
        return fn
    return deco


@P("I01", "input", "ticket missing 'message'")
def _():
    proc, _, clean, d, r = ticket_cli_probe("no_message.json", [{"id": "X1", "language": "en"}])
    return (OK if clean and proc.returncode == 2 else FINDING), d, r


@P("I02", "input", "ticket missing 'id'")
def _():
    proc, _, clean, d, r = ticket_cli_probe("no_id.json", [{"message": "hi"}])
    return (OK if clean and proc.returncode == 2 else FINDING), d, r


@P("I03", "input", "duplicate ids, incl. 'T1' vs ' T1 ' and 1 vs '1'")
def _():
    res = []
    for name, data in (("dup1.json", [{"id": "A", "message": "x"}, {"id": "A", "message": "y"}]),
                       ("dup2.json", [{"id": "T1", "message": "x"}, {"id": " T1 ", "message": "y"}]),
                       ("dup3.json", [{"id": 1, "message": "x"}, {"id": "1", "message": "y"}])):
        proc, _, clean, d, r = ticket_cli_probe(name, data)
        res.append(proc.returncode == 2 and clean)
    return (OK if all(res) else FINDING), f"rejected with exit 2: {res}", "python3 main.py --tickets dup*.json"


@P("I04", "input", "non-list JSON (object) / scalar / invalid JSON / missing file / empty file")
def _():
    outs = []
    for name, data, raw in (("obj.json", {"id": "X", "message": "m"}, None), ("scalar.json", 42, None),
                            ("bad.json", None, b"[{\"id\": \"X\", "), ("empty.json", None, b"")):
        proc, _, clean, d, r = ticket_cli_probe(name, data, raw)
        outs.append(f"{name}: exit {proc.returncode}{'' if clean else ' TRACEBACK'}")
    proc, _ = cli(["-q", "--tickets", str(TMP / "does_not_exist.json"), "--out-dir", str(TMP / "o")])
    outs.append(f"missing: exit {proc.returncode}{'' if 'Traceback' not in proc.stderr else ' TRACEBACK'}")
    bad = [o for o in outs if "TRACEBACK" in o or "exit 2" not in o]
    return (OK if not bad else FINDING), "; ".join(outs), "python3 main.py --tickets obj.json|scalar.json|bad.json|empty.json"


@P("I05", "input", "null / numeric / list / dict message")
def _():
    data = [{"id": "N1", "message": None}, {"id": "N2", "message": 12345}, {"id": "N3", "message": ["deposit", "missing"]},
            {"id": "N4", "message": {"text": "my deposit is missing"}}]
    proc, out, clean, d, r = ticket_cli_probe("odd_messages.json", data)
    if proc.returncode != 0:
        return CRASH, d, r
    res = json.loads((out / "results.json").read_text())
    summary = [(x["ticket_id"], x["intent"], x["safety_flags"]) for x in res]
    # N3/N4: str(list)/str(dict) is classified as a real ticket - questionable but not a crash
    return FINDING if res[2]["intent"] != "other" or res[3]["intent"] != "other" else OK, \
        f"accepted (no type check): {summary}", r


@P("I06", "input", "non-string ids: int, bool, float, null")
def _():
    data = [{"id": 5, "message": "I forgot my password"}, {"id": True, "message": "x"}, {"id": 1.0, "message": "y"}]
    proc, out, clean, d, r = ticket_cli_probe("ids.json", data)
    res = json.loads((out / "results.json").read_text()) if proc.returncode == 0 else []
    proc2, _, clean2, d2, r2 = ticket_cli_probe("null_id.json", [{"id": None, "message": "x"}])
    return OK if proc.returncode == 0 and proc2.returncode == 2 and clean2 else FINDING, \
        f"ids coerced to {[x['ticket_id'] for x in res]}; null id -> {d2}", r


@P("I07", "input", "language variants: null, 'EN', 'en-US', 'en_GB', 'english', 5, missing")
def _():
    langs = [None, "EN", "en-US", "en_GB", "english", 5, "MISSING"]
    data = []
    for i, l in enumerate(langs):
        t = {"id": f"L{i}", "message": "I forgot my password and the reset email never came"}
        if l != "MISSING":
            t["language"] = l
        data.append(t)
    proc, out, clean, d, r = ticket_cli_probe("langs.json", data)
    res = json.loads((out / "results.json").read_text())
    flagged = {str(langs[i]): x["needs_human_escalation"] for i, x in enumerate(res)}
    bad = [k for k in ("en-US", "en_GB", "english") if flagged[k]]
    return (FINDING if bad else OK), f"escalated (unsupported_language) per language value: {flagged}", r


@P("I08", "input", "array element is a string / nested array")
def _():
    proc, _, clean, d, r = ticket_cli_probe("elem.json", ["just a string"])
    return (OK if proc.returncode == 2 and clean else FINDING), d, r


@P("I09", "input", "empty ticket list []")
def _():
    proc, out, clean, d, r = ticket_cli_probe("empty_list.json", [])
    vproc, _ = cli(["--tickets", str(TMP / "empty_list.json"), "--out-dir", str(out)], script="validate.py")
    return (OK if proc.returncode == 0 else FINDING), \
        f"{d}; results.json={ (out / 'results.json').read_text().strip() if (out / 'results.json').exists() else 'none'}; " \
        f"validate.py exit {vproc.returncode}: {tail(vproc.stdout)}", r


@P("I10", "input", "UTF-8 file with BOM (e.g. saved by Windows Notepad)")
def _():
    raw = b"\xef\xbb\xbf" + json.dumps(SAMPLE).encode("utf-8")
    proc, _, clean, d, r = ticket_cli_probe("bom.json", raw=raw)
    return (OK if proc.returncode == 0 else FINDING), d + " (valid tickets rejected)" if proc.returncode else d, \
        "printf '\\xef\\xbb\\xbf' > bom.json; cat tickets.json >> bom.json; python3 main.py --tickets bom.json"


@P("I11", "input", "Latin-1 encoded file (non-UTF-8 bytes)")
def _():
    raw = json.dumps([{"id": "X1", "message": "Mon dépôt est en retard"}], ensure_ascii=False).encode("latin-1")
    proc, _, clean, d, r = ticket_cli_probe("latin1.json", raw=raw)
    status = OK if clean and proc.returncode == 2 else (CRASH if "Traceback" in proc.stderr else FINDING)
    return status, d, "python3 -c \"open('l1.json','wb').write('[{\\\"id\\\":\\\"X\\\",\\\"message\\\":\\\"d\\u00e9p\\u00f4t\\\"}]'.encode('latin-1'))\"; python3 main.py --tickets l1.json"


@P("I12", "input", "deeply nested JSON (100k levels)")
def _():
    raw = b"[" * 100000 + b"]" * 100000
    proc, _, clean, d, r = ticket_cli_probe("deep.json", raw=raw)
    status = OK if clean and proc.returncode == 2 else (CRASH if "Traceback" in proc.stderr else FINDING)
    return status, d, "python3 -c \"open('deep.json','w').write('['*100000+']'*100000)\"; python3 main.py --tickets deep.json"


@P("I13", "input", "lone UTF-16 surrogate in message (JSON escape \\ud800)")
def _():
    raw = b'[{"id": "S1", "message": "my deposit \\ud800 is missing", "language": "en"}]'
    proc, out, clean, d, r = ticket_cli_probe("surrogate_msg.json", raw=raw)
    status = OK if proc.returncode == 0 else (CRASH if "Traceback" in proc.stderr else FINDING)
    return status, d, "echo '[{\"id\":\"S1\",\"message\":\"my deposit \\ud800 is missing\"}]' > s.json; python3 main.py --tickets s.json --out-dir /tmp/o"


@P("I14", "input", "lone surrogate in ticket id")
def _():
    raw = b'[{"id": "S\\udfff", "message": "I forgot my password", "language": "en"}]'
    proc, out, clean, d, r = ticket_cli_probe("surrogate_id.json", raw=raw)
    status = OK if proc.returncode == 0 else (CRASH if "Traceback" in proc.stderr else FINDING)
    return status, d, "echo '[{\"id\":\"S\\udfff\",\"message\":\"I forgot my password\"}]' > s.json; python3 main.py --tickets s.json --out-dir /tmp/o"


@P("I15", "input", "NUL, zero-width, NBSP, RTL and emoji-only messages")
def _():
    data = [{"id": "U1", "message": "deposit\u0000missing"}, {"id": "U2", "message": "​​​"},
            {"id": "U3", "message": "  "}, {"id": "U4", "message": "‮ticket reversed"},
            {"id": "U5", "message": "\U0001F525\U0001F525"}]
    proc, out, clean, d, r = ticket_cli_probe("unicode.json", data)
    if proc.returncode != 0:
        return CRASH, d, r
    res = {x["ticket_id"]: x for x in json.loads((out / "results.json").read_text())}
    zw_empty = "empty_message" in res["U2"]["safety_flags"]
    return (OK if zw_empty else FINDING), \
        f"U2 zero-width-only flagged empty_message={zw_empty} (flags {res['U2']['safety_flags']}); NBSP-only flags {res['U3']['safety_flags']}; " \
        f"NUL msg intent {res['U1']['intent']}", r


@P("I16", "input", "markdown/HTML in ticket id breaks debug_report.md table")
def _():
    data = [{"id": "A|B<script>x</script>", "message": "I forgot my password"}]
    proc, out, clean, d, r = ticket_cli_probe("md_id.json", data)
    md = (out / "debug_report.md").read_text()
    row = next(l for l in md.splitlines() if l.startswith("| A|B"))
    return (FINDING if row.count("|") != 7 else OK), f"table row has {row.count('|')} pipes (7 expected): {row[:90]}", r


@P("I17", "input", "very long ticket id (10k chars) and 100 KB benign message")
def _():
    msg = ("I deposited money with my card yesterday and it is still not showing. " * 1500)[:100000]
    data = [{"id": "X" * 10000, "message": "I forgot my password"}, {"id": "LONG", "message": msg}]
    proc, out, clean, d, r = ticket_cli_probe("long.json", data)
    return (OK if proc.returncode == 0 else CRASH), d, r


# ----------------------------------------------------------------------------
# Scale / performance
# ----------------------------------------------------------------------------
def _rows(n):
    base = [t["message"] for t in SAMPLE] + ["Love the app!", "", "Mi cuenta está bloqueada"]
    return [{"id": f"R{i:05d}", "message": base[i % len(base)], "language": "es" if i % len(base) == 7 else "en"}
            for i in range(n)]


@P("S01", "scale", "1,000 and 10,000 ticket rows (wall time incl. writing outputs)")
def _():
    times = {}
    for n in (1000, 10000):
        path = write(f"rows{n}.json", _rows(n))
        proc, secs = cli(["-q", "--tickets", str(path), "--out-dir", str(TMP / f"out_rows{n}")])
        times[n] = (proc.returncode, round(secs, 2), (TMP / f"out_rows{n}" / "debug_report.json").stat().st_size)
    return OK, f"(exit, seconds, debug_report.json bytes): {times}", "python3 main.py -q --tickets rows1000.json --out-dir /tmp/o"


@P("S02", "scale", "ReDoS: 3.2 KB ticket 'when my when my ...' (account-specific regex)")
def _():
    path = write("redos3k.json", [{"id": "R1", "message": "when my " * 400}])
    proc, secs = cli(["-q", "--tickets", str(path), "--out-dir", str(TMP / "o_redos")])
    return (FINDING if secs > 1.0 else OK), \
        f"exit {proc.returncode}, {secs:.2f}s for ONE 3,200-char ticket (cubic: 6.4 KB ~21 s, 12.8 KB ~170 s)", \
        "python3 -c \"import json;json.dump([{'id':'R1','message':'when my '*400}],open('r.json','w'))\"; time python3 main.py -q --tickets r.json --out-dir /tmp/o"


def _slow_redos():
    path = write("redos6k.json", [{"id": "R1", "message": "when my " * 800}])
    proc, secs = cli(["-q", "--tickets", str(path), "--out-dir", str(TMP / "o_redos6")])
    return (FINDING if secs > 1.0 else OK), f"exit {proc.returncode}, {secs:.2f}s for one 6,400-char ticket", \
        "same as S02 with 'when my '*800"


# ----------------------------------------------------------------------------
# Knowledge base variants
# ----------------------------------------------------------------------------
def kb_run(name, articles=None, raw=None, tickets=None):
    kb = write(name, articles, raw)
    tk = write("kbt_" + name, tickets or SAMPLE)
    out = TMP / ("out_" + name.replace(".", "_"))
    proc, secs = cli(["-q", "--tickets", str(tk), "--kb", str(kb), "--out-dir", str(out)])
    return proc, out, kb, tk


@P("K01", "kb", "KB with an extra relevant article A6 + validate.py --kb")
def _():
    a6 = {"article_id": "A6", "title": "Withdrawal processing times",
          "body": "Approved withdrawals are sent to the original payment method. Processing can take several business days depending on the provider."}
    tickets = SAMPLE + [{"id": "W1", "message": "How long does a withdrawal take to reach my bank after approval?"}]
    proc, out, kb, tk = kb_run("kb_a6.json", KB + [a6], tickets=tickets)
    res = {x["ticket_id"]: x for x in json.loads((out / "results.json").read_text())}
    v, _ = cli(["--tickets", str(tk), "--kb", str(kb), "--out-dir", str(out)], script="validate.py")
    return (OK if proc.returncode == 0 and v.returncode == 0 else FINDING), \
        f"W1 -> {res['W1']['retrieved_articles']} esc={res['W1']['needs_human_escalation']}; validate.py exit {v.returncode}: {tail(v.stdout)}", \
        "python3 main.py --kb kb_a6.json --tickets t.json --out-dir /tmp/o && python3 validate.py --kb kb_a6.json --tickets t.json --out-dir /tmp/o"


@P("K02", "kb", "KB article with empty body / empty article_id / id with space")
def _():
    arts = KB + [{"article_id": "A7", "title": "Empty", "body": ""}, {"article_id": "", "title": "Deposit limits", "body": "Deposit limits apply per day."},
                 {"article_id": "A 8", "title": "Deposit currencies", "body": "Deposits are accepted in USD and EUR."}]
    proc, out, kb, tk = kb_run("kb_weird.json", arts,
                                tickets=[{"id": "Q1", "message": "what are the deposit limits per day"},
                                         {"id": "Q2", "message": "which currencies can I deposit in, USD or EUR?"}])
    if proc.returncode != 0:
        return CRASH, f"exit {proc.returncode}: {tail(proc.stderr)}", ""
    res = json.loads((out / "results.json").read_text())
    return FINDING, f"empty article_id accepted by loader; results: " + \
        "; ".join(f"{x['ticket_id']} arts={x['retrieved_articles']} grounded={x['grounded']} reply={x['reply_draft'][:90]!r}" for x in res), \
        "kb_weird.json = kb_articles.json + {'article_id':'', ...} + {'article_id':'A 8', ...}"


@P("K03", "kb", "empty KB [] and non-list KB")
def _():
    p1, *_ = kb_run("kb_empty.json", [])
    p2, *_ = kb_run("kb_obj.json", {"article_id": "A1"})
    ok = p1.returncode == 2 and p2.returncode == 2 and "Traceback" not in p1.stderr + p2.stderr
    return (OK if ok else FINDING), f"empty -> exit {p1.returncode} ({tail(p1.stderr)}); object -> exit {p2.returncode}", ""


@P("K04", "kb", "KB without the A5 trading policy article")
def _():
    proc, out, *_ = kb_run("kb_no_a5.json", [a for a in KB if a["article_id"] != "A5"])
    t4 = next(x for x in json.loads((out / "results.json").read_text()) if x["ticket_id"] == "T4")
    return (OK if is_safe_refusal(t4["reply_draft"]) and t4["needs_human_escalation"] else FINDING), \
        f"T4 refused={is_safe_refusal(t4['reply_draft'])} escalated={t4['needs_human_escalation']} flags={t4['safety_flags']}", ""


@P("K05", "kb", "KB article that itself contains a promise ('guaranteed within 24 hours')")
def _():
    arts = [dict(a) for a in KB]
    arts[2]["body"] += " Withdrawals are guaranteed to arrive within 24 hours."
    proc, out, *_ = kb_run("kb_promise.json", arts)
    t3 = next(x for x in json.loads((out / "results.json").read_text()) if x["ticket_id"] == "T3")
    has = "within 24 hours" in t3["reply_draft"]
    return (FINDING if has else OK), \
        f"promise copied into reply={has}; flags={t3['safety_flags']} esc={t3['needs_human_escalation']} " \
        f"(by design KB-backed claims are allowed; the reply then contradicts the agent-review closer)", ""


# ----------------------------------------------------------------------------
# CLI flags and LLM fallbacks
# ----------------------------------------------------------------------------
@P("L01", "cli", "--generator llm / --classifier llm without API key")
def _():
    base, _ = cli(["-q", "--out-dir", str(TMP / "llm_base")])
    outs = {}
    for flags in (["--generator", "llm"], ["--classifier", "llm"], ["--classifier", "llm", "--generator", "llm"]):
        d = TMP / ("llm_" + "_".join(flags[1::2]))
        p, _ = cli(["-q", "--out-dir", str(d)] + flags)
        same = (d / "results.json").read_bytes() == (TMP / "llm_base" / "results.json").read_bytes()
        outs[" ".join(flags)] = (p.returncode, same)
    ok = all(v == (0, True) for v in outs.values())
    return (OK if ok else FINDING), f"(exit, results.json identical to offline): {outs}", \
        "env -u ANTHROPIC_API_KEY python3 main.py --classifier llm --generator llm --out-dir /tmp/o"


@P("L02", "cli", "bogus ANTHROPIC_API_KEY with SDK not installed (Python 3.9)")
def _():
    env = env_offline()
    env["ANTHROPIC_API_KEY"] = "sk-bogus"
    p, _ = cli(["-q", "--classifier", "llm", "--generator", "llm", "--out-dir", str(TMP / "llm_bogus")], env=env)
    return (OK if p.returncode == 0 else FINDING), f"exit {p.returncode}; stderr: {tail(p.stderr, 2)}", \
        "ANTHROPIC_API_KEY=sk-bogus python3 main.py --classifier llm --generator llm"


@P("L03", "cli", "unknown --classifier / --out-dir pointing at an existing file / unwritable dir")
def _():
    p1, _ = cli(["--classifier", "foo"])
    f = write("iam_a_file.txt", None, b"x")
    p2, _ = cli(["-q", "--out-dir", str(f)])
    ro = TMP / "readonly_dir"
    ro.mkdir(exist_ok=True)
    ro.chmod(0o500)
    p3, _ = cli(["-q", "--out-dir", str(ro / "sub")])
    ro.chmod(0o700)
    res = {"--classifier foo": p1.returncode, "--out-dir <file>": (p2.returncode, "Traceback" in p2.stderr),
           "--out-dir <unwritable>": (p3.returncode, "Traceback" in p3.stderr)}
    tb = "Traceback" in p2.stderr or "Traceback" in p3.stderr
    return (FINDING if tb else OK), f"{res}; e.g. {tail(p2.stderr)}", \
        "touch f; python3 main.py --out-dir f   # and: mkdir ro; chmod 500 ro; python3 main.py --out-dir ro/sub"


class _Stub:
    def __init__(self, fn):
        self.fn = fn

    def complete_json(self, system, user, schema):
        return self.fn(system, user, schema)


def _api_run(tickets, **kw):
    path = write("api_tickets.json", tickets)
    return build_pipeline(**kw).run(path, config.DEFAULT_KB_PATH, out_dir=None)


@P("L04", "llm", "stub LLM: off-label intent, exception, advice reply, promise reply")
def _():
    out = []
    r = _api_run(SAMPLE, classifier="llm", llm_client=_Stub(lambda s, u, sc: {"intent": "refund", "confidence": 1}))
    out.append(f"off-label -> {[x['intent'] for x in r.records]}")

    def boom(s, u, sc):
        raise RuntimeError("boom")
    r = _api_run(SAMPLE, classifier="llm", generator="llm", llm_client=_Stub(boom))
    out.append(f"exception -> intents {[x['intent'] for x in r.records]}")
    adv = _api_run([SAMPLE[3]], generator="llm",
                   llm_client=_Stub(lambda s, u, sc: {"reply": "You should buy gold, it will go up today [A5].", "answerable": True}))
    out.append(f"advice reply -> refused={is_safe_refusal(adv.records[0]['reply_draft'])}")
    prom = _api_run([SAMPLE[2]], generator="llm",
                    llm_client=_Stub(lambda s, u, sc: {"reply": "Your withdrawal will be approved within 24 hours [A3].", "answerable": True}))
    rp = prom.records[0]["reply_draft"]
    out.append(f"promise reply -> kept={'24 hours' in rp}")
    other = _api_run([SAMPLE[3]], classifier="llm", llm_client=_Stub(lambda s, u, sc: {"intent": "other", "confidence": 0.99}))
    out.append(f"LLM says T4 is 'other' -> refused={is_safe_refusal(other.records[0]['reply_draft'])} flags={other.records[0]['safety_flags']}")
    ok = "refused=True" in out[2] and "kept=False" in out[3] and "refused=True" in out[4]
    return (OK if ok else FINDING), "; ".join(out), "see tests/test_llm_components.py for the StubClient pattern"


# ----------------------------------------------------------------------------
# validate.py
# ----------------------------------------------------------------------------
@P("V01", "validate", "validate.py against outputs written with --out-dir (default and custom tickets)")
def _():
    out = TMP / "v_default"
    cli(["-q", "--out-dir", str(out)])
    v1, _ = cli(["--out-dir", str(out)], script="validate.py")
    custom = write("v_custom.json", [{"id": "C1", "message": "my account is locked"}, {"id": "C2", "message": "which coin will pump?"}])
    out2 = TMP / "v_custom"
    cli(["-q", "--tickets", str(custom), "--out-dir", str(out2)])
    v2, _ = cli(["--tickets", str(custom), "--out-dir", str(out2)], script="validate.py")
    v3, _ = cli(["--out-dir", str(out2)], script="validate.py")  # wrong tickets file on purpose
    ok = v1.returncode == 0 and v2.returncode == 0 and v3.returncode != 0
    return (OK if ok else FINDING), \
        f"default: exit {v1.returncode} ({tail(v1.stdout)}); custom: exit {v2.returncode}; custom outputs vs default tickets: exit {v3.returncode} (expected non-zero)", \
        "python3 main.py -q --out-dir /tmp/o && python3 validate.py --out-dir /tmp/o"


@P("V02", "validate", "validate.py on a tampered but schema-valid results.json")
def _():
    out = TMP / "v_tamper"
    cli(["-q", "--out-dir", str(out)])
    res = json.loads((out / "results.json").read_text())
    res[1]["intent"] = "other"          # T2 password_reset -> other
    res[0]["confidence"] = 0.12         # T1 confidence changed
    res[4]["retrieved_articles"] = ["A4", "A1"]
    (out / "results.json").write_text(json.dumps(res, indent=2) + "\n")
    v, _ = cli(["--out-dir", str(out)], script="validate.py")
    repro_line = next((l for l in v.stdout.splitlines() if "reproducible" in l), "")
    return (FINDING if v.returncode == 0 else OK), \
        f"validate.py exit {v.returncode}; {repro_line.strip()[:200]}", \
        "python3 main.py -q --out-dir /tmp/o; edit /tmp/o/results.json (T2 intent->other, T1 confidence->0.12); python3 validate.py --out-dir /tmp/o"


@P("V03", "validate", "validate.py with results.json missing a record / extra field")
def _():
    out = TMP / "v_broken"
    cli(["-q", "--out-dir", str(out)])
    res = json.loads((out / "results.json").read_text())
    res = res[:-1]
    res[0]["extra"] = 1
    (out / "results.json").write_text(json.dumps(res))
    v, _ = cli(["--out-dir", str(out)], script="validate.py")
    return (OK if v.returncode != 0 and "Traceback" not in v.stderr else FINDING), f"exit {v.returncode}: {tail(v.stdout, 1)}", ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--out", type=Path, default=EVAL_DIR / "edge_results.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.CRITICAL)
    print(f"temp dir: {TMP}")
    for run in PROBES:
        run()
    if args.slow:
        probe("S03", "scale", "ReDoS: 6.4 KB ticket (slow)")(_slow_redos)()
    counts = {s: sum(r["status"] == s for r in RESULTS) for s in (OK, FINDING, CRASH)}
    print(f"\n{counts}")
    args.out.write_text(json.dumps({"counts": counts, "probes": RESULTS}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
