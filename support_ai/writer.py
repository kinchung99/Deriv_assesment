"""WRITE_RESULTS stage: results.json plus the explainability artifacts.

Output contains no timestamps or run-specific values, so identical inputs
produce byte-identical files.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence

from . import config
from .models import Article, TicketTrace


def _dump(data: object, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_results(records: Sequence[dict], out_dir: Path) -> Path:
    path = out_dir / config.RESULTS_FILENAME
    _dump(list(records), path)
    return path


def trace_to_dict(trace: TicketTrace, kb: Dict[str, Article]) -> dict:
    c, s, d, g, a = trace.classification, trace.signals, trace.draft, trace.grounding, trace.assessment
    result = trace.result.to_dict() if trace.result else {}
    return {
        "ticket_id": trace.ticket.id,
        "message": trace.ticket.message,
        "language": trace.ticket.language,
        "predicted_intent": c.intent if c else None,
        "classifier": {
            "source": c.source,
            "certainty": c.certainty,
            "scores": {k: v for k, v in c.scores.items() if v},
            "matched_rules": c.matched_rules,
            "notes": c.notes,
        }
        if c
        else None,
        "request_signals": s.evidence if s else [],
        "retrieved": [
            {"article_id": h.article_id, "title": h.title, "bm25_score": h.score} for h in trace.hits
        ],
        "reply": {
            "generator": d.source,
            "citations": d.citations,
            "kb_sentences_used": [
                {"article_id": u.article_id, "kb_sentence": u.source, "rendered": u.rendered}
                for u in d.used_sentences
            ],
            "internal_guidance_honoured": d.constraints,
            "notes": d.notes,
        }
        if d
        else None,
        "grounding": {
            "grounded": g.grounded,
            "score": g.score,
            "weak_sentences": g.weak_sentences,
            "unsupported_framing": g.unsupported_framing,
            "unsupported_claims": g.unsupported_claims,
            "invalid_citations": g.invalid_citations,
        }
        if g
        else None,
        "confidence": a.confidence if a else None,
        "confidence_breakdown": a.confidence_breakdown if a else None,
        "safety_flags": result.get("safety_flags", []),
        "needs_human_escalation": result.get("needs_human_escalation"),
        "escalation_reasons": a.escalation_reasons if a else [],
        "decision_explanation": a.explanation if a else None,
        "validation": {"errors": trace.validation_errors, "fixes_applied": trace.validation_fixes},
    }


def write_debug_report(
    traces: Sequence[TicketTrace], kb: Dict[str, Article], run_info: dict, out_dir: Path
) -> List[Path]:
    entries = [trace_to_dict(t, kb) for t in traces]
    json_path = out_dir / config.DEBUG_JSON_FILENAME
    _dump({"run": run_info, "tickets": entries}, json_path)

    lines = [
        "# Debug report",
        "",
        f"Components: classifier=`{run_info['classifier']}`, retriever=`{run_info['retriever']}`, "
        f"generator=`{run_info['generator']}`",
        "",
        "| Ticket | Intent | Retrieved articles | Confidence | Escalate | Flags |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        titles = "<br>".join(f"{r['article_id']} {r['title']}" for r in e["retrieved"]) or "—"
        lines.append(
            f"| {e['ticket_id']} | {e['predicted_intent']} | {titles} | {e['confidence']} | "
            f"{'**yes**' if e['needs_human_escalation'] else 'no'} | "
            f"{', '.join(e['safety_flags']) or '—'} |"
        )
    for e in entries:
        lines += ["", f"## {e['ticket_id']}", "", f"> {e['message'] or '(empty message)'}", ""]
        cls = e["classifier"] or {}
        lines.append(
            f"- **Predicted intent:** `{e['predicted_intent']}` "
            f"(source: {cls.get('source')}, certainty {cls.get('certainty')}, scores {cls.get('scores')})"
        )
        if e["retrieved"]:
            lines.append("- **Retrieved articles:**")
            lines += [
                f"  - {r['article_id']} — {r['title']} (BM25 {r['bm25_score']})" for r in e["retrieved"]
            ]
        else:
            lines.append("- **Retrieved articles:** none above relevance threshold")
        lines.append(f"- **Confidence:** {e['confidence']} — {e['confidence_breakdown']}")
        if e["request_signals"]:
            lines.append(f"- **Request signals:** {'; '.join(e['request_signals'])}")
        if e["reply"] and e["reply"]["internal_guidance_honoured"]:
            lines.append(f"- **KB guidance honoured:** {'; '.join(e['reply']['internal_guidance_honoured'])}")
        lines.append(f"- **Decision:** {e['decision_explanation']}")
        if e["validation"]["errors"] or e["validation"]["fixes_applied"]:
            lines.append(
                f"- **Validation:** errors {e['validation']['errors']}; fixes {e['validation']['fixes_applied']}"
            )
    md_path = out_dir / config.DEBUG_MD_FILENAME
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [json_path, md_path]
