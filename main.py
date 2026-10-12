#!/usr/bin/env python3
"""Run the support-ticket pipeline.

    python main.py                                   # offline, deterministic
    python main.py --classifier llm --generator llm  # Claude-backed (needs ANTHROPIC_API_KEY)
"""
from __future__ import annotations

import argparse
import logging
import sys

from support_ai import config
from support_ai.loader import DataError
from support_ai.pipeline import build_pipeline


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify, retrieve and draft replies for support tickets.")
    parser.add_argument("--tickets", default=str(config.DEFAULT_TICKETS_PATH), help="path to tickets.json")
    parser.add_argument("--kb", default=str(config.DEFAULT_KB_PATH), help="path to kb_articles.json")
    parser.add_argument("--out-dir", default=str(config.DEFAULT_OUTPUT_DIR), help="where to write outputs")
    parser.add_argument("--classifier", choices=["rules", "llm"], default="rules")
    parser.add_argument("--generator", choices=["template", "llm"], default="template")
    parser.add_argument("--retriever", choices=["bm25"], default="bm25")
    parser.add_argument("-q", "--quiet", action="store_true", help="only print warnings")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    pipeline = build_pipeline(classifier=args.classifier, generator=args.generator, retriever=args.retriever)
    try:
        run = pipeline.run(args.tickets, args.kb, args.out_dir)
    except DataError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{'ticket':<8}{'intent':<24}{'articles':<14}{'conf':>5}  {'escalate':<9}flags")
    for r in run.records:
        print(
            f"{r['ticket_id']:<8}{r['intent']:<24}{','.join(r['retrieved_articles']) or '-':<14}"
            f"{r['confidence']:>5.2f}  {'yes' if r['needs_human_escalation'] else 'no':<9}"
            f"{','.join(r['safety_flags']) or '-'}"
        )
    print("\nWrote: " + ", ".join(p.name for p in run.written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
