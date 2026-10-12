"""Shared helpers for the test suite."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List

from support_ai import config
from support_ai.pipeline import PipelineRun, build_pipeline


def run_on_messages(messages: List[str], language: str = "en", **pipeline_kwargs) -> PipelineRun:
    """Run the pipeline on ad-hoc tickets against the real KB, without writing outputs."""
    tickets = [{"id": f"X{i}", "message": m, "language": language} for i, m in enumerate(messages, 1)]
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tickets.json"
        path.write_text(json.dumps(tickets), encoding="utf-8")
        return build_pipeline(**pipeline_kwargs).run(path, config.DEFAULT_KB_PATH, out_dir=None)


def by_id(run: PipelineRun) -> Dict[str, dict]:
    return {r["ticket_id"]: r for r in run.records}
