"""Stage orchestration.

LOAD_DATA -> INDEX_KB -> CLASSIFY_INTENT -> RETRIEVE_CONTEXT -> GENERATE_REPLY
          -> VALIDATE_OUTPUT -> WRITE_RESULTS

The pipeline only talks to the Classifier / Retriever / Generator protocols,
so each component can be swapped independently.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

from . import config
from .classifier import RuleClassifier
from .grounding import analyze_grounding
from .interfaces import Classifier, Generator, Retriever
from .loader import load_articles, load_tickets
from .models import Article, TicketResult, TicketTrace
from .policy import analyze_request, assess
from .retriever import BM25Retriever
from .validator import enforce
from .writer import write_debug_report, write_results

log = logging.getLogger(__name__)

STAGES = (
    "LOAD_DATA",
    "INDEX_KB",
    "CLASSIFY_INTENT",
    "RETRIEVE_CONTEXT",
    "GENERATE_REPLY",
    "VALIDATE_OUTPUT",
    "WRITE_RESULTS",
)


@contextmanager
def stage(name: str) -> Iterator[None]:
    start = time.perf_counter()
    log.info("[%s] start", name)
    yield
    log.info("[%s] done in %.1f ms", name, (time.perf_counter() - start) * 1000)


@dataclass
class PipelineRun:
    records: List[dict]
    traces: List[TicketTrace]
    kb: Dict[str, Article]
    run_info: dict
    written: List[Path] = field(default_factory=list)


class Pipeline:
    def __init__(self, classifier: Classifier, retriever: Retriever, generator: Generator) -> None:
        self.classifier = classifier
        self.retriever = retriever
        self.generator = generator
        # KB articles are topic-tagged with the deterministic rules so that
        # classifier/retrieval agreement can be checked whatever classifier runs.
        self._topic_tagger = RuleClassifier()

    def run(
        self,
        tickets_path: Union[str, Path] = config.DEFAULT_TICKETS_PATH,
        kb_path: Union[str, Path] = config.DEFAULT_KB_PATH,
        out_dir: Optional[Union[str, Path]] = config.DEFAULT_OUTPUT_DIR,
    ) -> PipelineRun:
        with stage("LOAD_DATA"):
            tickets = load_tickets(tickets_path)
            articles = load_articles(kb_path)
            log.info("loaded %d tickets, %d articles", len(tickets), len(articles))

        with stage("INDEX_KB"):
            self.retriever.index(articles)
            kb = {a.article_id: a for a in articles}
            article_topics = {a.article_id: self._topic_tagger.classify_text(a.text).intent for a in articles}

        traces = [TicketTrace(ticket=t) for t in tickets]

        with stage("CLASSIFY_INTENT"):
            for tr in traces:
                tr.classification = self.classifier.classify(tr.ticket)
                tr.signals = analyze_request(tr.ticket, tr.classification)

        with stage("RETRIEVE_CONTEXT"):
            for tr in traces:
                query = tr.ticket.message  # ticket text only; intent is not used
                tr.hits = self.retriever.retrieve(query, config.MAX_RETRIEVED) if query.strip() else []

        with stage("GENERATE_REPLY"):
            for tr in traces:
                tr.draft = self.generator.generate(
                    tr.ticket, tr.classification, tr.signals, tr.hits, kb, article_topics
                )

        with stage("VALIDATE_OUTPUT"):
            records = []
            for tr in traces:
                retrieved = [h.article_id for h in tr.hits]
                tr.grounding = analyze_grounding(tr.draft.reply, retrieved, kb)
                tr.assessment = assess(
                    tr.ticket, tr.classification, tr.signals, tr.hits, tr.draft, tr.grounding, kb, article_topics
                )
                candidate = TicketResult(
                    ticket_id=tr.ticket.id,
                    intent=tr.classification.intent,
                    retrieved_articles=retrieved,
                    reply_draft=tr.draft.reply,
                    grounded=tr.grounding.grounded,
                    confidence=tr.assessment.confidence,
                    needs_human_escalation=tr.assessment.needs_human_escalation,
                    safety_flags=tr.assessment.safety_flags,
                ).to_dict()
                record, tr.validation_errors, tr.validation_fixes = enforce(candidate, kb)
                if tr.validation_errors:
                    log.warning("%s: validation issues %s -> %s", tr.ticket.id, tr.validation_errors, tr.validation_fixes)
                tr.result = TicketResult(**record)
                records.append(record)

        run_info = {
            "classifier": getattr(self.classifier, "name", type(self.classifier).__name__),
            "retriever": getattr(self.retriever, "name", type(self.retriever).__name__),
            "generator": getattr(self.generator, "name", type(self.generator).__name__),
            "stages": list(STAGES),
        }
        run = PipelineRun(records=records, traces=traces, kb=kb, run_info=run_info)

        with stage("WRITE_RESULTS"):
            if out_dir is not None:
                out = Path(out_dir)
                out.mkdir(parents=True, exist_ok=True)
                run.written.append(write_results(records, out))
                run.written.extend(write_debug_report(traces, kb, run_info, out))
                log.info("wrote %s", ", ".join(str(p) for p in run.written))
        return run


def build_pipeline(classifier: str = "rules", generator: str = "template", retriever: str = "bm25", llm_client=None) -> Pipeline:
    """Factory that maps CLI names to implementations."""
    if retriever != "bm25":
        raise ValueError(f"unknown retriever '{retriever}'")
    retriever_impl = BM25Retriever()

    if classifier == "rules":
        classifier_impl = RuleClassifier()
    elif classifier == "llm":
        from .classifier import LLMClassifier

        classifier_impl = LLMClassifier(client=llm_client)
    else:
        raise ValueError(f"unknown classifier '{classifier}'")

    if generator == "template":
        from .generator import TemplateGenerator

        generator_impl = TemplateGenerator()
    elif generator == "llm":
        from .generator import LLMGenerator

        generator_impl = LLMGenerator(client=llm_client)
    else:
        raise ValueError(f"unknown generator '{generator}'")
    return Pipeline(classifier_impl, retriever_impl, generator_impl)
