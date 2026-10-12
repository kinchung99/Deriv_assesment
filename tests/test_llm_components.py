"""LLM components with a stubbed client: constrained labels and safe fallback."""
import unittest
from unittest import mock

from support_ai import config
from support_ai.classifier import LLMClassifier
from support_ai.generator import LLMGenerator
from support_ai.models import Ticket
from support_ai.prompts import build_generator_prompt

from _support import by_id, run_on_messages


class StubClient:
    """Returns canned JSON (or raises) and records the prompts it was sent."""

    def __init__(self, classify=None, generate=None):
        self.classify = classify
        self.generate = generate
        self.calls = []

    def complete_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        is_classifier = "intent" in schema["properties"]
        response = self.classify if is_classifier else self.generate
        if isinstance(response, Exception):
            raise response
        if callable(response):
            return response(user)
        return response


class LLMClassifierTest(unittest.TestCase):
    def test_valid_label_is_used(self):
        stub = StubClient(classify={"intent": "password_reset", "confidence": 0.9})
        c = LLMClassifier(client=stub).classify(Ticket("X", "I forgot my password"))
        self.assertEqual((c.intent, c.source, c.certainty), ("password_reset", "llm", 0.9))
        self.assertEqual(stub.calls[0][2]["properties"]["intent"]["enum"], list(config.ALLOWED_INTENTS))

    def test_off_label_output_falls_back_to_rules(self):
        stub = StubClient(classify={"intent": "refund", "confidence": 0.99})
        c = LLMClassifier(client=stub).classify(Ticket("X", "I forgot my password"))
        self.assertEqual((c.intent, c.source), ("password_reset", "rules_fallback"))

    def test_api_error_falls_back_to_rules(self):
        c = LLMClassifier(client=StubClient(classify=RuntimeError("boom"))).classify(
            Ticket("X", "My withdrawal was declined")
        )
        self.assertEqual((c.intent, c.source), ("withdrawal_issue", "rules_fallback"))

    def test_disagreement_halves_certainty(self):
        stub = StubClient(classify={"intent": "deposit_issue", "confidence": 0.8})
        c = LLMClassifier(client=stub).classify(Ticket("X", "I forgot my password"))
        self.assertEqual(c.certainty, 0.4)


class LLMGeneratorTest(unittest.TestCase):
    def test_grounded_llm_reply_is_accepted(self):
        reply = (
            "Card deposits are usually instant, but may be delayed by issuer checks or pending review [A1]. "
            "Please confirm successful payment and check transaction history [A1]."
        )
        stub = StubClient(generate={"reply": reply, "cited_article_ids": ["A1"], "answerable": True})
        r = run_on_messages(["My card deposit is not showing"], generator="llm", llm_client=stub).records[0]
        self.assertEqual(r["reply_draft"], reply)
        self.assertTrue(r["grounded"])
        _, user, _ = stub.calls[0]
        self.assertIn('<snippet id="A1"', user)  # the prompt carries the retrieved snippets

    def test_hallucinated_promise_falls_back_to_template(self):
        bad = {"reply": "Your deposit will be credited within 2 hours [A1].", "cited_article_ids": ["A1"], "answerable": True}
        run = run_on_messages(["My card deposit is not showing"], generator="llm", llm_client=StubClient(generate=bad))
        r, trace = run.records[0], run.traces[0]
        self.assertNotIn("2 hours", r["reply_draft"])
        self.assertEqual(trace.draft.source, "template_fallback")
        self.assertTrue(any("grounding" in n for n in trace.draft.notes))

    def test_advice_not_refused_falls_back_to_refusal(self):
        bad = {"reply": "Gold is likely to go up, so consider buying it [A5].", "cited_article_ids": ["A5"], "answerable": True}
        r = run_on_messages(["Which asset will go up today?"], generator="llm", llm_client=StubClient(generate=bad)).records[0]
        self.assertIn("not able to provide", r["reply_draft"])

    def test_invalid_json_or_error_falls_back(self):
        run = run_on_messages(["I forgot my password"], generator="llm", llm_client=StubClient(generate=ValueError("bad json")))
        self.assertEqual(run.traces[0].draft.source, "template_fallback")
        self.assertTrue(run.records[0]["grounded"])

    def test_no_credentials_means_offline_fallback(self):
        with mock.patch("support_ai.llm_client.get_default_client", return_value=None):
            run = run_on_messages(["I forgot my password"], classifier="llm", generator="llm")
        trace = run.traces[0]
        self.assertEqual(trace.classification.source, "rules_fallback")
        self.assertEqual(trace.draft.source, "template_fallback")
        self.assertEqual(by_id(run)["X1"]["intent"], "password_reset")
        self.assertTrue(by_id(run)["X1"]["grounded"])

    def test_prompt_instructs_model_to_stay_within_snippets(self):
        from support_ai.loader import load_articles
        from support_ai.models import RequestSignals, RetrievalHit

        kb = {a.article_id: a for a in load_articles(config.DEFAULT_KB_PATH)}
        system, user, _ = build_generator_prompt(
            "help", "deposit_issue", RequestSignals(), [RetrievalHit("A1", kb["A1"].title, 3.0)], kb
        )
        self.assertIn("only the help-center snippets", system)
        self.assertIn("Never promise", system)
        self.assertIn(kb["A1"].body, user)


if __name__ == "__main__":
    unittest.main()
