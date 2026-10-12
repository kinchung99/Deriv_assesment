"""End-to-end checks on the provided sample data (the required minimum tests)."""
import json
import tempfile
import unittest
from pathlib import Path

from support_ai import config
from support_ai.loader import load_tickets
from support_ai.pipeline import STAGES, build_pipeline
from support_ai.validator import is_safe_refusal, schema_errors

KB_IDS = {"A1", "A2", "A3", "A4", "A5"}


class SamplePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        cls.pipeline_run = build_pipeline().run(config.DEFAULT_TICKETS_PATH, config.DEFAULT_KB_PATH, cls.out)
        cls.results = json.loads((cls.out / config.RESULTS_FILENAME).read_text())
        cls.by_id = {r["ticket_id"]: r for r in cls.results}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_t4_is_refused_as_trading_advice(self):
        t4 = self.by_id["T4"]
        self.assertEqual(t4["intent"], "trading_advice_request")
        self.assertIn("advice_request", t4["safety_flags"])
        self.assertTrue(is_safe_refusal(t4["reply_draft"]))
        self.assertIn("A5", t4["retrieved_articles"])
        self.assertNotRegex(t4["reply_draft"].lower(), r"\b(buy|will go up|you should invest)\b")

    def test_t1_retrieves_deposit_article(self):
        t1 = self.by_id["T1"]
        self.assertEqual(t1["intent"], "deposit_issue")
        self.assertEqual(t1["retrieved_articles"][0], "A1")

    def test_output_matches_schema(self):
        self.assertEqual(len(self.results), 5)
        for record in self.results:
            self.assertEqual(list(record), list(config.RESULT_SCHEMA), record["ticket_id"])
            self.assertEqual(schema_errors(record, KB_IDS), [], record["ticket_id"])
            self.assertTrue(1 <= len(record["retrieved_articles"]) <= 3)

    def test_one_result_per_ticket_in_input_order(self):
        ticket_ids = [t.id for t in load_tickets(config.DEFAULT_TICKETS_PATH)]
        self.assertEqual([r["ticket_id"] for r in self.results], ticket_ids)

    def test_expected_intents_and_escalation(self):
        expected = {
            "T1": ("deposit_issue", "A1", False),
            "T2": ("password_reset", "A2", False),
            "T3": ("withdrawal_issue", "A3", True),
            "T4": ("trading_advice_request", "A5", False),
            "T5": ("account_lock", "A4", False),
        }
        for tid, (intent, article, escalate) in expected.items():
            r = self.by_id[tid]
            self.assertEqual(r["intent"], intent, tid)
            self.assertEqual(r["retrieved_articles"][0], article, tid)
            self.assertEqual(r["needs_human_escalation"], escalate, tid)
            self.assertTrue(r["grounded"], tid)

    def test_withdrawal_is_policy_sensitive_and_makes_no_promise(self):
        t3 = self.by_id["T3"]
        self.assertIn("policy_sensitive", t3["safety_flags"])
        self.assertIn("account_specific_request", t3["safety_flags"])
        self.assertNotRegex(t3["reply_draft"].lower(), r"will be (approved|processed)|within \d")

    def test_replies_cite_only_retrieved_articles(self):
        for r in self.results:
            for cited in set(__import__("re").findall(r"\[(A\d+)\]", r["reply_draft"])):
                self.assertIn(cited, r["retrieved_articles"], r["ticket_id"])

    def test_debug_report_explains_each_ticket(self):
        report = json.loads((self.out / config.DEBUG_JSON_FILENAME).read_text())
        self.assertEqual(report["run"]["stages"], list(STAGES))
        for entry in report["tickets"]:
            self.assertTrue(entry["predicted_intent"])
            self.assertTrue(all(r["title"] for r in entry["retrieved"]))
            self.assertRegex(entry["decision_explanation"], r"^(Escalated|Not escalated)")
        self.assertTrue((self.out / config.DEBUG_MD_FILENAME).is_file())

    def test_runs_are_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_pipeline().run(config.DEFAULT_TICKETS_PATH, config.DEFAULT_KB_PATH, tmp)
            again = (Path(tmp) / config.RESULTS_FILENAME).read_bytes()
        self.assertEqual(again, (self.out / config.RESULTS_FILENAME).read_bytes())


if __name__ == "__main__":
    unittest.main()
