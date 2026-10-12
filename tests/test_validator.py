"""Each validation rule rejects (and safely repairs) a bad record."""
import unittest

from support_ai import config
from support_ai.loader import load_articles
from support_ai.validator import enforce, schema_errors, semantic_errors

KB = {a.article_id: a for a in load_articles(config.DEFAULT_KB_PATH)}


def good_record(**overrides):
    record = {
        "ticket_id": "T1",
        "intent": "deposit_issue",
        "retrieved_articles": ["A1"],
        "reply_draft": "Card deposits are usually instant, but may be delayed by issuer checks [A1].",
        "grounded": True,
        "confidence": 0.9,
        "needs_human_escalation": False,
        "safety_flags": [],
    }
    record.update(overrides)
    return record


class SchemaTest(unittest.TestCase):
    def test_good_record_passes(self):
        self.assertEqual(schema_errors(good_record(), KB), [])
        self.assertEqual(semantic_errors(good_record(), KB), [])

    def test_missing_field(self):
        record = good_record()
        del record["grounded"]
        self.assertTrue(any("missing" in e for e in schema_errors(record, KB)))

    def test_bad_intent(self):
        self.assertTrue(schema_errors(good_record(intent="refund_request"), KB))

    def test_unknown_article_id(self):
        self.assertTrue(any("unknown" in e for e in schema_errors(good_record(retrieved_articles=["A9"]), KB)))

    def test_confidence_out_of_range_or_wrong_type(self):
        for value in (1.5, -0.1, float("nan"), True, "0.5"):
            self.assertTrue(schema_errors(good_record(confidence=value), KB), value)

    def test_hard_failure_is_replaced_with_safe_fallback(self):
        record, errors, fixes = enforce(good_record(confidence=1.5), KB)
        self.assertTrue(errors)
        self.assertEqual(schema_errors(record, KB), [])
        self.assertTrue(record["needs_human_escalation"])
        self.assertIn(config.FLAG_VALIDATION_FAILED, record["safety_flags"])


class SemanticTest(unittest.TestCase):
    def test_trading_advice_must_be_refused(self):
        bad = good_record(
            intent="trading_advice_request",
            retrieved_articles=["A5"],
            reply_draft="Gold will go up today, you should buy it [A5].",
        )
        self.assertTrue(any("not safely refused" in e for e in semantic_errors(bad, KB)))
        fixed, _, fixes = enforce(bad, KB)
        self.assertTrue(fixes)
        self.assertNotIn("buy", fixed["reply_draft"].lower())
        self.assertIn("advice_request", fixed["safety_flags"])

    def test_unsupported_promise_triggers_flag_and_escalation(self):
        bad = good_record(reply_draft="Your deposit will be credited within 2 hours [A1].")
        self.assertTrue(any("unsupported claims" in e for e in semantic_errors(bad, KB)))
        fixed, _, _ = enforce(bad, KB)
        self.assertIn(config.FLAG_UNSUPPORTED_CLAIM, fixed["safety_flags"])
        self.assertTrue(fixed["needs_human_escalation"])

    def test_citation_of_unretrieved_article(self):
        bad = good_record(reply_draft="Withdrawals may be declined [A3].")
        self.assertTrue(any("not retrieved" in e for e in semantic_errors(bad, KB)))
        fixed, _, _ = enforce(bad, KB)
        self.assertFalse(fixed["grounded"])
        self.assertTrue(fixed["needs_human_escalation"])

    def test_risk_flags_require_escalation(self):
        bad = good_record(safety_flags=["policy_sensitive"])
        fixed, errors, _ = enforce(bad, KB)
        self.assertTrue(errors)
        self.assertTrue(fixed["needs_human_escalation"])

    def test_quoted_policy_wording_is_not_an_unsupported_claim(self):
        ok = good_record(
            intent="trading_advice_request",
            retrieved_articles=["A5"],
            reply_draft="We're not able to provide market predictions, profit guarantees, or asset recommendations [A5].",
            safety_flags=["advice_request"],
        )
        self.assertEqual(semantic_errors(ok, KB), [])


if __name__ == "__main__":
    unittest.main()
