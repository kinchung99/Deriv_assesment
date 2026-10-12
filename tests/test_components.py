"""Unit tests for the retriever and classifier, including paraphrased tickets."""
import unittest

from support_ai import config
from support_ai.classifier import RuleClassifier
from support_ai.loader import load_articles
from support_ai.retriever import BM25Retriever


class RetrieverTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = BM25Retriever()
        cls.retriever.index(load_articles(config.DEFAULT_KB_PATH))

    def top(self, query):
        hits = self.retriever.retrieve(query)
        return hits[0].article_id if hits else None

    def test_paraphrases_hit_the_right_article(self):
        cases = {
            "My card top-up has not arrived in my balance yet": "A1",
            "The password reset link never shows up in my inbox": "A2",
            "My payout got rejected, why?": "A3",
            "I'm locked out after entering the wrong sign in details several times": "A4",
            "What stock should I buy to make money this week?": "A5",
        }
        for query, expected in cases.items():
            self.assertEqual(self.top(query), expected, query)

    def test_retrieval_is_deterministic_and_bounded(self):
        query = "I made a deposit but my withdrawal was declined after too many login attempts"
        first = self.retriever.retrieve(query)
        self.assertEqual(first, self.retriever.retrieve(query))
        self.assertLessEqual(len(first), config.MAX_RETRIEVED)
        scores = [h.score for h in first]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_irrelevant_query_returns_nothing(self):
        self.assertEqual(self.retriever.retrieve("asdf qwerty zxcv"), [])
        self.assertEqual(self.retriever.retrieve("What are your office opening hours?"), [])

    def test_retrieve_before_index_raises(self):
        with self.assertRaises(RuntimeError):
            BM25Retriever().retrieve("deposit")


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        self.classifier = RuleClassifier()

    def test_paraphrases(self):
        cases = {
            "Topped up with my debit card, money isn't showing": "deposit_issue",
            "Can't remember my password and the recovery email never comes": "password_reset",
            "My cash out was rejected": "withdrawal_issue",
            "Account blocked after several failed attempts to sign in": "account_lock",
            "Which crypto should I buy for guaranteed profit?": "trading_advice_request",
            "How do I update my mailing address?": "other",
        }
        for text, expected in cases.items():
            self.assertEqual(self.classifier.classify_text(text).intent, expected, text)

    def test_labels_are_always_allowed_and_certainty_bounded(self):
        for text in ["", "deposit", "deposit withdrawal password lock profit", "???"]:
            c = self.classifier.classify_text(text)
            self.assertIn(c.intent, config.ALLOWED_INTENTS)
            self.assertGreaterEqual(c.certainty, 0.0)
            self.assertLessEqual(c.certainty, 1.0)

    def test_unknown_text_is_other_with_zero_certainty(self):
        c = self.classifier.classify_text("hello there")
        self.assertEqual((c.intent, c.certainty), ("other", 0.0))


if __name__ == "__main__":
    unittest.main()
