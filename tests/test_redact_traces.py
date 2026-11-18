"""v4T09 + v4T02: the trace store never receives personal data from the question.

Uses the offline API fixtures (fake retriever, fake LLM, temporary trace store).
"""
import importlib.util
import os
import unittest
from pathlib import Path

from app import traces
from tests.api_support import ApiTestCase

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
QUESTION = "I'm Sara Klein (sara.klein@acme.com, +33 6 12 34 56 78): what is error E47?"


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class TraceRedactionTest(ApiTestCase):
    def test_trace_stores_the_redacted_question_and_counts(self):
        r = self.ask(corpus="own", question=QUESTION, retrievers=["bm25"])
        self.assertEqual(r.status_code, 200, r.text)
        [root] = traces.get_trace(r.json()["trace_id"])["spans"]
        stored = root["attributes"]["question"]
        self.assertEqual(stored, "I'm [NAME_1] ([EMAIL_1], [PHONE_1]): what is error E47?")
        self.assertEqual(root["attributes"]["redactions"], {"EMAIL": 1, "PHONE": 1, "NAME": 1})

    def test_no_personal_data_anywhere_in_the_store_file(self):
        self.ask(corpus="own", question=QUESTION, retrievers=["bm25"])
        raw = Path(os.environ["TRACES_DB"]).read_bytes()
        for secret in ("Sara Klein", "sara.klein@acme.com", "+33 6 12 34 56 78"):
            self.assertNotIn(secret.encode(), raw)


if __name__ == "__main__":
    unittest.main()
