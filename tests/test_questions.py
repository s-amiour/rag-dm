"""Guards on the question files (professor's rules: append-only, evidence quote for every hop)."""
import json
import unittest

from app import core
from app.evaluation import canonical_type, question_hops, gold_chunks

OFFICIAL_KESTREL = [f"q{n:02d}" for n in range(1, 11)]


class QuestionFileTests(unittest.TestCase):
    def test_official_kestrel_questions_still_present_and_first(self):
        ids = [q["id"] for q in core.load_questions("kestrel")]
        self.assertEqual(ids[:10], OFFICIAL_KESTREL)
        self.assertEqual(len(ids), len(set(ids)))

    def test_at_least_three_kestrel_two_hop_questions(self):
        two_hop = [q for q in core.load_questions("kestrel") if canonical_type(q["type"]) == "two_hop"]
        self.assertGreaterEqual(len(two_hop), 3)
        self.assertIn("q08", [q["id"] for q in two_hop])

    def test_every_hop_has_a_quote_that_matches_a_chunk(self):
        records = core.build_chunks("kestrel", 800, 10)
        for q in core.load_questions("kestrel"):
            if q.get("answerable", True):
                hops = question_hops(q)
                self.assertTrue(hops, q["id"])
                for hop in hops:
                    self.assertTrue(gold_chunks(records, hop), f"{q['id']}: no chunk contains {hop['quote']!r}")

    def test_two_hop_questions_have_one_quote_per_hop(self):
        for q in core.load_questions("kestrel"):
            if q["type"] == "two_hop":
                self.assertGreaterEqual(len(q["evidence"]), 2, q["id"])

    def test_own_quotes_match_at_v2_chunking(self):
        records = core.build_chunks("own", 500, 20)
        for q in core.load_questions("own"):
            for hop in question_hops(q):
                self.assertTrue(gold_chunks(records, hop), q["id"])


if __name__ == "__main__":
    unittest.main()


CANONICAL_TYPES = {"exact_identifier", "split_answer", "vocabulary", "not_in_corpus", "versions", "two_hop", "other"}


class OwnQuestionLabelTests(unittest.TestCase):
    """v3T13: own.jsonl uses the assignment's type names, and every relabel is recorded."""

    def setUp(self):
        self.own = core.load_questions("own")

    def test_own_types_are_canonical(self):
        for q in self.own:
            self.assertIn(q["type"], CANONICAL_TYPES, q["id"])

    def test_relabelled_rows_keep_their_old_type_and_are_in_the_changelog(self):
        changelog = (core.DATA / "questions" / "CHANGELOG.md").read_text(encoding="utf-8")
        for q in self.own:
            if "type_original" in q:
                self.assertNotEqual(q["type_original"], q["type"], q["id"])
                self.assertIn(q["id"], changelog)

    def test_own_has_an_unanswerable_question(self):
        unanswerable = [q for q in self.own if q.get("answerable") is False]
        self.assertTrue(unanswerable)
        for q in unanswerable:
            self.assertEqual(q["type"], "not_in_corpus", q["id"])
            self.assertEqual(q.get("evidence", []), [], q["id"])

    def test_two_hop_questions_have_one_quote_per_hop(self):
        for q in self.own:
            if q["type"] == "two_hop":
                self.assertGreaterEqual(len(question_hops(q)), 2, q["id"])