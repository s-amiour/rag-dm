"""T11: the Words tab uses the same tokenizer as BM25 (bm25.analyse)."""
import unittest
from unittest import mock

from app import bm25, core

RECORDS = [
    {"id": "catalog:0", "document": "service-catalog.md", "index": 0,
     "text": "| Messaging | Inès Moreau | #oncall-messaging |"},
    {"id": "runbook:0", "document": "support-runbook-errors.md", "index": 0,
     "text": "KST-5031: escalate to the team that owns postroom."},
]


class WordsTokenizerTest(unittest.TestCase):
    def setUp(self):
        bm25.set_stopwords(True)
        patcher = mock.patch.object(core, "build_chunks", return_value=RECORDS)
        patcher.start()
        self.addCleanup(patcher.stop)

    def vocab(self):
        return {v["word"] for v in core.words_for_corpus("kestrel", 800, 10)["vocabulary"]}

    def test_accented_name_is_one_folded_word(self):
        words = self.vocab()
        self.assertIn("ines", words)          # before T11: "in" and "s"
        self.assertNotIn("s", words)

    def test_identifier_kept_whole_and_split(self):
        words = self.vocab()
        self.assertTrue({"kst-5031", "kst", "5031"} <= words)

    def test_stop_words_not_in_vocabulary(self):
        self.assertNotIn("the", self.vocab())

    def test_vocabulary_matches_bm25_tokens(self):
        expected = set()
        for r in RECORDS:
            expected.update(bm25.tokenize(r["text"]))
        self.assertEqual(self.vocab(), expected)

    def test_question_mode_present_absent_ignored(self):
        out = core.question_words("kestrel", "Who is Inès for KST-5031 and my phone?", 800, 10)
        present = {w["word"] for w in out["present"]}
        self.assertTrue({"ines", "kst-5031"} <= present)
        self.assertIn("phone", out["absent"])
        self.assertTrue({"who", "is", "for", "and", "my"} <= set(out["ignored"]))

    def test_tokenize_unchanged_by_refactor(self):
        self.assertEqual(bm25.tokenize("KST-5031"), ["kst-5031", "kst", "5031"])
        self.assertEqual(bm25.tokenize("What does E47 mean?"), ["e47", "mean"])


if __name__ == "__main__":
    unittest.main()