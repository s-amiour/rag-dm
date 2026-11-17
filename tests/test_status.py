"""v3T18: /api/status counts chunks at each corpus's evaluated chunking.

core.corpus_status() holds the logic, so most tests run without the web server.
"""
import importlib.util
import unittest
from unittest import mock

from app import core

HAS_FASTAPI = importlib.util.find_spec("fastapi") is not None
SIX_TYPES = {"exact_identifier", "split_answer", "vocabulary", "not_in_corpus", "versions", "two_hop"}


class CorpusStatusTest(unittest.TestCase):
    def test_own_uses_500_20(self):
        s = core.corpus_status("own")
        self.assertEqual((s["chunk_size"], s["overlap"]), (500, 20))
        self.assertEqual(s["chunks"], len(core.build_chunks("own", 500, 20)))

    def test_kestrel_uses_800_10(self):
        s = core.corpus_status("kestrel")
        self.assertEqual((s["chunk_size"], s["overlap"]), (800, 10))
        self.assertEqual(s["chunks"], len(core.build_chunks("kestrel", 800, 10)))

    def test_follows_default_chunking_without_touching_main(self):
        with mock.patch.dict(core.DEFAULT_CHUNKING, {"own": (300, 0)}):
            s = core.corpus_status("own")
        self.assertEqual((s["chunk_size"], s["overlap"], s["chunks"]),
                         (300, 0, len(core.build_chunks("own", 300, 0))))

    def test_questions_by_type(self):
        s = core.corpus_status("kestrel")
        self.assertEqual(sum(s["questions_by_type"].values()), s["questions"])
        self.assertEqual(set(s["questions_by_type"]), SIX_TYPES)  # Kestrel uses exactly the six course labels
        self.assertEqual(list(s["questions_by_type"])[0], "exact_identifier")  # course order


@unittest.skipUnless(HAS_FASTAPI, "fastapi not installed")
class StatusEndpointTest(unittest.TestCase):
    def test_endpoint(self):
        from fastapi.testclient import TestClient
        from app import main

        d = TestClient(main.app).get("/api/status").json()
        self.assertNotIn("deadline_note", d)
        self.assertEqual(d["corpora"]["own"], core.corpus_status("own"))
        self.assertEqual(d["corpora"]["kestrel"]["chunk_size"], 800)


if __name__ == "__main__":
    unittest.main()
