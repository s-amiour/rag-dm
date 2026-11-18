"""v3T15: RRF reproduces the two worked examples from the course slides (C8, Session 3, Part IV).

Each retriever returns its top 10, k = 60. The slides only list the chunks that matter; the other
positions are filled with chunks found by one retriever only, at ranks too low to change the result.
"""
import unittest

from app.fusion import rrf


def ranked(ids):
    return [{"id": i, "score": 0.0} for i in ids]


def filler(prefix, ranks):
    return {r: f"{prefix}-filler-{r}" for r in ranks}


def top10(placed: dict[int, str], prefix: str) -> list[dict]:
    slots = {**filler(prefix, range(1, 11)), **placed}
    return ranked([slots[r] for r in range(1, 11)])


class CourseExamplesTest(unittest.TestCase):
    def test_e47_identifier_question(self):
        # "What does error E47 mean?": dense ranks the E47 entry #10, BM25 ranks it #1.
        dense = top10({1: "E46", 2: "E57", 3: "E41-E45", 4: "Troubleshooting intro", 10: "E47"}, "dense")
        bm25 = top10({1: "E47", 2: "E41-E45", 3: "Troubleshooting intro", 4: "Warranty", 5: "E46"}, "bm25")
        fused = rrf({"dense": dense, "bm25": bm25}, k=60)
        order = [h["id"] for h in fused]
        scores = {h["id"]: h["score"] for h in fused}

        self.assertEqual(order[:6], ["E41-E45", "E46", "Troubleshooting intro", "E47", "E57", "Warranty"])
        self.assertAlmostEqual(scores["E47"], 1 / 70 + 1 / 61, places=4)
        self.assertEqual(round(scores["E47"], 4), 0.0307)
        self.assertEqual(round(scores["E41-E45"], 4), 0.0320)
        self.assertEqual(order.index("E47") + 1, 4)  # outside dense's top 5, inside after fusion

    def test_calcium_vocabulary_question(self):
        # "How do I get rid of calcium deposits?": dense ranks descaling #1, BM25 does not find it.
        dense = top10({1: "Descaling procedure", 2: "Water tank and filter", 3: "Cleaning the machine",
                       4: "Unplugging and safety"}, "dense")
        bm25 = top10({1: "Unplugging and safety", 2: "Milk system", 3: "Scheduled switch-on",
                      4: "Cleaning the machine", 6: "Water tank and filter"}, "bm25")
        fused = rrf({"dense": dense, "bm25": bm25}, k=60)
        order = [h["id"] for h in fused]
        scores = {h["id"]: h["score"] for h in fused}

        self.assertEqual(order[:6], ["Unplugging and safety", "Cleaning the machine", "Water tank and filter",
                                     "Descaling procedure", "Milk system", "Scheduled switch-on"])
        self.assertEqual(round(scores["Descaling procedure"], 4), 0.0164)
        self.assertEqual(order.index("Descaling procedure") + 1, 4)  # demoted from #1: a top 3 would lose it

    def test_weight_doubles_a_single_retriever_score(self):
        dense = ranked(["only-dense"])
        plain = rrf({"dense": dense, "bm25": ranked(["x"])}, k=60)
        weighted = rrf({"dense": dense, "bm25": ranked(["x"])}, k=60, weights={"dense": 2.0})
        score = lambda fused: next(h["score"] for h in fused if h["id"] == "only-dense")
        self.assertAlmostEqual(score(weighted), 2 * score(plain))


if __name__ == "__main__":
    unittest.main()
