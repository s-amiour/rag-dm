"""v3T21: a question is "worse" only when a gold chunk leaves the top k (k = 5 by default).

Synthetic result pairs, built in the shape evaluate_corpus produces: no evaluation run needed.
"""
import unittest

from app import evaluation as ev
from app import evaluate as cli_eval


def row(qid, ranks, qtype="exact_identifier"):
    """ranks: the best rank of each hop's gold chunk (None = not retrieved at all)."""
    found = lambda n: sum(1 for r in ranks if r is not None and r <= n) / len(ranks)
    out = {"id": qid, "question": f"question {qid}", "type": qtype, "answerable": True,
           "first_gold_rank": min((r for r in ranks if r is not None), default=None)}
    out.update({f"recall_at_{n}": found(n) for n in ev.CUTOFFS})
    return out


def result(*rows, corpus="kestrel", label="dense"):
    return {"results": list(rows),
            "summary": {"corpus": corpus, "label": label, "config": {"top_k": 10}}}


class RegressionRuleTest(unittest.TestCase):
    def test_rank_2_to_3_inside_top_5_is_moved_not_worse(self):
        out = ev.regressions(result(row("a", [2])), result(row("a", [3])))
        self.assertEqual(out["worse"], [])
        self.assertEqual([r["id"] for r in out["moved"]], ["a"])

    def test_rank_3_to_7_leaves_the_top_5_and_is_worse(self):
        out = ev.regressions(result(row("a", [3])), result(row("a", [7])))
        self.assertEqual([(r["id"], r["rank_before"], r["rank_after"]) for r in out["worse"]], [("a", 3, 7)])
        self.assertEqual((out["worse"][0]["recall_before"], out["worse"][0]["recall_after"]), (1.0, 0.0))

    def test_two_hop_second_hop_dropping_out_is_worse_even_if_first_hop_keeps_its_rank(self):
        before = result(row("q08", [1, 4], qtype="two_hop"))
        after = result(row("q08", [1, 9], qtype="two_hop"))  # the old rule compared only rank 1 -> 1
        out = ev.regressions(before, after)
        self.assertEqual([r["id"] for r in out["worse"]], ["q08"])
        self.assertEqual((out["worse"][0]["recall_before"], out["worse"][0]["recall_after"]), (1.0, 0.5))

    def test_recall_going_up_is_better(self):
        out = ev.regressions(result(row("a", [7])), result(row("a", [2])))
        self.assertEqual([r["id"] for r in out["better"]], ["a"])
        self.assertEqual(out["worse"], [])

    def test_unchanged_question_is_in_no_list(self):
        out = ev.regressions(result(row("a", [2])), result(row("a", [2])))
        self.assertEqual((out["worse"], out["better"], out["moved"]), ([], [], []))

    def test_k_must_be_a_stored_cutoff(self):
        with self.assertRaises(ValueError):
            ev.regressions(result(row("a", [2])), result(row("a", [3])), k=4)

    def test_k_changes_what_counts_as_worse(self):
        out = ev.regressions(result(row("a", [2])), result(row("a", [7])), k=10)
        self.assertEqual(out["worse"], [])  # rank 7 is still inside the top 10
        self.assertEqual(out["k"], 10)

    def test_unanswerable_questions_are_ignored(self):
        q = {**row("u", [1]), "answerable": False}
        out = ev.regressions(result(q), result({**q, "first_gold_rank": None}))
        self.assertEqual((out["worse"], out["better"], out["moved"]), ([], [], []))


class RegressionReportTest(unittest.TestCase):
    def test_summary_shows_the_three_lists_and_the_k(self):
        dense = result(row("w", [3]), row("b", [7]), row("m", [2]), label="dense")
        hybrid = result(row("w", [8]), row("b", [1]), row("m", [4]), label="dense+bm25")
        text = cli_eval.regression_report([("kestrel", "dense", dense), ("kestrel", "dense+bm25", hybrid)])
        self.assertIn("1 worse, 1 better, 1 moved within the top 5", text)
        self.assertIn("_worse: a gold chunk left the top 5_", text)
        self.assertIn("- `w`", text.split("_better")[0])
        self.assertIn("- `b`", text.split("_better")[1].split("_moved")[0])
        self.assertIn("- `m`", text.split("_moved")[1])


if __name__ == "__main__":
    unittest.main()
