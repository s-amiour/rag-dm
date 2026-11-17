"""v3T09: the summary is rebuilt from every run file in the folder, so a single run never erases the others.

Uses one real BM25 run (no model, no LLM) and copies of it relabelled as other configurations,
all in a temporary folder.
"""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

from app import evaluate as cli_eval

_BASE = None


def base_result():
    global _BASE
    if _BASE is None:
        _, _BASE, _ = cli_eval.run_and_save("kestrel", ["bm25"], out_dir=Path(tempfile.mkdtemp()))
    return _BASE


def write_run(out: Path, corpus: str, name: str, run_at: str = "2026-11-17 10:00:00") -> None:
    r = copy.deepcopy(base_result())
    r["summary"].update({"corpus": corpus, "label": name, "run_at": run_at})
    cli_eval.save(r, out, corpus, name)


def configurations(out: Path) -> list[tuple[str, str]]:
    rows = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return [(r["corpus"], r["configuration"]) for r in rows]


class RebuildSummaryTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())

    def test_two_files_then_a_third_all_present_and_ordered(self):
        write_run(self.out, "kestrel", "dense+bm25")
        write_run(self.out, "kestrel", "dense")
        cli_eval.rebuild_summary(self.out)
        self.assertEqual(configurations(self.out), [("kestrel", "dense"), ("kestrel", "dense+bm25")])

        write_run(self.out, "kestrel", "dense+bm25+graph")
        cli_eval.rebuild_summary(self.out)
        self.assertEqual(configurations(self.out),
                         [("kestrel", "dense"), ("kestrel", "dense+bm25"), ("kestrel", "dense+bm25+graph")])

    def test_order_is_corpus_then_reference_runs_then_alphabetical(self):
        for corpus, name in [("own", "dense"), ("kestrel", "zz-extra"), ("kestrel", "bm25"), ("kestrel", "dense")]:
            write_run(self.out, corpus, name)
        cli_eval.rebuild_summary(self.out)
        self.assertEqual(configurations(self.out),
                         [("kestrel", "dense"), ("kestrel", "bm25"), ("kestrel", "zz-extra"), ("own", "dense")])

    def test_single_run_after_a_matrix_keeps_the_other_rows(self):
        write_run(self.out, "kestrel", "dense")
        write_run(self.out, "own", "dense")
        cli_eval.rebuild_summary(self.out)
        cli_eval.run_and_save("kestrel", ["bm25"], out_dir=self.out)  # one run, like `python -m app.evaluate run`
        self.assertEqual(configurations(self.out), [("kestrel", "dense"), ("kestrel", "bm25"), ("own", "dense")])

    def test_same_configuration_twice_gives_one_row_the_latest(self):
        write_run(self.out, "kestrel", "dense", run_at="2026-11-17 10:00:00")
        write_run(self.out, "kestrel", "dense", run_at="2026-11-17 22:00:00")
        cli_eval.rebuild_summary(self.out)
        rows = json.loads((self.out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["summary"]["run_at"], "2026-11-17 22:00:00")

    def test_summary_shows_run_at_chunking_and_backend(self):
        write_run(self.out, "kestrel", "dense", run_at="2026-11-17 22:00:00")
        cli_eval.rebuild_summary(self.out)
        md = (self.out / "summary.md").read_text(encoding="utf-8")
        self.assertIn("## Configurations in this folder", md)
        self.assertIn("| kestrel | dense | 2026-11-17 22:00:00 | 800/10 |", md)

    def test_real_run_records_run_at(self):
        before = time.strftime("%Y-%m-%d")
        _, result, _ = cli_eval.run_and_save("kestrel", ["bm25"], out_dir=self.out)
        self.assertTrue(result["summary"]["run_at"].startswith(before))

    def test_non_result_files_are_ignored(self):
        write_run(self.out, "kestrel", "dense")
        (self.out / "graph_triples_graded.json").write_text('{"meta": {}, "triples": []}', encoding="utf-8")
        cli_eval.rebuild_summary(self.out)
        cli_eval.rebuild_summary(self.out)  # summary.json itself must not be read back as a run
        self.assertEqual(configurations(self.out), [("kestrel", "dense")])

    def test_regression_pairs_appear_when_both_files_exist(self):
        write_run(self.out, "kestrel", "dense")
        cli_eval.rebuild_summary(self.out)
        self.assertNotIn("`dense` → `dense+bm25`", (self.out / "summary.md").read_text(encoding="utf-8"))
        write_run(self.out, "kestrel", "dense+bm25")
        cli_eval.rebuild_summary(self.out)
        self.assertIn("`dense` → `dense+bm25`", (self.out / "summary.md").read_text(encoding="utf-8"))

    def test_no_temporary_files_left(self):
        write_run(self.out, "kestrel", "dense")
        cli_eval.rebuild_summary(self.out)
        self.assertEqual(list(self.out.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
