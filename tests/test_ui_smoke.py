"""v3T16: browser smoke test of the Ask tab (Playwright).

Starts the real app on a free port and drives it in a headless browser:
switch retrievers on and off, ask a question, and check what the user sees
(answer, sources with retriever/rank badges, the graph path panel).

/api/ask and /api/config are answered by the test (page.route), so the test
needs no LLM key, no built graph and no embedding download, and always sees
the same answer. It tests the page, not retrieval quality.

Setup, once:  pip install -r requirements-dev.txt   then   python -m playwright install chromium
Run it alone: python -m pytest tests/test_ui_smoke.py -rs
It is a plain unittest module, so the project's command also runs it:
python -m unittest discover -s tests -t .
Without Playwright or its browser, the whole module is skipped with the reason.
"""
import json
import os
import socket
import subprocess
import sys
import time
import unittest
import urllib.request
from pathlib import Path

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    raise unittest.SkipTest(
        "UI test needs Playwright: pip install -r requirements-dev.txt, "
        "then python -m playwright install chromium"
    )

ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "retrievers": ["dense", "bm25", "graph"],
    "available": {"dense": True, "bm25": True, "graph": True},
    "default": ["dense", "bm25"],
    "graph_corpora": ["kestrel"],
}

RUNBOOK = {"id": "support-runbook-errors_md:8", "document": "support-runbook-errors.md",
           "text": "KST-5031 Scheduled report not delivered. Escalate to the team that owns postroom, as SEV-2.",
           "score": 0.05, "sources": {"dense": {"rank": 1}, "bm25": {"rank": 1}, "graph": {"rank": 1}}}
CATALOG = {"id": "service-catalog_md:2", "document": "service-catalog.md",
           "text": "### postroom Owner: **Messaging**.", "score": 0.03, "sources": {"graph": {"rank": 2}}}

GRAPH_PATH = {
    "entities": [{"id": 1, "name": "KST-5031", "type": "ErrorCode", "matched_text": "kst-5031"}],
    "edges": [
        {"source": "KST-5031", "relation": "raised_by", "target": "postroom", "hop": 1,
         "chunk_ids": ["support-runbook-errors_md:8"]},
        {"source": "postroom", "relation": "owned_by", "target": "Messaging", "hop": 2,
         "chunk_ids": ["service-catalog_md:2"]},
    ],
    "chunks": ["support-runbook-errors_md:8", "service-catalog_md:2"],
    "capped": [],
}


TRACE_ID = "3f2a9c"
T0 = "2026-11-18T09:00:00.000+00:00"


def span(name, ms, start_ms, attributes, children=(), status="ok", error=None):
    return {"id": name, "trace_id": TRACE_ID, "parent_id": None, "name": name,
            "started_at": f"2026-11-18T09:00:00.{start_ms:03d}+00:00", "duration_ms": ms, "status": status,
            "error": error, "attributes": attributes, "children": list(children)}


TRACE = {
    "trace": {"id": TRACE_ID, "kind": "question", "started_at": T0, "duration_ms": 900.0, "outcome": "answered",
              "fingerprint": "a1b2c3", "corpus": "kestrel", "attributes": {}},
    "spans": [span("request", 900.0, 0, {"question": "I'm [NAME_1]: KST-5031?", "retrievers": ["dense", "bm25"],
                                         "top_k": 5, "redactions": {"NAME": 1}}, [
        span("retrieve.dense", 40.0, 5, {"backend": "sentence-transformers",
                                         "results": [{"id": "support-runbook-errors_md:8", "rank": 1, "score": 0.61}]}),
        span("retrieve.bm25", 3.0, 46, {"index": "memory", "stopwords": True,
                                        "results": [{"id": "support-runbook-errors_md:8", "rank": 1, "score": 7.9}]}),
        span("fuse.rrf", 1.0, 50, {"rrf_k": 60, "results": [{"id": "support-runbook-errors_md:8", "rank": 1,
                                                            "score": 0.03, "sources": {"dense": 1, "bm25": 1}}]}),
        span("generate", 800.0, 52, {"gen_ai.request.model": "nvidia/nemotron", "gen_ai.usage.input_tokens": 1093,
                                     "gen_ai.usage.output_tokens": 120, "gen_ai.response.finish_reasons": ["stop"],
                                     "thinking": "off", "cost_usd": 0.0, "price_known": False,
                                     "redactions": {"EMAIL": 1, "NAME": 1}}),
        span("validate", 1.0, 853, {"valid": True, "cited_ids": ["support-runbook-errors_md:8"], "dropped_citations": 0}),
    ])],
}


INDEX_ID = "9b8a7c"
INDEX_TRACE = {
    "trace": {"id": INDEX_ID, "kind": "index", "started_at": "2026-11-18T09:00:00.000+00:00", "duration_ms": 30.0,
              "outcome": "ok", "fingerprint": "c0ffee", "corpus": "kestrel", "attributes": {}},
    "spans": [{
        "id": "r", "trace_id": INDEX_ID, "parent_id": None, "name": "index", "started_at": "2026-11-18T09:00:00.000+00:00",
        "duration_ms": 30.0, "status": "ok", "error": None, "attributes": {"trigger": "graph.build"},
        "children": [
            {"id": "a", "trace_id": INDEX_ID, "parent_id": "r", "name": "graph.extract", "started_at": "2026-11-18T09:00:00.001+00:00",
             "duration_ms": 10.0, "status": "ok", "error": None, "attributes": {"chunk_id": "runbook:1", "cached": False, "result": "ok", "entities": 4, "relations": 3, "attempts": 1}, "children": []},
            {"id": "b", "trace_id": INDEX_ID, "parent_id": "r", "name": "graph.extract", "started_at": "2026-11-18T09:00:00.011+00:00",
             "duration_ms": 10.0, "status": "ok", "error": None, "attributes": {"chunk_id": "billing:2", "cached": False, "result": "partial", "entities": 2, "relations": 0, "attempts": 2, "dropped_items": 1}, "children": []},
            {"id": "c", "trace_id": INDEX_ID, "parent_id": "r", "name": "graph.extract", "started_at": "2026-11-18T09:00:00.021+00:00",
             "duration_ms": 9.0, "status": "error", "error": "skipped: invalid output twice", "attributes": {"chunk_id": "catalog:4", "cached": False, "result": "skipped", "entities": 0, "relations": 0, "attempts": 2}, "children": []},
        ],
    }],
}


def ask_response(body):
    graph_on = "graph" in body.get("retrievers", [])
    return {
        "answer": "Contact Messaging in #oncall-messaging, as SEV-2.",
        "valid": True,
        "validation_error": None,
        "citations": [{**RUNBOOK, "citation": RUNBOOK["id"]}],
        "retrieved": [RUNBOOK, CATALOG] if graph_on else [RUNBOOK],
        "retrievers": body.get("retrievers", []),
        "backend": "dense=sentence-transformers, bm25=memory",
        "graph_path": GRAPH_PATH if graph_on else None,
        "timings": {"total_s": 0.01},
        "usage": {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 20},
        "trace_id": TRACE_ID,
    }


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AskTabSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        port = free_port()
        cls.url = f"http://127.0.0.1:{port}"
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(100):
            try:
                urllib.request.urlopen(cls.url + "/api/config", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            cls.tearDownClass()
            raise RuntimeError("the app did not start on " + cls.url)

        cls.pw = sync_playwright().start()
        try:
            # PLAYWRIGHT_BROWSER_PATH: optional, an installed Chrome/Chromium to use instead
            cls.browser = cls.pw.chromium.launch(executable_path=os.getenv("PLAYWRIGHT_BROWSER_PATH") or None)
        except Exception as exc:  # browser not downloaded
            cls.tearDownClass()
            raise unittest.SkipTest(
                f"no Playwright browser: run python -m playwright install chromium ({exc.__class__.__name__})"
            )

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "browser", None):
            cls.browser.close()
        if getattr(cls, "pw", None):
            cls.pw.stop()
        if getattr(cls, "server", None):
            cls.server.terminate()
            cls.server.wait(timeout=10)

    def setUp(self):
        self.page = self.browser.new_page()
        self.addCleanup(self.page.close)
        self.sent = []

        def handle_ask(route):
            body = json.loads(route.request.post_data or "{}")
            self.sent.append(body)
            route.fulfill(status=200, content_type="application/json", body=json.dumps(ask_response(body)))

        self.page.route("**/api/config", lambda r: r.fulfill(status=200, content_type="application/json",
                                                              body=json.dumps(CONFIG)))
        self.page.route("**/api/ask", handle_ask)
        self.page.route(f"**/api/traces/{TRACE_ID}", lambda r: r.fulfill(
            status=200, content_type="application/json", body=json.dumps(TRACE)))
        self.page.route(f"**/api/traces/{INDEX_ID}", lambda r: r.fulfill(
            status=200, content_type="application/json", body=json.dumps(INDEX_TRACE)))
        self.page.route("**/api/traces?*", lambda r: r.fulfill(
            status=200, content_type="application/json", body=json.dumps([
                {k: INDEX_TRACE["trace"][k] for k in ("id", "kind", "corpus", "outcome", "started_at", "duration_ms", "fingerprint")}])))
        self.page.goto(self.url)
        self.page.select_option("#askCorpus", "kestrel")

    def box(self, name):
        return self.page.locator(f'#askRet input[value="{name}"]')

    def ask(self, retrievers, question="Error KST-5031 happened twice. Which team should support contact?"):
        for name in ("dense", "bm25", "graph"):
            self.box(name).set_checked(name in retrievers)
        self.page.fill("#question", question)
        self.page.click("#askBtn")
        self.page.locator("#answer h3", has_text="Sources").wait_for()

    def test_graph_checkbox_only_for_kestrel(self):
        expect(self.box("graph")).to_be_enabled()
        self.page.select_option("#askCorpus", "own")
        expect(self.box("graph")).to_be_disabled()

    def test_switches_are_sent_to_the_api(self):
        self.ask(["dense", "bm25", "graph"])
        self.assertEqual(self.sent[-1]["retrievers"], ["dense", "bm25", "graph"])
        self.assertEqual(self.sent[-1]["corpus"], "kestrel")
        self.ask(["bm25"])
        self.assertEqual(self.sent[-1]["retrievers"], ["bm25"])

    def test_answer_sources_and_rank_badges(self):
        self.ask(["dense", "bm25"])
        answer = self.page.locator("#answer")
        expect(answer).to_contain_text("Contact Messaging in #oncall-messaging")
        expect(answer).to_contain_text("support-runbook-errors_md:8")
        # the cited source shows which retriever found it, at which rank (the same badges are
        # repeated, hidden, in the collapsed "All retrieved chunks" list)
        expect(answer.locator(".badge:visible", has_text="dense #1")).to_have_count(1)
        expect(answer.locator(".badge:visible", has_text="bm25 #1")).to_have_count(1)

    def test_graph_path_panel_when_graph_on(self):
        self.ask(["dense", "bm25", "graph"])
        panel = self.page.locator("#answer details", has_text="Graph path")
        expect(panel).to_contain_text("KST-5031")
        expect(panel).to_contain_text("ErrorCode")
        expect(panel).to_contain_text("Hop 2")
        expect(panel).to_contain_text("postroom —[owned_by]→ Messaging")
        expect(panel).to_contain_text("in the final list at rank 1")

    def test_no_graph_panel_when_graph_off(self):
        self.ask(["dense", "bm25"])
        expect(self.page.locator("#answer")).not_to_contain_text("Graph path")

    def test_details_opens_this_answers_trace(self):
        self.ask(["dense", "bm25"])
        self.page.click("#detailsBtn")
        view = self.page.locator("#traceView")
        expect(view).to_contain_text("What happened")
        expect(view).to_contain_text("answered from the documents")
        for step in ("request", "retrieve.dense", "retrieve.bm25", "fuse.rrf", "generate", "validate"):
            expect(view).to_contain_text(step)
        expect(view).to_contain_text("1093 in + 120 out")
        expect(view).to_contain_text("free tier")
        expect(view).to_contain_text("personal data removed before the call")
        self.page.click("#detailsBtn")  # closes again
        expect(view).to_be_empty()

    def test_index_trace_from_recent_traces_with_filters(self):
        self.page.click("#recentTracesBtn")
        self.page.click(f'[data-open-trace="{INDEX_ID}"]')
        view = self.page.locator("#recentTraceView")
        expect(view).to_contain_text("3 graph.extract")
        rows = view.locator(".span-row")
        expect(rows).to_have_count(4)  # the index root + 3 graph.extract spans
        view.locator('[data-filter="error"]').check()
        expect(view.locator(".span-row:visible")).to_have_count(1)
        expect(view.locator(".span-row:visible")).to_contain_text("skipped: invalid output twice")
        view.locator('[data-filter="error"]').uncheck()
        view.locator('[data-filter="zero"]').check()
        zero = view.locator(".span-row:visible")
        expect(zero).to_have_count(2)  # a partial chunk with 0 relations, and the skipped one
        expect(zero.first).to_contain_text("billing:2")
        expect(zero.last).to_contain_text("catalog:4")


if __name__ == "__main__":
    unittest.main()
