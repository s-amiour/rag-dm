"""v3T16: browser smoke test of the Ask tab (Playwright).

Starts the real app on a free port and drives it in a headless browser:
switch retrievers on and off, ask a question, and check what the user sees
(answer, sources with retriever/rank badges, the graph path panel).

/api/ask and /api/config are answered by the test (page.route), so the test
needs no LLM key, no built graph and no embedding download, and always sees
the same answer. It tests the page, not retrieval quality.

Setup, once:   pip install playwright   then   python -m playwright install chromium
Without Playwright or its browser the test is skipped, with the reason (pytest -rs).
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="Playwright not installed: pip install playwright")

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
    }


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/api/config", timeout=1)
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("the app did not start on " + url)
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            # PLAYWRIGHT_BROWSER_PATH: optional, an installed Chrome/Chromium to use instead
            b = p.chromium.launch(executable_path=os.getenv("PLAYWRIGHT_BROWSER_PATH") or None)
        except Exception as exc:  # browser not downloaded
            pytest.skip(f"no Playwright browser: run `python -m playwright install chromium` ({exc.__class__.__name__})")
        yield b
        b.close()


@pytest.fixture
def page(server, browser):
    page = browser.new_page()
    sent = []

    def handle_ask(route):
        body = json.loads(route.request.post_data or "{}")
        sent.append(body)
        route.fulfill(status=200, content_type="application/json", body=json.dumps(ask_response(body)))

    page.route("**/api/config", lambda r: r.fulfill(status=200, content_type="application/json",
                                                     body=json.dumps(CONFIG)))
    page.route("**/api/ask", handle_ask)
    page.goto(server)
    page.select_option("#askCorpus", "kestrel")
    page.sent = sent
    yield page
    page.close()


def box(page, name):
    return page.locator(f'#askRet input[value="{name}"]')


def ask(page, retrievers, question="Error KST-5031 happened twice. Which team should support contact?"):
    for name in ("dense", "bm25", "graph"):
        box(page, name).set_checked(name in retrievers)
    page.fill("#question", question)
    page.click("#askBtn")
    page.locator("#answer h3", has_text="Sources").wait_for()


def test_graph_checkbox_only_for_kestrel(page):
    sync_api.expect(box(page, "graph")).to_be_enabled()
    page.select_option("#askCorpus", "own")
    sync_api.expect(box(page, "graph")).to_be_disabled()


def test_switches_are_sent_to_the_api(page):
    ask(page, ["dense", "bm25", "graph"])
    assert page.sent[-1]["retrievers"] == ["dense", "bm25", "graph"]
    assert page.sent[-1]["corpus"] == "kestrel"
    ask(page, ["bm25"])
    assert page.sent[-1]["retrievers"] == ["bm25"]


def test_answer_sources_and_rank_badges(page):
    ask(page, ["dense", "bm25"])
    answer = page.locator("#answer")
    sync_api.expect(answer).to_contain_text("Contact Messaging in #oncall-messaging")
    sync_api.expect(answer).to_contain_text("support-runbook-errors_md:8")
    # the cited source shows which retriever found it, at which rank (the same badges are
    # repeated, hidden, in the collapsed "All retrieved chunks" list)
    sync_api.expect(answer.locator(".badge:visible", has_text="dense #1")).to_have_count(1)
    sync_api.expect(answer.locator(".badge:visible", has_text="bm25 #1")).to_have_count(1)


def test_graph_path_panel_when_graph_on(page):
    ask(page, ["dense", "bm25", "graph"])
    panel = page.locator("#answer details", has_text="Graph path")
    sync_api.expect(panel).to_contain_text("KST-5031")
    sync_api.expect(panel).to_contain_text("ErrorCode")
    sync_api.expect(panel).to_contain_text("Hop 2")
    sync_api.expect(panel).to_contain_text("postroom —[owned_by]→ Messaging")
    sync_api.expect(panel).to_contain_text("in the final list at rank 1")


def test_no_graph_panel_when_graph_off(page):
    ask(page, ["dense", "bm25"])
    sync_api.expect(page.locator("#answer")).not_to_contain_text("Graph path")