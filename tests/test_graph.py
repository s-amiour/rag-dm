"""Concept-graph extraction (v3T01) with a fake LLM: no key, no network."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from app import core, graph
from app.graph_schema import ENTITY_TYPES, RELATIONS, relation_allowed

RUNBOOK = (
    "### KST-5031 — Scheduled report not delivered\n\n"
    "**Cause:** This error is raised by postroom, the service that sends all Kestrel e-mails.\n"
    "**Escalate:** escalate to the team that owns postroom, as SEV-2."
)
CATALOG = (
    "### postroom\n\nSends every e-mail that leaves Kestrel.\nOwner: **Messaging**.\n\n"
    "postroom replaced the older mailman service in November 2025."
)
TEAMS = (
    "| Team | Engineering lead | On-call channel |\n|---|---|---|\n"
    "| Messaging | Inès Moreau | #oncall-messaging |"
)


def rec(doc, idx, text):
    return {"id": f"{core.safe_name(doc)}:{idx}", "document": doc, "index": idx, "text": text}


RECORDS = [rec("support-runbook-errors.md", 0, RUNBOOK), rec("service-catalog.md", 0, CATALOG),
           rec("service-catalog.md", 1, TEAMS)]


def reply(entities, relations):
    return json.dumps({
        "entities": [{"name": n, "type": t} for n, t in entities],
        "relations": [{"source": s, "relation": r, "target": o, "quote": q} for s, r, o, q in relations],
    })


GOOD = {
    "support-runbook-errors_md:0": reply(
        [("KST-5031", "ErrorCode"), ("postroom", "Service")],
        [("KST-5031", "raised_by", "postroom", "This error is raised by postroom"),
         ("KST-5031", "escalates_to", "postroom", "escalate to the team that owns postroom")]),
    "service-catalog_md:0": reply(
        [("postroom", "Service"), ("Messaging", "Team"), ("mailman", "Service")],
        [("postroom", "owned_by", "Messaging", "Owner: **Messaging**."),
         ("postroom", "replaces", "mailman", "postroom replaced the older mailman service")]),
    "service-catalog_md:1": reply([("Messaging", "Team")], []),
}


class FakeLLM:
    """Answers from a script: {chunk_id: [reply for call 1, reply for call 2, ...]}.
    A reply can be an exception instance, which is raised instead."""
    model = "fake-model"

    def __init__(self, script=None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []

    def complete(self, messages):
        chunk_id = messages[1]["content"].split("\n", 1)[0].removeprefix("Chunk id: ")
        self.calls.append((chunk_id, messages))
        queue = self.script.get(chunk_id)
        out = queue.pop(0) if queue else GOOD[chunk_id]
        if isinstance(out, Exception):
            raise out
        return graph.LLMReply(out, prompt_tokens=100, completion_tokens=20)


class GraphTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.patches = [
            mock.patch.object(core, "INDEX_DIR", tmp / "index"),
            mock.patch.object(graph, "RESULTS_DIR", tmp / "results"),
            mock.patch.object(graph, "_sleep", lambda s: None),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def run_build(self, llm, records=RECORDS, **kw):
        return graph.build("kestrel", records=records, llm=llm, delay_s=0, log=lambda *_: None, **kw)

    def query(self, sql, *params):
        con = graph.connect("kestrel")
        try:
            return [tuple(r) for r in con.execute(sql, params)]
        finally:
            con.close()


class SchemaTests(unittest.TestCase):
    def test_fixed_lists_and_added_relations(self):
        self.assertEqual(set(ENTITY_TYPES), {"Product", "Feature", "Plan", "Service", "ErrorCode",
                                             "Policy", "Procedure", "Team"})
        self.assertIn("raised_by", RELATIONS)
        self.assertIn("escalates_to", RELATIONS)

    def test_relation_signatures(self):
        self.assertTrue(relation_allowed("owned_by", "Service", "Team"))
        self.assertFalse(relation_allowed("owned_by", "Team", "Service"))
        self.assertTrue(relation_allowed("raised_by", "ErrorCode", "Service"))
        self.assertTrue(relation_allowed("escalates_to", "ErrorCode", "Team"))
        self.assertTrue(relation_allowed("replaces", "Service", "Service"))
        self.assertFalse(relation_allowed("replaces", "Service", "Policy"))  # same type only
        self.assertFalse(relation_allowed("invented", "Service", "Team"))


class PromptTests(unittest.TestCase):
    def test_prompt_example_passes_our_own_validation(self):
        ex = graph.parse_reply(json.dumps(graph.EXAMPLE_REPLY), graph.EXAMPLE_CHUNK)
        self.assertEqual(len(ex.relations), 3)

    def test_prompt_does_not_suggest_the_product_name(self):
        self.assertNotIn("Kestrel Analytics", graph.SYSTEM_PROMPT)


class ValidationTests(unittest.TestCase):
    def test_good_reply_and_json_fence(self):
        ex = graph.parse_reply("```json\n" + GOOD["service-catalog_md:0"] + "\n```", CATALOG)
        self.assertEqual(len(ex.relations), 2)

    def assertRejected(self, raw, chunk, fragment):
        with self.assertRaises(graph.ValidationError) as ctx:
            graph.parse_reply(raw, chunk)
        self.assertIn(fragment, graph.format_errors(ctx.exception))

    def test_bad_json(self):
        self.assertRejected("Sure! Here are the triples:", CATALOG, "Invalid JSON")

    def test_unknown_entity_type(self):
        self.assertRejected(reply([("Messaging", "Department")], []), CATALOG, "unknown entity type")

    def test_unknown_relation(self):
        self.assertRejected(reply([("postroom", "Service"), ("Messaging", "Team")],
                                  [("postroom", "managed_by", "Messaging", "Owner: Messaging")]),
                            CATALOG, "unknown relation")

    def test_quote_not_in_chunk(self):
        self.assertRejected(reply([("postroom", "Service"), ("Messaging", "Team")],
                                  [("postroom", "owned_by", "Messaging", "postroom belongs to Messaging")]),
                            CATALOG, "not copied verbatim")

    def test_endpoint_not_in_entities(self):
        self.assertRejected(reply([("postroom", "Service")],
                                  [("postroom", "owned_by", "Messaging", "Owner: Messaging")]),
                            CATALOG, "not in the entities list")

    def test_types_not_allowed_for_relation(self):
        self.assertRejected(reply([("postroom", "Service"), ("Messaging", "Team")],
                                  [("Messaging", "owned_by", "postroom", "Owner: Messaging")]),
                            CATALOG, "cannot go from Team to Service")

    def test_quote_must_name_an_end_of_the_relation(self):
        plans = ("| | Free | Business | Enterprise |\n|---|---|---|---|\n"
                 "| SCIM user provisioning | — | — | Yes |\n| Single sign-on (SAML) | — | Yes | Yes |")
        ents = [("Business", "Plan"), ("SCIM user provisioning", "Feature")]
        # the false edge of the real build: "Yes" is in the chunk, but it comes from another cell
        self.assertRejected(reply(ents, [("Business", "has_feature", "SCIM user provisioning", "Yes")]),
                            plans, "names neither")
        # a quote that is the table row naming the feature is accepted
        ok = reply(ents, [("Business", "has_feature", "SCIM user provisioning",
                           "| SCIM user provisioning | — | — | Yes |")])
        self.assertEqual(len(graph.parse_reply(ok, plans).relations), 1)

    def test_relation_ends_are_found_by_name_before_alias(self):
        raw = json.dumps({
            "entities": [{"name": "postroom", "type": "Service", "aliases": ["mailman"]},
                         {"name": "mailman", "type": "Service", "aliases": []}],
            "relations": [{"source": "postroom", "relation": "replaces", "target": "mailman",
                           "quote": "postroom replaced the older mailman service"}]})
        ex = graph.parse_reply(raw, CATALOG)
        src, dst = graph.resolve_endpoints(ex.relations[0], graph._by_key(ex.entities))
        self.assertEqual((src.name, dst.name), ("postroom", "mailman"))   # was postroom -> postroom

    def test_entity_not_mentioned_in_chunk(self):
        self.assertRejected(reply([("ledger", "Service")], []), CATALOG, "appears in the chunk")


class BuildTests(GraphTestCase):
    def test_full_build_summary_and_integrity(self):
        s = self.run_build(FakeLLM())
        self.assertEqual((s["chunks"], s["extracted"], s["skipped"], s["errors"]), (3, 3, 0, 0))
        self.assertEqual(s["run"]["llm_calls"], 3)
        self.assertEqual(s["run"]["prompt_tokens"], 300)
        self.assertEqual(s["entities"], 4)        # KST-5031, postroom, Messaging, mailman
        self.assertEqual(s["relations"], 4)
        self.assertTrue(s["integrity"]["ok"], s["integrity"])
        self.assertEqual(s["state"], "complete")
        text = graph.format_summary(s)
        for word in ("chunks processed", "entities", "relations", "LLM calls", "tokens", "wall time"):
            self.assertIn(word, text)

    def test_node_in_several_chunks_keeps_all_evidence(self):
        self.run_build(FakeLLM())
        chunks = self.query(
            "SELECT e.chunk_id FROM evidence e JOIN nodes n ON e.kind='node' AND e.owner_id=n.id "
            "WHERE n.name='postroom' ORDER BY e.chunk_id")
        self.assertEqual([c[0] for c in chunks], ["service-catalog_md:0", "support-runbook-errors_md:0"])
        quotes = self.query("SELECT doc, chunk_id, quote FROM evidence WHERE kind='edge'")
        self.assertIn(("service-catalog.md", "service-catalog_md:0", "Owner: **Messaging**."), quotes)

    def test_bad_json_is_retried_once_then_accepted(self):
        llm = FakeLLM({"service-catalog_md:0": ["not json at all"]})
        s = self.run_build(llm)
        self.assertEqual((s["extracted"], s["retried"], s["skipped"]), (3, 1, 0))
        retry_prompt = llm.calls[2][1][-1]["content"]           # 2nd call on that chunk
        self.assertIn("Invalid JSON", retry_prompt)
        outcomes = self.query("SELECT outcome FROM llm_calls WHERE chunk_id='service-catalog_md:0' ORDER BY id")
        self.assertEqual([o[0] for o in outcomes], ["invalid", "retried"])

    def test_invalid_twice_is_skipped_logged_and_build_finishes(self):
        bad_cases = {
            "bad JSON": "{entities: [",
            "unknown type": reply([("Messaging", "Department")], []),
            "invented quote": reply([("postroom", "Service"), ("Messaging", "Team")],
                                    [("postroom", "owned_by", "Messaging", "Messaging runs postroom")]),
        }
        for label, bad in bad_cases.items():
            with self.subTest(label):
                graph.db_path("kestrel").unlink(missing_ok=True)
                llm = FakeLLM({"service-catalog_md:0": [bad, bad]})
                s = self.run_build(llm, strict=True)                    # the ticket's rule
                self.assertEqual(len([c for c in llm.calls if c[0] == "service-catalog_md:0"]), 2)
                self.assertEqual((s["extracted"], s["skipped"]), (2, 1))
                self.assertTrue(s["integrity"]["ok"])
                errors = self.query("SELECT chunk_id, kind, raw_reply FROM graph_errors")
                self.assertEqual(errors, [("service-catalog_md:0", "skipped", bad)])
                exported = (graph.RESULTS_DIR / "graph_errors.jsonl").read_text(encoding="utf-8")
                self.assertIn("service-catalog_md:0", exported)

    def test_default_mode_keeps_the_valid_part_and_logs_what_was_dropped(self):
        invented = reply([("postroom", "Service"), ("Messaging", "Team"), ("Kestrel Analytics", "Product")],
                         [("postroom", "owned_by", "Messaging", "Owner: **Messaging**."),
                          ("Messaging", "owned_by", "postroom", "Owner: **Messaging**."),          # backwards
                          ("postroom", "replaces", "mailman", "postroom replaced the older mailman service")])  # no entity
        s = self.run_build(FakeLLM({"service-catalog_md:0": [invented, invented]}))
        self.assertEqual((s["extracted"], s["partial"], s["skipped"]), (3, 1, 0))
        self.assertTrue(s["integrity"]["ok"])
        names = {r[0] for r in self.query("SELECT name FROM nodes")}
        self.assertNotIn("Kestrel Analytics", names)                     # not written in the chunk
        edges = self.query("SELECT n1.name, e.relation, n2.name FROM edges e JOIN nodes n1 ON n1.id=e.source_id "
                           "JOIN nodes n2 ON n2.id=e.target_id WHERE e.relation='owned_by'")
        self.assertEqual(edges, [("postroom", "owned_by", "Messaging")])  # the backwards copy was dropped
        kind, error = self.query("SELECT kind, error FROM graph_errors")[0]
        self.assertEqual(kind, "partial")
        self.assertIn("Kestrel Analytics", error)
        self.assertIn("cannot go from Team to Service", error)
        self.assertIn("'mailman' is not in the entities list", error)
        outcomes = self.query("SELECT outcome FROM llm_calls WHERE chunk_id='service-catalog_md:0' ORDER BY id")
        self.assertEqual([o[0] for o in outcomes], ["invalid", "partial"])

    def test_nothing_valid_left_is_still_skipped(self):
        for bad in ("{entities: [", reply([("Messaging", "Department")], [])):
            with self.subTest(bad=bad):
                graph.db_path("kestrel").unlink(missing_ok=True)
                s = self.run_build(FakeLLM({"service-catalog_md:0": [bad, bad]}))
                self.assertEqual((s["partial"], s["skipped"]), (0, 1))

    def test_chunks_skipped_by_a_strict_build_are_recovered_without_any_call(self):
        invented = reply([("postroom", "Service"), ("Messaging", "Team")],
                         [("postroom", "owned_by", "Messaging", "Messaging runs postroom")])
        self.run_build(FakeLLM({"service-catalog_md:0": [invented, invented]}), strict=True)

        class NoCalls(FakeLLM):
            def complete(self, messages):
                raise AssertionError("the LLM must not be called")

        s = self.run_build(NoCalls())
        self.assertEqual((s["run"]["llm_calls"], s["salvaged_from_cache"], s["partial"], s["skipped"]), (0, 1, 1, 0))
        self.assertTrue(s["integrity"]["ok"])
        s = self.run_build(NoCalls(), strict=True)                       # strict ignores kept parts
        self.assertEqual((s["partial"], s["skipped"]), (0, 1))

    def test_cached_extractions_are_rechecked_with_current_rules_without_calls(self):
        self.run_build(FakeLLM())
        # simulate an extraction cached under older, looser rules: a one-word quote
        old = json.loads(GOOD["service-catalog_md:0"])
        old["relations"].append({"source": "mailman", "relation": "replaces", "target": "postroom", "quote": "older"})
        con = graph.connect("kestrel")
        con.execute("UPDATE extraction_cache SET extraction=? WHERE chunk_hash=?",
                    (json.dumps({"entities": [{**e, "aliases": []} for e in old["entities"]],
                                 "relations": old["relations"]}), graph.chunk_hash(CATALOG)))
        con.commit(); con.close()

        class NoCalls(FakeLLM):
            def complete(self, messages):
                raise AssertionError("the LLM must not be called")

        s = self.run_build(NoCalls())
        self.assertEqual((s["run"]["llm_calls"], s["rechecked"]["chunks"], s["rechecked"]["items_dropped"]), (0, 1, 1))
        self.assertEqual((s["partial"], s["relations"]), (1, 4))          # the loose edge is gone, the rest kept
        self.assertIn("names neither", s["rechecked"]["details"]["service-catalog_md:0"][0])
        self.assertIn("re-checked", graph.format_summary(s))
        s = self.run_build(NoCalls(), strict=True)
        self.assertEqual(s["skipped"], 1)

    def test_validation_version_is_part_of_the_graph_fingerprint(self):
        before = graph.graph_fingerprint(RECORDS)
        with mock.patch.object(graph, "VALIDATION_VERSION", "999"):
            self.assertNotEqual(graph.graph_fingerprint(RECORDS), before)
        self.assertEqual(graph.chunk_hash("x"), graph.chunk_hash("x"))   # cache keys do not depend on it

    def test_salvaged_part_always_passes_full_validation(self):
        raw = reply([("postroom", "Service"), ("Ghost", "Team")],
                    [("postroom", "owned_by", "Ghost", "Owner: **Messaging**.")])
        kept, dropped = graph.salvage(raw, CATALOG)
        self.assertEqual([e.name for e in kept.entities], ["postroom"])
        self.assertEqual(kept.relations, [])
        self.assertEqual(len(dropped), 2)
        graph.parse_reply(kept.model_dump_json(), CATALOG)               # raises if anything invalid slipped in

    def test_rerun_with_unchanged_documents_makes_no_call(self):
        self.run_build(FakeLLM())
        first = self.query("SELECT COUNT(*) FROM llm_calls")[0][0]

        class NoCalls(FakeLLM):
            def complete(self, messages):
                raise AssertionError("the LLM must not be called")

        s = self.run_build(NoCalls())
        self.assertEqual((s["run"]["llm_calls"], s["cached"]), (0, 3))
        self.assertEqual(self.query("SELECT COUNT(*) FROM llm_calls")[0][0], first)
        self.assertEqual(s["graph_total"]["llm_calls"], 3)       # cost of the graph still reported
        self.assertEqual(s["relations"], 4)

    def test_skipped_chunk_is_not_paid_again_unless_asked(self):
        bad = "nope"
        self.run_build(FakeLLM({"service-catalog_md:0": [bad, bad]}))
        s = self.run_build(FakeLLM())
        self.assertEqual((s["run"]["llm_calls"], s["skipped"]), (0, 1))
        s = self.run_build(FakeLLM(), retry_skipped=True)
        self.assertEqual((s["run"]["llm_calls"], s["skipped"], s["relations"]), (1, 0, 4))

    def test_changed_document_marks_graph_stale_and_reextracts_only_that_chunk(self):
        self.run_build(FakeLLM())
        self.assertEqual(graph.graph_status(records=RECORDS)["state"], "fresh")

        changed = [*RECORDS[:2], rec("service-catalog.md", 1, TEAMS + "\n| Messaging | x | y |")]
        self.assertEqual(graph.graph_status(records=changed)["state"], "stale")

        GOOD_CHANGED = {"service-catalog_md:1": [reply([("Messaging", "Team")], [])]}
        llm = FakeLLM(GOOD_CHANGED)
        s = self.run_build(llm, records=changed)
        self.assertEqual([c[0] for c in llm.calls], ["service-catalog_md:1"])
        self.assertEqual(graph.graph_status(records=changed)["state"], "fresh")
        self.assertTrue(s["integrity"]["ok"])

    def test_limit_means_the_first_n_chunks_so_a_rerun_is_free(self):
        llm = FakeLLM()
        s = self.run_build(llm, limit=2)
        self.assertEqual([c[0] for c in llm.calls], ["support-runbook-errors_md:0", "service-catalog_md:0"])
        self.assertEqual((s["extracted"], s["not_attempted"]), (2, 1))
        llm = FakeLLM()
        s = self.run_build(llm, limit=2)                      # same command again: nothing new to pay for
        self.assertEqual((len(llm.calls), s["cached"], s["not_attempted"]), (0, 2, 1))
        s = self.run_build(FakeLLM())                         # the full build pays only for the rest
        self.assertEqual((s["run"]["llm_calls"], s["state"]), (1, "complete"))

    def test_rate_limit_backs_off_then_succeeds(self):
        waits = []
        with mock.patch.object(graph, "_sleep", waits.append):
            llm = FakeLLM({"service-catalog_md:0": [
                graph.TransientLLMError("429", retry_after=3, kind="rate_limited")]})
            s = self.run_build(llm)
        self.assertEqual((s["extracted"], s["errors"]), (3, 0))
        self.assertTrue(any(w >= 3 for w in waits))
        outcomes = self.query("SELECT outcome FROM llm_calls WHERE chunk_id='service-catalog_md:0' ORDER BY id")
        self.assertEqual([o[0] for o in outcomes], ["rate_limited", "ok"])

    def test_rate_limit_that_never_ends_is_an_error_not_a_crash(self):
        always = [graph.TransientLLMError("429", kind="rate_limited")] * (graph.MAX_TRANSIENT_RETRIES + 1)
        s = self.run_build(FakeLLM({"service-catalog_md:0": always}))
        self.assertEqual((s["extracted"], s["errors"]), (2, 1))
        self.assertTrue(s["state"].startswith("partial"))
        self.assertEqual(self.query("SELECT kind FROM graph_errors"), [("error",)])
        # not cached: the next run tries again
        s = self.run_build(FakeLLM())
        self.assertEqual((s["run"]["llm_calls"], s["state"]), (1, "complete"))

    def test_unexpected_client_exception_fails_the_chunk_not_the_build(self):
        s = self.run_build(FakeLLM({"service-catalog_md:0": [KeyError("choices")]}))
        self.assertEqual((s["extracted"], s["errors"]), (2, 1))
        self.assertTrue(s["integrity"]["ok"])
        self.assertIn("KeyError", self.query("SELECT error FROM graph_errors")[0][0])

    def test_failed_request_is_still_logged_as_a_call(self):
        s = self.run_build(FakeLLM({"service-catalog_md:0": [KeyError("choices")]}))
        rows = self.query("SELECT outcome, prompt_tokens FROM llm_calls WHERE chunk_id='service-catalog_md:0'")
        self.assertEqual(rows, [("error", 0)])
        self.assertEqual(s["run"]["llm_calls"], 3)

    def test_fatal_error_call_is_logged(self):
        self.run_build(FakeLLM({"support-runbook-errors_md:0": [graph.FatalLLMError("410 Gone")]}))
        self.assertEqual(self.query("SELECT outcome FROM llm_calls"), [("error",)])

    def test_wrong_key_stops_calling_but_still_builds_and_reports(self):
        llm = FakeLLM({"service-catalog_md:0": [graph.FatalLLMError("AuthenticationError: 401")]})
        s = self.run_build(llm)
        self.assertEqual(len(llm.calls), 2)                    # chunk 3 never sent
        self.assertEqual((s["extracted"], s["errors"], s["not_attempted"]), (1, 1, 1))
        self.assertIn("401", s["aborted"])
        self.assertIn("STOPPED EARLY", graph.format_summary(s))
        s = self.run_build(FakeLLM())                          # key fixed: finishes the rest only
        self.assertEqual((s["run"]["llm_calls"], s["state"]), (2, "complete"))

    def test_integrity_detects_unknown_chunk_ids(self):
        self.run_build(FakeLLM())
        con = graph.connect("kestrel")
        try:
            result = graph.check_integrity(con, RECORDS[:2])          # chunking without chunk 1
        finally:
            con.close()
        self.assertFalse(result["ok"])
        self.assertEqual(result["unknown_chunk_ids"], ["service-catalog_md:1"])

    def test_only_kestrel(self):
        with self.assertRaises(ValueError):
            graph.build("own", records=RECORDS, llm=FakeLLM(), delay_s=0)


class ClientOptionsTests(unittest.TestCase):
    """OpenAICompatibleLLM with a fake openai client: what is sent, what comes back."""

    def make(self, env, reply_text="{}", finish="stop"):
        sent = {}

        class Completions:
            def create(self, **kw):
                sent.update(kw)
                msg = mock.Mock(content=reply_text)
                return mock.Mock(choices=[mock.Mock(message=msg, finish_reason=finish)],
                                 usage=mock.Mock(prompt_tokens=10, completion_tokens=5))

        fake_client = mock.Mock()
        fake_client.chat.completions = Completions()
        base = {"LLM_API_KEY": "k", "LLM_BASE_URL": "http://x", "LLM_MODEL": "answer-model"}
        with mock.patch.dict("os.environ", {**base, **env}, clear=True), \
                mock.patch("dotenv.load_dotenv"), mock.patch("openai.OpenAI", return_value=fake_client):
            llm = graph.OpenAICompatibleLLM()
        return llm, sent

    def test_thinking_switch_is_sent_only_when_asked(self):
        llm, sent = self.make({"GRAPH_LLM_MODEL": "nvidia/nemotron", "GRAPH_DISABLE_THINKING": "1"})
        llm.complete([{"role": "user", "content": "x"}])
        self.assertEqual(sent["model"], "nvidia/nemotron")
        self.assertEqual(sent["temperature"], 0)
        self.assertFalse(sent["extra_body"]["chat_template_kwargs"]["enable_thinking"])
        self.assertEqual(llm.options, {"model": "nvidia/nemotron", "thinking": "off"})

        llm, sent = self.make({})
        llm.complete([{"role": "user", "content": "x"}])
        self.assertEqual(sent["model"], "answer-model")
        self.assertIsNone(sent["extra_body"])

    def test_empty_reply_cut_by_token_limit_is_explained(self):
        llm, _ = self.make({}, reply_text=None, finish="length")
        text = llm.complete([{"role": "user", "content": "x"}]).text
        self.assertIn("GRAPH_DISABLE_THINKING", text)


class RetrieverStillUnavailableTests(unittest.TestCase):
    def test_graph_selection_still_unavailable_until_t3(self):
        from app import retrieval
        self.assertFalse(retrieval.graph_available())
        with self.assertRaises(retrieval.RetrieverUnavailable):
            retrieval._run_graph("kestrel", "q", 5, 800, 10)


if __name__ == "__main__":
    unittest.main()
