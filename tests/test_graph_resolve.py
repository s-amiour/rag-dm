"""Entity resolution (v3T02)."""
import json
import unittest

from app import graph, graph_resolve as gr
from tests.test_graph import FakeLLM, GraphTestCase, rec

RUNBOOK = ("### KST-5031 - Scheduled report not delivered\n\n"
           "**Cause:** This error is raised by postroom, the service that sends all Kestrel e-mails.\n"
           "**Escalate:** escalate to the team that owns postroom, as SEV-2.")
CATALOG = ("### postroom\n\nSends every e-mail.\nOwner: **Messaging**.\n\n"
           "postroom replaced the older mailman service; old alerts that mention mailman refer to postroom today.")
TEAMS = "| Team | On-call channel |\n|---|---|\n| the Messaging team | #oncall-messaging |"
RECORDS = [rec("support-runbook-errors.md", 8, RUNBOOK), rec("service-catalog.md", 2, CATALOG),
           rec("service-catalog.md", 3, TEAMS)]


def answer(entities, relations):
    return json.dumps({"entities": [{"name": n, "type": t, "aliases": a} for n, t, a in entities],
                       "relations": [{"source": s, "relation": r, "target": o, "quote": q} for s, r, o, q in relations]})


SCRIPT = {
    "support-runbook-errors_md:8": [answer(
        [("KST-5031", "ErrorCode", []), ("postroom", "Service", [])],
        [("KST-5031", "raised_by", "postroom", "This error is raised by postroom")])],
    "service-catalog_md:2": [answer(
        # the trap: mailman listed as an alias of postroom AND a replaces edge between them
        [("postroom", "Service", ["mailman"]), ("Messaging", "Team", []), ("mailman", "Service", [])],
        [("postroom", "owned_by", "Messaging", "Owner: **Messaging**."),
         ("postroom", "replaces", "mailman", "postroom replaced the older mailman service")])],
    "service-catalog_md:3": [answer([("the Messaging team", "Team", [])], [])],
}


class NormaliseTests(unittest.TestCase):
    def test_articles_suffixes_brackets(self):
        self.assertEqual(gr.normalise_name("the Messaging team"), "messaging")
        self.assertEqual(gr.normalise_name("Messaging (owner of postroom)"), "messaging")
        self.assertEqual(gr.normalise_name("Messaging"), "messaging")
        self.assertEqual(gr.normalise_name("postroom service"), "postroom")

    def test_accents_and_case(self):
        self.assertEqual(gr.normalise_name("Inès Moreau"), gr.normalise_name("Ines Moreau"))

    def test_punctuation_but_not_inner_hyphens(self):
        self.assertEqual(gr.normalise_name("connector-hub"), "connector-hub")
        self.assertEqual(gr.normalise_name("#oncall-payments"), "oncall-payments")
        self.assertEqual(gr.normalise_name("Single sign-on (SAML)"), "single sign-on")

    def test_never_strips_the_last_word(self):
        # the plan called "Team" must not become "" and merge with anything else that became ""
        self.assertEqual(gr.normalise_name("Team"), "team")
        self.assertEqual(gr.normalise_name("The service"), "service")


class IdentifierTests(unittest.TestCase):
    def test_error_codes_exact(self):
        self.assertEqual(gr.identifiers("KST-5031", "ErrorCode"), {"KST-5031"})
        self.assertEqual(gr.identifiers("KST-5031 - Scheduled report not delivered", "ErrorCode"), {"KST-5031"})
        self.assertNotEqual(gr.identifiers("KST-5031", "ErrorCode"), gr.identifiers("KST-5038", "ErrorCode"))

    def test_ip_addresses(self):
        self.assertEqual(gr.identifiers("203.0.113.10", "Service"), {"203.0.113.10"})

    def test_plan_names(self):
        self.assertEqual(gr.identifiers("Enterprise plans", "Plan"), {"plan:enterprise"})
        self.assertEqual(gr.identifiers("Free plan", "Plan"), gr.identifiers("Free", "Plan"))
        self.assertEqual(gr.identifiers("Team", "Plan"), {"plan:team"})
        self.assertEqual(gr.identifiers("Annual plans", "Plan"), frozenset())
        self.assertEqual(gr.identifiers("Team", "Team"), frozenset())   # a team called Team is not a plan

    def test_plain_names_have_none(self):
        self.assertEqual(gr.identifiers("postroom", "Service"), frozenset())


def surfaces(*items):
    """items: (type, name, chunk). Builds T1 nodes the way graph.py does (same type + same key)."""
    out = {}
    for typ, name, chunk in items:
        key = gr._key(name)
        s = out.setdefault((typ, key), gr.Surface(typ, key))
        s.names[name] = s.names.get(name, 0) + 1
        s.chunks.add(chunk)
    return out


def same(res, a, b):
    return res.root[(a[0], gr._key(a[1]))] == res.root[(b[0], gr._key(b[1]))]


class ResolveTests(unittest.TestCase):
    def test_messaging_team_and_messaging_merge(self):
        s = surfaces(("Team", "Messaging", "catalog:2"), ("Team", "the Messaging team", "catalog:3"))
        res = gr.resolve(s)
        self.assertTrue(same(res, ("Team", "Messaging"), ("Team", "the Messaging team")))
        self.assertEqual(res.merges[0]["method"], "normalised")
        self.assertEqual(res.merges[0]["chunks"], ["catalog:3"])
        self.assertEqual(res.canonical_name(res.root[("Team", "messaging")], s), "Messaging")

    def test_error_codes_never_merge(self):
        s = surfaces(("ErrorCode", "KST-5031", "rb:8"), ("ErrorCode", "KST-5038", "rb:8"),
                     ("ErrorCode", "KST-5031 - Scheduled report not delivered", "rb:7"))
        res = gr.resolve(s)
        self.assertFalse(same(res, ("ErrorCode", "KST-5031"), ("ErrorCode", "KST-5038")))
        self.assertTrue(same(res, ("ErrorCode", "KST-5031"), ("ErrorCode", "KST-5031 - Scheduled report not delivered")))

    def test_accented_names_merge(self):
        s = surfaces(("Team", "Inès Moreau", "c:1"), ("Team", "Ines Moreau", "c:2"))
        self.assertTrue(same(gr.resolve(s), ("Team", "Inès Moreau"), ("Team", "Ines Moreau")))

    def test_different_types_never_merge(self):
        s = surfaces(("Service", "Payments Core", "c:3"), ("Team", "Payments Core", "c:4"),
                     ("Feature", "Owner", "g:1"), ("Team", "Owner", "c:0"))
        res = gr.resolve(s)
        self.assertEqual(len(res.clusters), 4)
        self.assertEqual(res.merges, [])

    def test_plans_merge_by_identifier(self):
        s = surfaces(("Plan", "Enterprise", "g:1"), ("Plan", "Enterprise plan", "g:5"),
                     ("Plan", "Enterprise plans", "b:2"), ("Plan", "Team", "g:1"), ("Plan", "Plan", "b:3"))
        res = gr.resolve(s)
        self.assertTrue(same(res, ("Plan", "Enterprise"), ("Plan", "Enterprise plans")))
        self.assertFalse(same(res, ("Plan", "Team"), ("Plan", "Plan")))
        self.assertEqual({m["method"] for m in res.merges}, {"identifier"})

    def test_alias_rule(self):
        s = surfaces(("Feature", "single sign-on", "rb:2"), ("Feature", "SSO", "g:6"))
        links = [(("Feature", "single sign-on"), "SSO", "rb:2")]
        self.assertTrue(same(gr.resolve(s, links), ("Feature", "single sign-on"), ("Feature", "SSO")))
        self.assertFalse(same(gr.resolve(s, links, rules=("normalised",)), ("Feature", "single sign-on"),
                              ("Feature", "SSO")))

    def test_no_rules_is_the_t1_graph(self):
        s = surfaces(("Team", "Messaging", "c:2"), ("Team", "the Messaging team", "c:3"))
        res = gr.resolve(s, rules=())
        self.assertEqual((len(res.clusters), res.merges), (2, []))

    def test_mailman_is_never_merged_into_postroom(self):
        # the trap of the ticket: the LLM lists mailman as an alias of postroom, AND the catalog says postroom replaced it
        s = surfaces(("Service", "postroom", "c:2"), ("Service", "mailman", "c:2"))
        links = [(("Service", "postroom"), "mailman", "c:2")]
        edges = [(("Service", "postroom"), "replaces", ("Service", "mailman"))]
        self.assertTrue(same(gr.resolve(s, links), ("Service", "postroom"), ("Service", "mailman")))  # without the guard
        res = gr.resolve(s, links, never_merge=gr.replaces_pairs(edges))
        self.assertFalse(same(res, ("Service", "postroom"), ("Service", "mailman")))
        self.assertEqual(res.refused[0]["reason"], "linked by a replaces edge")
        self.assertEqual((res.refused[0]["a"], res.refused[0]["b"]), (["postroom"], ["mailman"]))

    def test_unknown_rule(self):
        with self.assertRaises(ValueError):
            gr.resolve({}, rules=("embedding",))


class ResolvedBuildTests(GraphTestCase):
    def build(self, **kw):
        return graph.build("kestrel", records=RECORDS, llm=FakeLLM({k: list(v) for k, v in SCRIPT.items()}),
                           delay_s=0, log=lambda *_: None, **kw)

    def test_counts_merges_table_and_aliases(self):
        s = self.build()
        r = s["resolution"]
        self.assertEqual((r["entities_before"], r["entities_after"], r["merges"]), (5, 4, 1))
        self.assertEqual(s["entities"], 4)
        self.assertTrue(s["integrity"]["ok"], s["integrity"])
        rows = self.query("SELECT n.name, m.merged_names, m.method, m.evidence_chunk_ids FROM merges m "
                          "JOIN nodes n ON n.id = m.node_id")
        self.assertEqual(rows, [("Messaging", '["the Messaging team"]', "normalised", '["service-catalog_md:3"]')])
        aliases = self.query("SELECT a.alias, a.chunk_id FROM aliases a JOIN nodes n ON n.id=a.node_id "
                             "WHERE n.name='Messaging'")
        self.assertEqual(aliases, [("the Messaging team", "service-catalog_md:3")])   # still searchable for T3
        exported = (graph.RESULTS_DIR / "merges.jsonl").read_text(encoding="utf-8")
        self.assertIn('"method": "normalised"', exported)
        self.assertIn("resolution         5 -> 4 entities", graph.format_summary(s))

    def test_messaging_node_keeps_the_evidence_of_both_names(self):
        self.build()
        chunks = self.query("SELECT e.chunk_id FROM evidence e JOIN nodes n ON e.kind='node' AND e.owner_id=n.id "
                            "WHERE n.name='Messaging' ORDER BY e.chunk_id")
        self.assertEqual([c[0] for c in chunks], ["service-catalog_md:2", "service-catalog_md:3"])

    def test_mailman_stays_a_separate_node(self):
        s = self.build()
        names = {n for (n,) in self.query("SELECT name FROM nodes WHERE type='Service'")}
        self.assertEqual(names, {"postroom", "mailman"})
        self.assertEqual(s["resolution"]["refused"][0]["reason"], "linked by a replaces edge")
        edge = self.query("SELECT n1.name, e.relation, n2.name FROM edges e JOIN nodes n1 ON n1.id=e.source_id "
                          "JOIN nodes n2 ON n2.id=e.target_id WHERE e.relation='replaces'")
        self.assertEqual(edge, [("postroom", "replaces", "mailman")])

    def test_integrity_flags_a_merge_without_evidence(self):
        self.build()
        con = graph.connect("kestrel")
        con.execute("UPDATE merges SET evidence_chunk_ids='[]'")
        con.commit()
        result = graph.check_integrity(con, RECORDS)
        con.close()
        self.assertEqual((result["ok"], result["merges_without_method_or_evidence"]), (False, 1))


if __name__ == "__main__":
    unittest.main()
