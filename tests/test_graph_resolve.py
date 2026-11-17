"""Entity resolution (v3T02)."""
import unittest

from app import graph_resolve as gr


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

    def test_unknown_rule(self):
        with self.assertRaises(ValueError):
            gr.resolve({}, rules=("embedding",))


if __name__ == "__main__":
    unittest.main()
