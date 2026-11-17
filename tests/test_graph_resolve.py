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


if __name__ == "__main__":
    unittest.main()
