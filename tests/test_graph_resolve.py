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


if __name__ == "__main__":
    unittest.main()
