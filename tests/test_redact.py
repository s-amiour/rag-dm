"""v4T09: personal data redaction (app/redact.py) and where it is applied.

The key check is the last class: with a fake LLM client, the prompt that would leave the
machine contains no e-mail, phone number or name, and the answer shown to the user has
the real values back. No API key needed.
"""
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from app import core, redact


class RedactRulesTest(unittest.TestCase):
    def test_email_phone_name(self):
        text = "Hi, my name is Léa Fontaine, call me on +33 6 12 34 56 78 or write to lea@example.com"
        out, mapping = redact.redact(text)
        self.assertEqual(out, "Hi, my name is [NAME_1], call me on [PHONE_1] or write to [EMAIL_1]")
        self.assertEqual(mapping["[NAME_1]"], "Léa Fontaine")
        self.assertEqual(redact.restore(out, mapping), text)

    def test_phone_formats(self):
        for phone in ("+33 6 12 34 56 78", "06 12 34 56 78", "06.12.34.56.78", "+44 20 7946 0958",
                      "(555) 123-4567", "555-123-4567"):
            with self.subTest(phone=phone):
                self.assertEqual(redact.redact(f"Call {phone} today")[0], "Call [PHONE_1] today")

    def test_card_and_iban_need_a_valid_checksum(self):
        out, _ = redact.redact("Card 4111 1111 1111 1111, IBAN FR76 3000 6000 0112 3456 7890 189")
        self.assertEqual(out, "Card [CARD_1], IBAN [IBAN_1]")
        # same shapes, wrong checksum: an order number is not a card
        self.assertEqual(redact.redact("Order 1234567890123")[0], "Order 1234567890123")

    def test_corpus_facts_are_left_alone(self):
        # difficult case: things that look like numbers or names but are answers we need
        text = ("Error KST-5031 on 2026-11-18. Allow 203.0.113.10 and 203.0.113.11. "
                "Refund within 10 business days, tank 1.8 litres, version 4.2.1, 49 EUR. "
                "Contact Messaging on #oncall-messaging. I am locked out.")
        out, mapping = redact.redact(text)
        self.assertEqual(out, text)
        self.assertEqual(mapping, {})

    def test_same_value_same_placeholder_across_texts(self):
        _, mapping = redact.redact("write to a@b.co")
        out, mapping = redact.redact("again a@b.co, or c@d.co", mapping)
        self.assertEqual(out, "again [EMAIL_1], or [EMAIL_2]")

    def test_restore_handles_ten_and_more(self):
        mapping = {f"[EMAIL_{i}]": f"user{i}@x.co" for i in range(1, 12)}
        self.assertEqual(redact.restore("[EMAIL_1] [EMAIL_11]", mapping), "user1@x.co user11@x.co")

    def test_counts_hold_no_values(self):
        _, mapping = redact.redact("my name is Sara, sara@x.co, 06 12 34 56 78")
        self.assertEqual(redact.counts(mapping), {"EMAIL": 1, "PHONE": 1, "NAME": 1})

    def test_known_miss_name_without_a_cue(self):
        # D-37: names are only found after a cue ("my name is", Mr, Dr...)
        self.assertEqual(redact.redact("Jonas Berg leads Platform.")[0], "Jonas Berg leads Platform.")

    def test_role_and_plan_words_are_not_names(self):
        for q in ("I am Owner of the workspace, how do I transfer it?",
                  "I'm Admin and can't add a connector",
                  "I am Enterprise customer, what is the SLA?"):
            self.assertEqual(redact.redact(q), (q, {}), q)
        out, mapping = redact.redact("I am Lea Martin, Owner of the workspace")
        self.assertEqual(out, "I am [NAME_1], Owner of the workspace")

    def test_switch_off(self):
        with mock.patch.dict(os.environ, {"PII_REDACTION": "0"}):
            self.assertFalse(redact.enabled())
        with mock.patch.dict(os.environ, {"PII_REDACTION": "1"}):
            self.assertTrue(redact.enabled())


ENV = {"LLM_API_KEY": "k", "LLM_BASE_URL": "http://fake", "LLM_MODEL": "fake-model"}
HITS = [{"id": "policy:3", "text": "Disputed invoices: write to billing@kestrel.example within 30 days."}]


def fake_client(reply):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=reply), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
    )
    client = mock.MagicMock()
    client.chat.completions.create.return_value = response
    return client


class AppliedBeforeTheLlmTest(unittest.TestCase):
    QUESTION = "I'm Sara Klein (sara.klein@acme.com, +33 6 12 34 56 78). Who do I write to about a disputed invoice?"

    def ask(self, reply, **env):
        client = fake_client(reply)
        with mock.patch.dict(os.environ, {**ENV, **env}), \
             mock.patch("app.core.load_dotenv"), \
             mock.patch("openai.OpenAI", return_value=client):
            raw, usage = core.generate_answer_with_usage(self.QUESTION, HITS)
        sent = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        return json.loads(raw), usage, sent

    def test_prompt_leaving_the_machine_has_no_personal_data(self):
        reply = '{"answer": "Write to [EMAIL_2], [NAME_1].", "citations": ["policy:3"]}'
        answer, usage, sent = self.ask(reply)
        for secret in ("Sara Klein", "sara.klein@acme.com", "+33 6 12 34 56 78", "billing@kestrel.example"):
            self.assertNotIn(secret, sent)
        self.assertIn("[EMAIL_1]", sent)
        # the user sees the real values again
        self.assertEqual(answer["answer"], "Write to billing@kestrel.example, Sara Klein.")
        self.assertEqual(usage["redactions"], {"EMAIL": 2, "PHONE": 1, "NAME": 1})

    def test_switch_off_sends_text_as_is(self):
        _, usage, sent = self.ask('{"answer": "ok", "citations": []}', PII_REDACTION="0")
        self.assertIn("sara.klein@acme.com", sent)
        self.assertEqual(usage["redactions"], {})


if __name__ == "__main__":
    unittest.main()
