"""Answer generation with reasoning models (DECISIONS D-35): thinking switch,
token limit, and a clear message when the reply is cut off. Fake OpenAI client."""
import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from app import core

HITS = [{"id": "runbook:8", "text": "KST-5031: escalate to the team that owns postroom."}]
ENV = {"LLM_API_KEY": "k", "LLM_BASE_URL": "http://fake", "LLM_MODEL": "fake-model"}


def fake_client(content, finish_reason="stop"):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
    )
    client = mock.MagicMock()
    client.chat.completions.create.return_value = response
    return client


class AnswerGenerationTest(unittest.TestCase):
    def run_with(self, client, **env):
        with mock.patch.dict(os.environ, {**ENV, **env}, clear=False), \
             mock.patch("app.core.load_dotenv"), \
             mock.patch("openai.OpenAI", return_value=client):
            raw, usage = core.generate_answer_with_usage("What is KST-5031?", HITS)
        return json.loads(raw), usage, client.chat.completions.create.call_args.kwargs

    def test_thinking_off_sends_chat_template_kwargs(self):
        client = fake_client('{"answer": "Messaging.", "citations": ["runbook:8"]}')
        answer, usage, sent = self.run_with(client, LLM_DISABLE_THINKING="1")
        self.assertEqual(sent["extra_body"]["chat_template_kwargs"]["enable_thinking"], False)
        self.assertTrue(answer["valid"])
        self.assertEqual(usage["thinking"], "off")

    def test_default_sends_no_extra_body(self):
        client = fake_client('{"answer": "Messaging.", "citations": []}')
        _, usage, sent = self.run_with(client, LLM_DISABLE_THINKING="0")
        self.assertNotIn("extra_body", sent)
        self.assertEqual(usage["thinking"], "default")

    def test_max_tokens_from_env(self):
        client = fake_client('{"answer": "x", "citations": []}')
        _, _, sent = self.run_with(client, LLM_MAX_TOKENS="4096")
        self.assertEqual(sent["max_tokens"], 4096)

    def test_cut_off_reply_gets_a_clear_message(self):
        client = fake_client('{"answer": "The team that ow', finish_reason="length")
        answer, usage, _ = self.run_with(client)
        self.assertFalse(answer["valid"])
        self.assertIn("token limit", answer["error"])
        self.assertEqual(answer["user_message"], core.TRUNCATED_ANSWER_MESSAGE)
        self.assertEqual(usage["finish_reason"], "length")

    def test_valid_reply_cut_at_length_is_still_valid(self):
        # finish_reason=length with complete JSON: keep the answer
        client = fake_client('{"answer": "ok", "citations": []}', finish_reason="length")
        answer, _, _ = self.run_with(client)
        self.assertTrue(answer["valid"])


if __name__ == "__main__":
    unittest.main()