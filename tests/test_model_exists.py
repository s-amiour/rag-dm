"""v3T14: live check that the model configured in .env answers.

Uses the app's own settings (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_DISABLE_THINKING),
so it checks the model the app really uses. It costs one tiny LLM call, so it only runs
when asked: RUN_LIVE_LLM_TEST=1 python -m pytest tests/test_model_exists.py
Otherwise it is skipped, with the reason shown by `pytest -rs`.
"""
import os
import unittest
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

REQUIRED = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")
MISSING = [name for name in REQUIRED if not os.getenv(name)]
LIVE = os.getenv("RUN_LIVE_LLM_TEST", "0") == "1"


@unittest.skipUnless(LIVE, "live LLM test: set RUN_LIVE_LLM_TEST=1 to run it (one small paid call)")
@unittest.skipIf(MISSING, f"live LLM test: {', '.join(MISSING) or 'nothing'} missing in .env")
class ConfiguredModelTest(unittest.TestCase):
    def test_configured_model_answers(self):
        from openai import OpenAI

        client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                        timeout=60, max_retries=0)
        request = {
            "model": os.environ["LLM_MODEL"],
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 64,
        }
        if os.getenv("LLM_DISABLE_THINKING", "0").strip().lower() in ("1", "true", "yes"):
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False, "thinking": False}}

        response = client.chat.completions.create(**request)
        text = (response.choices[0].message.content or "").strip()

        self.assertTrue(text, f"{os.environ['LLM_MODEL']} returned an empty reply "
                              f"(finish_reason={response.choices[0].finish_reason})")


if __name__ == "__main__":
    unittest.main()