"""T12: the LLM reply is validated with Pydantic (core.LLMAnswer).

These tests call parse_llm_answer directly, so no LLM and no API key are needed.
"""
import json
import unittest

from app.core import parse_llm_answer

ALLOWED = {"runbook:3", "catalog:1"}


class ParseLLMAnswerTest(unittest.TestCase):
    def test_valid_answer(self):
        content = json.dumps({"answer": "Add both IPs.", "citations": ["runbook:3"]})
        result = parse_llm_answer(content, ALLOWED)
        self.assertTrue(result["valid"])
        self.assertEqual(result["answer"], "Add both IPs.")
        self.assertEqual(result["citations"], ["runbook:3"])

    def test_json_inside_code_fence(self):
        content = '```json\n{"answer": "Yes.", "citations": ["catalog:1"]}\n```'
        result = parse_llm_answer(content, ALLOWED)
        self.assertTrue(result["valid"])
        self.assertEqual(result["citations"], ["catalog:1"])

    def test_plain_text_is_invalid(self):
        result = parse_llm_answer("Sure! The answer is to add both IPs.", ALLOWED)
        self.assertFalse(result["valid"])
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["citations"], [])

    def test_json_list_is_invalid(self):
        # Before T12 this crashed with AttributeError (list has no .get).
        result = parse_llm_answer('["runbook:3"]', ALLOWED)
        self.assertFalse(result["valid"])

    def test_missing_key_is_invalid(self):
        result = parse_llm_answer('{"answer": "Yes."}', ALLOWED)
        self.assertFalse(result["valid"])

    def test_wrong_type_is_invalid(self):
        result = parse_llm_answer('{"answer": 42, "citations": []}', ALLOWED)
        self.assertFalse(result["valid"])

    def test_empty_reply_is_invalid(self):
        result = parse_llm_answer("", ALLOWED)
        self.assertFalse(result["valid"])

    def test_unknown_citation_is_dropped(self):
        content = json.dumps({"answer": "Yes.", "citations": ["runbook:3", "made-up:9"]})
        result = parse_llm_answer(content, ALLOWED)
        self.assertTrue(result["valid"])
        self.assertEqual(result["citations"], ["runbook:3"])
        self.assertEqual(result["dropped_citations"], 1)


if __name__ == "__main__":
    unittest.main()