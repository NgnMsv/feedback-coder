import io
import json
import os
import unittest
from unittest import mock

from feedback_coder.codebook import load_codebook
from feedback_coder.providers import KeywordCoder, make_coder
from feedback_coder.providers.anthropic_client import AnthropicClient
from feedback_coder.providers.openai_client import OpenAIClient

from helpers import COMMENTS, ROOT, TINY_CODEBOOK


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def fake_urlopen(payload: dict, seen: list):
    def _open(req, timeout=None):
        seen.append(req)
        return FakeResponse(json.dumps(payload).encode())
    return _open


class HttpClientTests(unittest.TestCase):
    def test_openai_request_and_response(self):
        seen = []
        payload = {"choices": [{"message": {"content": '{"results": []}'}}]}
        with mock.patch("urllib.request.urlopen", fake_urlopen(payload, seen)):
            out = OpenAIClient("sk-test", "gpt-test").complete("SYS", "USER")
        self.assertEqual(out, '{"results": []}')
        req = seen[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "https://api.openai.com/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer sk-test")
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual([m["role"] for m in body["messages"]], ["system", "user"])

    def test_anthropic_request_and_response(self):
        seen = []
        payload = {"content": [{"type": "text", "text": '{"results": '}, {"type": "text", "text": "[]}"}]}
        with mock.patch("urllib.request.urlopen", fake_urlopen(payload, seen)):
            out = AnthropicClient("ak-test", "claude-test").complete("SYS", "USER")
        self.assertEqual(out, '{"results": []}')
        req = seen[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "https://api.anthropic.com/v1/messages")
        self.assertEqual(req.get_header("X-api-key"), "ak-test")
        self.assertEqual(body["system"], "SYS")
        self.assertEqual(body["messages"], [{"role": "user", "content": "USER"}])


class MakeCoderTests(unittest.TestCase):
    def test_missing_key_gives_a_clear_error(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(RuntimeError):
            make_coder("anthropic")

    def test_cached_mode_needs_no_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(make_coder("openai", require_key=False).name, "openai")

    def test_unknown_provider(self):
        with self.assertRaises(ValueError):
            make_coder("gemini")


class KeywordCoderTests(unittest.TestCase):
    def test_output_has_the_same_shape_as_an_llm_answer(self):
        data = json.loads(KeywordCoder().code_batch(COMMENTS, TINY_CODEBOOK))
        self.assertEqual([r["id"] for r in data["results"]], ["A1", "A2"])
        self.assertEqual(data["results"][0]["codes"][0]["code"], "bug_crash")
        self.assertEqual(data["results"][1]["codes"][0]["code"], "pricing")

    def test_whole_word_matching(self):
        # "plan" must not match inside "explained"
        cb = load_codebook(ROOT / "codebook.yaml")
        from feedback_coder.data import Comment
        data = json.loads(KeywordCoder().code_batch([Comment("x", "", "", "She explained it well.")], cb))
        self.assertEqual(data["results"][0]["codes"], [])

    def test_evidence_is_a_real_sentence_from_the_comment(self):
        data = json.loads(KeywordCoder().code_batch(COMMENTS, TINY_CODEBOOK))
        self.assertIn(data["results"][0]["codes"][0]["evidence"], COMMENTS[0].text)


if __name__ == "__main__":
    unittest.main()
