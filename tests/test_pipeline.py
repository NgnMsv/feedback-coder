import tempfile
import unittest
from pathlib import Path

from feedback_coder.pipeline import Cache, batched, parse_answer, run

from helpers import COMMENTS, TINY_CODEBOOK, FakeCoder, answer

GOOD = answer(
    {"id": "A1", "codes": [{"code": "bug_crash", "evidence": "crashes on start", "confidence": 0.9}], "sentiment": "negative"},
    {"id": "A2", "codes": [{"code": "pricing", "evidence": "price is fair", "confidence": 0.8}], "sentiment": "positive"},
)


class ParseTests(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(len(parse_answer(GOOD).results), 2)

    def test_markdown_fence_is_tolerated(self):
        self.assertEqual(len(parse_answer("```json\n" + GOOD + "\n```").results), 2)

    def test_schema_violation_raises(self):
        bad = answer({"id": "A1", "codes": [{"code": "x", "evidence": "y", "confidence": 7}], "sentiment": "negative"})
        with self.assertRaises(ValueError):
            parse_answer(bad)


class RunTests(unittest.TestCase):
    def test_batching(self):
        self.assertEqual([len(b) for b in batched(list(range(25)), 10)], [10, 10, 5])

    def test_retry_sends_the_error_back_and_then_succeeds(self):
        coder = FakeCoder("this is not json", GOOD)
        result = run(COMMENTS, TINY_CODEBOOK, coder)
        self.assertEqual(result.stats.retries, 1)
        self.assertIsNone(coder.calls[0])
        self.assertIn("Invalid JSON", coder.calls[1])  # the validation error went back to the model
        self.assertEqual([c.code_ids for c in result.comments], [{"bug_crash"}, {"pricing"}])

    def test_gives_up_after_three_attempts_and_flags_the_batch(self):
        coder = FakeCoder("x", "y", "z")
        result = run(COMMENTS, TINY_CODEBOOK, coder)
        self.assertEqual(len(coder.calls), 3)
        self.assertTrue(all(c.flags[0].kind == "invalid_output" for c in result.comments))

    def test_cache_avoids_a_second_api_call_and_replays_offline(self):
        with tempfile.TemporaryDirectory() as d:
            first = run(COMMENTS, TINY_CODEBOOK, FakeCoder(GOOD), cache_dir=Path(d))
            self.assertEqual(first.stats.api_calls, 1)
            replay = run(COMMENTS, TINY_CODEBOOK, FakeCoder(), cache_dir=Path(d), offline_ok=True)
            self.assertEqual((replay.stats.api_calls, replay.stats.cache_hits), (0, 1))
            self.assertEqual([c.code_ids for c in replay.comments], [c.code_ids for c in first.comments])

    def test_offline_replay_without_cache_fails_clearly(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(RuntimeError):
            run(COMMENTS, TINY_CODEBOOK, FakeCoder(), cache_dir=Path(d), offline_ok=True)

    def test_cache_file_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.jsonl"
            Cache(path).put("k", "raw text")
            self.assertEqual(Cache(path).get("k"), "raw text")


if __name__ == "__main__":
    unittest.main()
