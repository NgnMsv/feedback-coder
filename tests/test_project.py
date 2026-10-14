"""Tests on the real project files and the whole pipeline end to end."""

import unittest

from feedback_coder.codebook import load_codebook
from feedback_coder.data import load_comments, load_gold
from feedback_coder.evaluate import evaluate
from feedback_coder.pipeline import run
from feedback_coder.prompts import system_prompt, user_prompt
from feedback_coder.providers import KeywordCoder
from feedback_coder.report import highlight, render
from feedback_coder.themes import build_themes
from feedback_coder.verify import VerifiedComment
from helpers import COMMENTS, ROOT


class DataTests(unittest.TestCase):
    def setUp(self):
        self.codebook = load_codebook(ROOT / "codebook.yaml")
        self.comments = load_comments(ROOT / "data/feedback.csv")
        self.gold = load_gold(ROOT / "data/gold_labels.csv")

    def test_every_comment_has_a_human_label(self):
        self.assertEqual({c.id for c in self.comments}, set(self.gold))

    def test_gold_labels_only_use_codebook_codes_and_sentiments(self):
        for g in self.gold.values():
            self.assertLessEqual(g.codes, set(self.codebook.ids), g.id)
            self.assertIn(g.sentiment, self.codebook.sentiments, g.id)

    def test_every_code_has_definition_and_examples(self):
        for c in self.codebook.codes:
            self.assertTrue(c.definition and c.examples, c.id)

    def test_prompt_contains_every_code_and_the_comments(self):
        sp = system_prompt(self.codebook)
        for c in self.codebook.codes:
            self.assertIn(c.id, sp)
        up = user_prompt(self.comments[:2])
        self.assertIn('"id": "F001"', up)
        self.assertIn("previous answer", user_prompt(self.comments[:1], error_hint="bad"))

    def test_end_to_end_with_the_offline_baseline(self):
        result = run(self.comments, self.codebook, KeywordCoder())
        self.assertEqual(len(result.comments), len(self.comments))
        ev = evaluate(result.comments, self.gold, self.codebook)
        self.assertEqual(ev.n, 60)
        self.assertGreater(ev.micro_f1, 0.3)  # a sanity floor, not a target
        page = render(result, ev, build_themes(result.comments, self.codebook), self.codebook)
        for heading in ("What users talk about", "Does it agree with a human coder?", "All comments"):
            self.assertIn(heading, page)


class ReportSafetyTests(unittest.TestCase):
    def test_comment_html_is_escaped(self):
        out = highlight('<script>alert(1)</script> it crashes', ["it crashes"])
        self.assertNotIn("<script>", out)
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("<mark>it crashes</mark>", out)

    def test_themes_count_comments(self):
        cb = load_codebook(ROOT / "codebook.yaml")
        vcs = [VerifiedComment(c.id, c.text) for c in COMMENTS]
        self.assertTrue(all(t.count == 0 for t in build_themes(vcs, cb)))


if __name__ == "__main__":
    unittest.main()
