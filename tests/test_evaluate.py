import unittest

from feedback_coder.data import GoldLabel
from feedback_coder.evaluate import CodeScore, cohens_kappa, evaluate
from feedback_coder.verify import VerifiedCode, VerifiedComment

from helpers import TINY_CODEBOOK


def vc(id_, *codes, sentiment="negative"):
    return VerifiedComment(id=id_, text="", codes=[VerifiedCode(c, "x", 0.9) for c in codes], sentiment=sentiment)


class KappaTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertAlmostEqual(cohens_kappa(tp=5, fp=0, fn=0, tn=5), 1.0)

    def test_textbook_example(self):
        # Classic 2x2 example: 20 both yes, 15 both no, 5 + 10 disagreements, n = 50.
        # p_o = 0.70, p_e = 0.50 * 0.60 + 0.50 * 0.40 = 0.50, kappa = 0.40
        self.assertAlmostEqual(cohens_kappa(tp=20, fp=5, fn=10, tn=15), 0.4)

    def test_agreement_no_better_than_chance_is_zero(self):
        self.assertAlmostEqual(cohens_kappa(tp=25, fp=25, fn=25, tn=25), 0.0)

    def test_empty_is_zero(self):
        self.assertEqual(cohens_kappa(0, 0, 0, 0), 0.0)


class ScoreTests(unittest.TestCase):
    def test_precision_recall_f1(self):
        s = CodeScore("x", tp=6, fp=2, fn=4, tn=8)
        self.assertAlmostEqual(s.precision, 0.75)
        self.assertAlmostEqual(s.recall, 0.6)
        self.assertAlmostEqual(s.f1, 2 * 0.75 * 0.6 / 1.35)

    def test_no_predictions_gives_zero_not_an_error(self):
        s = CodeScore("x", tp=0, fp=0, fn=3, tn=7)
        self.assertEqual((s.precision, s.recall, s.f1), (0.0, 0.0, 0.0))


class EvaluateTests(unittest.TestCase):
    def test_multi_label_counts_and_disagreements(self):
        gold = {
            "1": GoldLabel("1", frozenset({"bug_crash"}), "negative"),
            "2": GoldLabel("2", frozenset({"pricing", "bug_crash"}), "negative"),
            "3": GoldLabel("3", frozenset(), "neutral"),
        }
        comments = [vc("1", "bug_crash"), vc("2", "pricing"), vc("3", "pricing", sentiment="neutral")]
        ev = evaluate(comments, gold, TINY_CODEBOOK)
        bug = next(s for s in ev.per_code if s.code == "bug_crash")
        price = next(s for s in ev.per_code if s.code == "pricing")
        self.assertEqual((bug.tp, bug.fp, bug.fn, bug.tn), (1, 0, 1, 1))
        self.assertEqual((price.tp, price.fp, price.fn, price.tn), (1, 1, 0, 1))
        self.assertAlmostEqual(ev.exact_match, 1 / 3)
        self.assertEqual([d["id"] for d in ev.disagreements], ["2", "3"])
        self.assertEqual(ev.disagreements[0]["missed"], ["bug_crash"])
        self.assertEqual(ev.disagreements[1]["extra"], ["pricing"])
        # micro: TP 2, FP 1, FN 1 -> precision = recall = F1 = 2/3
        self.assertAlmostEqual(ev.micro_f1, 2 / 3)
        self.assertAlmostEqual(ev.sentiment_accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
