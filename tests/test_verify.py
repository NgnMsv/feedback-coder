import unittest

from feedback_coder.schema import CodeAssignment, CodedComment
from feedback_coder.verify import evidence_in_text, normalize, verify_batch
from helpers import COMMENTS, TINY_CODEBOOK


def coded(id_, sentiment="negative", **codes):
    return CodedComment(
        id=id_,
        sentiment=sentiment,
        codes=[CodeAssignment(code=c, evidence=e, confidence=conf) for c, (e, conf) in codes.items()],
    )


class EvidenceTests(unittest.TestCase):
    def test_exact_quote_is_found(self):
        self.assertTrue(evidence_in_text("crashes on start", "The app crashes on start."))

    def test_case_spaces_quotes_and_trailing_punctuation_do_not_matter(self):
        self.assertTrue(evidence_in_text("The  APP crashes on start.", "The app crashes on start."))
        self.assertTrue(evidence_in_text("can’t find it", "I can't find it anywhere"))

    def test_paraphrase_is_rejected(self):
        self.assertFalse(evidence_in_text("the application crashed", "The app crashes on start."))

    def test_empty_quote_is_rejected(self):
        self.assertFalse(evidence_in_text("  ", "The app crashes."))

    def test_normalize(self):
        self.assertEqual(normalize("  “Hello”  World! "), "hello\" world")


class VerifyBatchTests(unittest.TestCase):
    def test_valid_answer_passes_without_flags(self):
        answers = [coded("A1", bug_crash=("crashes on start", 0.9)),
                   coded("A2", "positive", pricing=("The price is fair", 0.8))]
        verified, batch_flags = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual([v.code_ids for v in verified], [{"bug_crash"}, {"pricing"}])
        self.assertFalse(any(v.needs_review for v in verified))
        self.assertEqual(batch_flags, [])

    def test_unknown_code_is_dropped_and_flagged(self):
        answers = [coded("A1", ux=("crashes", 0.9)), coded("A2", "positive")]
        verified, _ = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual(verified[0].code_ids, frozenset())
        self.assertEqual(verified[0].flags[0].kind, "unknown_code")

    def test_invented_quote_is_dropped_and_flagged(self):
        answers = [coded("A1", bug_crash=("the app is very slow", 0.9)), coded("A2", "positive")]
        verified, _ = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual(verified[0].code_ids, frozenset())
        self.assertEqual(verified[0].flags[0].kind, "evidence_missing")

    def test_low_confidence_is_kept_but_sent_to_review(self):
        answers = [coded("A1", bug_crash=("crashes", 0.3)), coded("A2", "positive")]
        verified, _ = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual(verified[0].code_ids, {"bug_crash"})
        self.assertEqual(verified[0].flags[0].kind, "low_confidence")

    def test_missing_and_extra_ids(self):
        answers = [coded("A1", bug_crash=("crashes", 0.9)), coded("ZZZ")]
        verified, batch_flags = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual([v.id for v in verified], ["A1", "A2"])  # input order kept
        self.assertEqual(verified[1].flags[0].kind, "missing_result")
        self.assertEqual(batch_flags[0].kind, "extra_result")

    def test_duplicate_code_counts_once(self):
        answers = [coded("A1", bug_crash=("crashes", 0.9)), coded("A2", "positive")]
        answers[0].codes.append(CodeAssignment(code="bug_crash", evidence="on start", confidence=0.9))
        verified, _ = verify_batch(COMMENTS, answers, TINY_CODEBOOK)
        self.assertEqual(len(verified[0].codes), 1)


if __name__ == "__main__":
    unittest.main()
