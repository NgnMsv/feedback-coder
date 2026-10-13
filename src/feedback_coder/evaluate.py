"""Step 7 – measure the result against the human (gold) labels.

Coding is multi-label: one comment can have several codes. So every code is
evaluated as its own yes/no question over all comments:

                      human: yes   human: no
    model: yes           TP           FP
    model: no            FN           TN

precision = TP / (TP + FP)   "when the model says yes, how often is it right?"
recall    = TP / (TP + FN)   "of the real cases, how many did it find?"
F1        = harmonic mean of the two

Cohen's kappa is the standard agreement measure in qualitative research. It
compares the observed agreement with the agreement two coders would reach by
chance. 1 = perfect, 0 = no better than chance. A common reading (Landis &
Koch): 0.61–0.80 substantial, above 0.80 almost perfect.

Micro-F1 adds up TP/FP/FN over all codes (frequent codes weigh more).
Macro-F1 is the plain average of the per-code F1 (every code weighs the same).

Everything is computed by hand here, on purpose: each formula is a few lines
you can explain, with no library hiding the maths.
"""

from __future__ import annotations

from dataclasses import dataclass

from .codebook import Codebook
from .data import GoldLabel
from .verify import VerifiedComment


@dataclass
class CodeScore:
    code: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def support(self) -> int:
        return self.tp + self.fn

    @property
    def precision(self) -> float:
        return safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return safe_div(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return safe_div(2 * p * r, p + r)

    @property
    def kappa(self) -> float:
        return cohens_kappa(self.tp, self.fp, self.fn, self.tn)


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float:
    """Cohen's kappa for two raters and a yes/no decision."""
    n = tp + fp + fn + tn
    if n == 0:
        return 0.0
    observed = (tp + tn) / n
    # Chance agreement: both say yes by chance + both say no by chance.
    model_yes, human_yes = (tp + fp) / n, (tp + fn) / n
    expected = model_yes * human_yes + (1 - model_yes) * (1 - human_yes)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


@dataclass
class Evaluation:
    per_code: list[CodeScore]
    micro_precision: float
    micro_recall: float
    micro_f1: float
    macro_f1: float
    mean_kappa: float
    exact_match: float  # share of comments where the full set of codes is identical
    sentiment_accuracy: float
    disagreements: list[dict]
    n: int

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "micro_precision": round(self.micro_precision, 3),
            "micro_recall": round(self.micro_recall, 3),
            "micro_f1": round(self.micro_f1, 3),
            "macro_f1": round(self.macro_f1, 3),
            "mean_kappa": round(self.mean_kappa, 3),
            "exact_match": round(self.exact_match, 3),
            "sentiment_accuracy": round(self.sentiment_accuracy, 3),
            "per_code": [
                {"code": s.code, "support": s.support, "precision": round(s.precision, 3),
                 "recall": round(s.recall, 3), "f1": round(s.f1, 3), "kappa": round(s.kappa, 3)}
                for s in self.per_code
            ],
            "disagreements": self.disagreements,
        }


def evaluate(comments: list[VerifiedComment], gold: dict[str, GoldLabel], codebook: Codebook) -> Evaluation:
    scored = [c for c in comments if c.id in gold]
    per_code = []
    for code in codebook.ids:
        tp = fp = fn = tn = 0
        for c in scored:
            predicted, actual = code in c.code_ids, code in gold[c.id].codes
            tp += predicted and actual
            fp += predicted and not actual
            fn += actual and not predicted
            tn += not predicted and not actual
        per_code.append(CodeScore(code, tp, fp, fn, tn))

    TP = sum(s.tp for s in per_code)
    FP = sum(s.fp for s in per_code)
    FN = sum(s.fn for s in per_code)
    micro_p, micro_r = safe_div(TP, TP + FP), safe_div(TP, TP + FN)

    disagreements = []
    for c in scored:
        g = gold[c.id]
        if c.code_ids != g.codes:
            disagreements.append({
                "id": c.id,
                "text": c.text,
                "model": sorted(c.code_ids),
                "human": sorted(g.codes),
                "missed": sorted(g.codes - c.code_ids),
                "extra": sorted(c.code_ids - g.codes),
            })

    n = len(scored)
    return Evaluation(
        per_code=per_code,
        micro_precision=micro_p,
        micro_recall=micro_r,
        micro_f1=safe_div(2 * micro_p * micro_r, micro_p + micro_r),
        macro_f1=safe_div(sum(s.f1 for s in per_code), len(per_code)),
        mean_kappa=safe_div(sum(s.kappa for s in per_code), len(per_code)),
        exact_match=safe_div(sum(c.code_ids == gold[c.id].codes for c in scored), n),
        sentiment_accuracy=safe_div(sum(c.sentiment == gold[c.id].sentiment for c in scored), n),
        disagreements=disagreements,
        n=n,
    )
