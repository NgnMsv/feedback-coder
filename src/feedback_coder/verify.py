"""Step 5 – verify every answer before trusting it.

This is the most important module. An LLM is confident even when it is
wrong, so nothing it says is used until it passes these checks:

  1. unknown_code      the code is not in the codebook        -> code dropped
  2. evidence_missing  the quoted evidence is not in the text -> code dropped
                       (the model invented or paraphrased a quote)
  3. low_confidence    confidence below the threshold         -> code kept,
                                                                 comment sent to human review
  4. missing_result    the model skipped a comment            -> comment sent to human review
  5. extra_result      the model answered an id we never sent -> answer ignored

Every problem is recorded as a Flag, so the report can show exactly which
comments a human still has to look at, and why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .codebook import Codebook
from .data import Comment
from .schema import CodedComment

LOW_CONFIDENCE = 0.6


@dataclass
class Flag:
    kind: str
    detail: str


@dataclass
class VerifiedCode:
    code: str
    evidence: str
    confidence: float


@dataclass
class VerifiedComment:
    id: str
    text: str
    codes: list[VerifiedCode] = field(default_factory=list)
    sentiment: str = "neutral"
    flags: list[Flag] = field(default_factory=list)

    @property
    def code_ids(self) -> frozenset[str]:
        return frozenset(c.code for c in self.codes)

    @property
    def needs_review(self) -> bool:
        return bool(self.flags)


def normalize(s: str) -> str:
    """Lower-case, unify quotes/dashes and collapse spaces, so harmless differences
    ("can't" vs "can’t", double spaces) do not count as a fake quote."""
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(" .!?\"'")


def evidence_in_text(evidence: str, text: str) -> bool:
    ev = normalize(evidence)
    return bool(ev) and ev in normalize(text)


def verify_batch(
    comments: list[Comment],
    answers: list[CodedComment],
    codebook: Codebook,
    low_confidence: float = LOW_CONFIDENCE,
) -> tuple[list[VerifiedComment], list[Flag]]:
    """Check the answers for one batch. Returns the verified comments (one per
    input comment, in input order) and batch-level flags (extra ids)."""
    by_id = {c.id: c for c in comments}
    answer_by_id: dict[str, CodedComment] = {}
    batch_flags: list[Flag] = []
    for a in answers:
        if a.id not in by_id:
            batch_flags.append(Flag("extra_result", f"answer for unknown id {a.id!r} ignored"))
        else:
            answer_by_id[a.id] = a

    verified: list[VerifiedComment] = []
    for comment in comments:
        vc = VerifiedComment(id=comment.id, text=comment.text)
        answer = answer_by_id.get(comment.id)
        if answer is None:
            vc.flags.append(Flag("missing_result", "the model returned no answer for this comment"))
            verified.append(vc)
            continue

        vc.sentiment = answer.sentiment
        seen: set[str] = set()
        for assignment in answer.codes:
            if assignment.code not in codebook.ids:
                vc.flags.append(Flag("unknown_code", f"{assignment.code!r} is not in the codebook"))
                continue
            if not evidence_in_text(assignment.evidence, comment.text):
                vc.flags.append(
                    Flag("evidence_missing", f"{assignment.code}: quote not found in comment: {assignment.evidence!r}")
                )
                continue
            if assignment.code in seen:
                continue  # the same code twice counts once
            seen.add(assignment.code)
            if assignment.confidence < low_confidence:
                vc.flags.append(
                    Flag("low_confidence", f"{assignment.code}: confidence {assignment.confidence:.2f}")
                )
            vc.codes.append(VerifiedCode(assignment.code, assignment.evidence, assignment.confidence))
        verified.append(vc)
    return verified, batch_flags
