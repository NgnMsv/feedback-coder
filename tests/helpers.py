"""Shared test helpers: paths, a tiny codebook, and a fake coder."""

from __future__ import annotations

import json
from pathlib import Path

from feedback_coder.codebook import Code, Codebook
from feedback_coder.data import Comment

ROOT = Path(__file__).resolve().parent.parent

TINY_CODEBOOK = Codebook(
    codes=[
        Code("bug_crash", "Bugs", "crashes and errors", keywords=["crashes"]),
        Code("pricing", "Pricing", "price and billing", keywords=["price"]),
    ],
    sentiments=["positive", "negative", "mixed", "neutral"],
)

COMMENTS = [
    Comment("A1", "test", "2026-01-01", "The app crashes on start."),
    Comment("A2", "test", "2026-01-01", "The price is fair."),
]


def answer(*results: dict) -> str:
    return json.dumps({"results": list(results)})


class FakeCoder:
    """Returns pre-written answers one after another, and records every call."""

    name = "fake"
    model = "fake-1"

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.calls: list[str | None] = []

    def code_batch(self, comments, codebook, error_hint=None):
        self.calls.append(error_hint)
        return self.answers.pop(0)
