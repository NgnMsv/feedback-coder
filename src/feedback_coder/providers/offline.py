"""The offline baseline: keyword matching, no AI, no API key, no cost.

Why have it?
* It is a BASELINE. "The LLM reaches F1 0.9" means little on its own;
  "a keyword list reaches X, the LLM reaches Y" shows what the model adds.
* It lets anyone run the whole pipeline and see the report in seconds.

It returns exactly the same JSON shape as the LLM, so it goes through the
same verification, evaluation and report. Its typical mistakes are
instructive: it cannot read negation ("not slow anymore" still contains
"slow"), and it cannot tell a request from a complaint.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from ..codebook import Codebook
from ..data import Comment

POSITIVE = {"love", "great", "best", "genius", "perfect", "perfectly", "thanks", "beautiful", "impressed",
            "nice", "enjoy", "helped", "friendly", "solved", "instantly", "fast", "saved", "5 stars", "works"}
NEGATIVE = {"crash", "crashes", "freezes", "error", "broken", "slow", "laggy", "wrong", "confusing", "mess",
            "useless", "never", "annoying", "hate", "too much", "not okay", "can't", "doesn't", "fails",
            "twice", "outdated", "tiny", "forever", "drain", "disconnects", "isn't fixed"}

_SENTENCE = re.compile(r"[^.!?]+[.!?]*")


def _pattern(keyword: str) -> re.Pattern[str]:
    # Whole-word match for normal words ("plan" must not match "explain"),
    # plain substring for symbols like "€".
    escaped = re.escape(keyword)
    if keyword[:1].isalnum() and keyword[-1:].isalnum():
        escaped = rf"\b{escaped}\b"
    return re.compile(escaped, re.IGNORECASE)


def _sentence_around(text: str, index: int) -> str:
    """Return the sentence of `text` that contains character `index`, exactly as written."""
    for m in _SENTENCE.finditer(text):
        if m.start() <= index < m.end():
            return m.group().strip()
    return text.strip()


def _sentiment(text: str) -> str:
    pos = sum(bool(_pattern(w).search(text)) for w in POSITIVE)
    neg = sum(bool(_pattern(w).search(text)) for w in NEGATIVE)
    if pos and neg:
        return "mixed"
    if pos:
        return "positive"
    if neg:
        return "negative"
    return "neutral"


class KeywordCoder:
    name = "offline"
    model = "keyword-baseline"

    def code_batch(self, comments: Sequence[Comment], codebook: Codebook, error_hint: str | None = None) -> str:
        results = []
        for c in comments:
            codes = []
            for code in codebook.codes:
                for kw in code.keywords:
                    m = _pattern(kw).search(c.text)
                    if m:
                        codes.append({"code": code.id, "evidence": _sentence_around(c.text, m.start()), "confidence": 0.6})
                        break  # one hit per code is enough
            results.append({"id": c.id, "codes": codes, "sentiment": _sentiment(c.text)})
        return json.dumps({"results": results}, ensure_ascii=False)
