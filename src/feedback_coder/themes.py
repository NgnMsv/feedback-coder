"""Step 8 – from coded comments to themes.

This is the part a product team actually reads: which topics come up most,
how people feel about each one, which topics appear together, and a few
representative quotes per topic (the evidence the model had to provide).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations

from .codebook import Codebook
from .verify import VerifiedComment


@dataclass
class Theme:
    code: str
    label: str
    count: int
    share: float  # share of all comments
    sentiment: Counter = field(default_factory=Counter)
    quotes: list[tuple[str, str]] = field(default_factory=list)  # (comment id, evidence)

    @property
    def negative_share(self) -> float:
        return self.sentiment["negative"] / self.count if self.count else 0.0


def build_themes(comments: list[VerifiedComment], codebook: Codebook, quotes_per_theme: int = 3) -> list[Theme]:
    n = len(comments) or 1
    themes = []
    for code in codebook.codes:
        hits = [(c, vc) for c in comments for vc in c.codes if vc.code == code.id]
        theme = Theme(code.id, code.label, len(hits), len(hits) / n)
        for c, _ in hits:
            theme.sentiment[c.sentiment] += 1
        best = sorted(hits, key=lambda h: -h[1].confidence)[:quotes_per_theme]
        theme.quotes = [(c.id, vc.evidence) for c, vc in best]
        themes.append(theme)
    return sorted(themes, key=lambda t: -t.count)


def co_occurrence(comments: list[VerifiedComment], top: int = 5) -> list[tuple[str, str, int]]:
    pairs: Counter = Counter()
    for c in comments:
        for a, b in combinations(sorted(c.code_ids), 2):
            pairs[(a, b)] += 1
    return [(a, b, k) for (a, b), k in pairs.most_common(top)]
