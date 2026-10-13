"""Step 2 – load the codebook (codebook.yaml).

The codebook is the single source of truth for the categories. The prompt is
built from it and the checks in verify.py use it, so adding a code means
editing one YAML file and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Code:
    id: str
    label: str
    definition: str
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Codebook:
    codes: list[Code]
    sentiments: list[str]

    @property
    def ids(self) -> list[str]:
        return [c.id for c in self.codes]

    def label(self, code_id: str) -> str:
        return next((c.label for c in self.codes if c.id == code_id), code_id)


def load_codebook(path: str | Path) -> Codebook:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    codes = [Code(**c) for c in raw["codes"]]
    ids = [c.id for c in codes]
    if len(ids) != len(set(ids)):
        raise ValueError("codebook: code ids must be unique")
    return Codebook(codes=codes, sentiments=list(raw["sentiments"]))
