"""Step 1 – load the input data.

Two CSV files:
  data/feedback.csv     id, source, date, text   – the comments to code
  data/gold_labels.csv  id, codes, sentiment     – how a human coded them

`codes` in the gold file is a semicolon-separated list, e.g. "pricing;support".
An empty value means "no code applies".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Comment:
    id: str
    source: str
    date: str
    text: str


@dataclass(frozen=True)
class GoldLabel:
    id: str
    codes: frozenset[str]
    sentiment: str


def load_comments(path: str | Path) -> list[Comment]:
    with open(path, newline="", encoding="utf-8") as f:
        comments = [Comment(r["id"], r["source"], r["date"], r["text"].strip()) for r in csv.DictReader(f)]
    ids = [c.id for c in comments]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: comment ids must be unique")
    return comments


def load_gold(path: str | Path) -> dict[str, GoldLabel]:
    gold: dict[str, GoldLabel] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            codes = frozenset(c.strip() for c in r["codes"].split(";") if c.strip())
            gold[r["id"]] = GoldLabel(r["id"], codes, r["sentiment"].strip())
    return gold
