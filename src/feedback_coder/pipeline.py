"""Step 6 – the pipeline: batches, retries and a cache.

    comments ──► batches of N ──► coder ──► parse + validate ──► verify ──► results
                                   ▲             │ invalid?
                                   └── retry with the error message (max 2×)

Batching:  sending 10 comments per request is cheaper and faster than 60
           separate requests, and still small enough for the model to stay accurate.
Retries:   if the answer is not valid JSON or breaks the schema, the exact
           validation error is sent back and the model gets another try.
Cache:     every successful raw answer is saved in runs/<provider>__<model>.jsonl,
           keyed by a hash of the prompt. Running again costs nothing, gives the
           same result, and lets you present a real model run without an API key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .codebook import Codebook
from .data import Comment
from .prompts import system_prompt, user_prompt
from .providers import Coder
from .schema import BatchResult
from .verify import Flag, VerifiedComment, verify_batch

MAX_RETRIES = 2


@dataclass
class RunStats:
    batches: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    retries: int = 0


@dataclass
class RunResult:
    provider: str
    model: str
    created: str
    comments: list[VerifiedComment]
    batch_flags: list[Flag] = field(default_factory=list)
    stats: RunStats = field(default_factory=RunStats)


class Cache:
    """A tiny append-only JSON-lines cache: one line per successful batch."""

    def __init__(self, path: Path | None):
        self.path = path
        self._data: dict[str, str] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["key"]] = row["raw"]

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def put(self, key: str, raw: str) -> None:
        self._data[key] = raw
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")


def cache_key(coder: Coder, comments: list[Comment], codebook: Codebook) -> str:
    text = "\n".join([coder.name, coder.model, system_prompt(codebook), user_prompt(comments)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_answer(raw: str) -> BatchResult:
    """Parse the model's text into a BatchResult. Tolerates a ```json fence around it."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return BatchResult.model_validate_json(text)


def batched(items: list[Comment], size: int) -> list[list[Comment]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def run(
    comments: list[Comment],
    codebook: Codebook,
    coder: Coder,
    *,
    batch_size: int = 10,
    cache_dir: Path | None = None,
    offline_ok: bool = False,
    progress=None,
) -> RunResult:
    """Code all comments. `offline_ok=True` means: use the cache only and never
    call the API (for presenting a cached run without an API key)."""
    use_cache = coder.name != "offline" and cache_dir is not None
    cache = Cache(cache_dir / f"{coder.name}__{coder.model}.jsonl" if use_cache else None)
    result = RunResult(
        provider=coder.name,
        model=coder.model,
        created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        comments=[],
    )

    for i, batch in enumerate(batched(comments, batch_size), 1):
        result.stats.batches += 1
        key = cache_key(coder, batch, codebook)
        raw = cache.get(key) if use_cache else None
        parsed: BatchResult | None = None

        if raw is not None:
            result.stats.cache_hits += 1
            parsed = parse_answer(raw)
        elif offline_ok:
            raise RuntimeError(
                f"Batch {i} is not in the cache for {coder.name}/{coder.model}. "
                "Run once with an API key to fill the cache."
            )
        else:
            error_hint = None
            for attempt in range(MAX_RETRIES + 1):
                raw = coder.code_batch(batch, codebook, error_hint)
                if coder.name != "offline":
                    result.stats.api_calls += 1
                try:
                    parsed = parse_answer(raw)
                    break
                except (ValidationError, ValueError) as e:
                    error_hint = str(e)[:1500]
                    if attempt < MAX_RETRIES:
                        result.stats.retries += 1
            if parsed is None:
                # Still invalid after all retries: every comment in the batch goes to human review.
                for c in batch:
                    vc = VerifiedComment(id=c.id, text=c.text)
                    vc.flags.append(Flag("invalid_output", f"no valid JSON after {MAX_RETRIES + 1} attempts"))
                    result.comments.append(vc)
                continue
            if use_cache:
                cache.put(key, raw)

        verified, batch_flags = verify_batch(batch, parsed.results, codebook)
        result.comments.extend(verified)
        result.batch_flags.extend(batch_flags)
        if progress:
            progress(i, len(comments), batch_size)
    return result


def result_to_dict(result: RunResult) -> dict:
    """Plain JSON-serialisable version of a run, written to output/results.json."""
    return {
        "provider": result.provider,
        "model": result.model,
        "created": result.created,
        "stats": result.stats.__dict__,
        "batch_flags": [f.__dict__ for f in result.batch_flags],
        "comments": [
            {
                "id": c.id,
                "text": c.text,
                "sentiment": c.sentiment,
                "codes": [vc.__dict__ for vc in c.codes],
                "flags": [f.__dict__ for f in c.flags],
            }
            for c in result.comments
        ],
    }
