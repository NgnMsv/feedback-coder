"""Command line interface.

    python -m feedback_coder run --provider offline
    python -m feedback_coder run --provider anthropic            # needs ANTHROPIC_API_KEY
    python -m feedback_coder run --provider openai --cached      # replay a saved run, no key needed
    python -m feedback_coder prompt                              # print the system prompt

Each run writes output/results.json, output/metrics.json and output/report.html.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .codebook import load_codebook
from .data import load_comments, load_gold
from .evaluate import evaluate
from .pipeline import result_to_dict, run
from .prompts import system_prompt
from .providers import DEFAULT_MODELS, KeywordCoder, make_coder
from .report import render
from .themes import build_themes

ROOT = Path.cwd()


def load_dotenv(path: Path) -> None:
    """Read KEY=value lines from .env into the environment (keys already set win)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def cmd_run(args: argparse.Namespace) -> int:
    codebook = load_codebook(args.codebook)
    comments = load_comments(args.data)
    gold = load_gold(args.gold)

    coder = make_coder(args.provider, args.model, require_key=not args.cached)
    print(f"Coding {len(comments)} comments with {coder.name} / {coder.model} …")

    def progress(i: int, n: int, size: int) -> None:
        print(f"  batch {i}: {min(i * size, n)}/{n} done")

    result = run(comments, codebook, coder, batch_size=args.batch_size,
                 cache_dir=Path(args.cache_dir), offline_ok=args.cached, progress=progress)
    ev = evaluate(result.comments, gold, codebook)
    themes = build_themes(result.comments, codebook)

    baseline = None
    if args.provider != "offline":
        base_run = run(comments, codebook, KeywordCoder(), batch_size=args.batch_size)
        baseline = evaluate(base_run.comments, gold, codebook)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result_to_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {"provider": result.provider, "model": result.model, **ev.to_dict()}
    if baseline:
        metrics["baseline"] = {k: v for k, v in baseline.to_dict().items() if k != "disagreements"}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    page = render(result, ev, themes, codebook, baseline)
    (out / "report.html").write_text(page, encoding="utf-8")
    if args.publish:
        Path(args.publish).parent.mkdir(parents=True, exist_ok=True)
        Path(args.publish).write_text(page, encoding="utf-8")

    review = sum(c.needs_review for c in result.comments)
    print(f"\nmicro-F1 {ev.micro_f1:.2f} · macro-F1 {ev.macro_f1:.2f} · mean kappa {ev.mean_kappa:.2f} · "
          f"{review} comments need review")
    if baseline:
        print(f"keyword baseline: micro-F1 {baseline.micro_f1:.2f} · mean kappa {baseline.mean_kappa:.2f}")
    print(f"API calls {result.stats.api_calls}, cache hits {result.stats.cache_hits}, retries {result.stats.retries}")
    print(f"Report: {out / 'report.html'}" + (f" and {args.publish}" if args.publish else ""))
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    print(system_prompt(load_codebook(args.codebook)))
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    p = argparse.ArgumentParser(prog="feedback_coder", description="LLM-assisted coding of user feedback")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="code all comments, evaluate and write the report")
    r.add_argument("--provider", choices=["offline", "openai", "anthropic"], default="offline")
    r.add_argument("--model", help=f"model name (defaults: {DEFAULT_MODELS})")
    r.add_argument("--batch-size", type=int, default=10)
    r.add_argument("--cached", action="store_true", help="use saved answers only; never call the API")
    r.add_argument("--data", default="data/feedback.csv")
    r.add_argument("--gold", default="data/gold_labels.csv")
    r.add_argument("--codebook", default="codebook.yaml")
    r.add_argument("--cache-dir", default="runs")
    r.add_argument("--out", default="output")
    r.add_argument("--publish", help="also write the report here, e.g. docs/index.html for GitHub Pages")
    r.set_defaults(func=cmd_run)

    pr = sub.add_parser("prompt", help="print the system prompt built from the codebook")
    pr.add_argument("--codebook", default="codebook.yaml")
    pr.set_defaults(func=cmd_prompt)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
