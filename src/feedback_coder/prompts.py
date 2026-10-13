"""Step 3 – build the prompt.

Two messages are sent to the model:

* the SYSTEM message: the role, the codebook and the rules. It is the same
  for every batch.
* the USER message: the comments of one batch, as JSON.

Design choices worth explaining in an interview:

1. The codebook is pasted in with definitions, include/exclude rules and
   examples. A label like "usability" alone is ambiguous; the rules are what
   make the model code like a trained human coder.
2. The model must quote EVIDENCE copied from the comment. verify.py then
   checks that the quote really appears in the comment. A quote that does not
   exist is the clearest sign of a hallucination.
3. Comments go in as JSON with ids, and come back with the same ids, so
   answers can never be matched to the wrong comment.
4. Output is JSON only, validated by schema.py.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from .codebook import Codebook
from .data import Comment


def system_prompt(codebook: Codebook) -> str:
    lines = [
        "You are an experienced qualitative researcher coding user feedback for a budgeting app.",
        "Assign codes from the codebook below to each comment. A comment can have several codes, or none.",
        "",
        "CODEBOOK",
    ]
    for c in codebook.codes:
        lines.append(f"\n## {c.id} — {c.label}")
        lines.append(f"Definition: {c.definition}")
        if c.include:
            lines.append("Include: " + "; ".join(c.include))
        if c.exclude:
            lines.append("Exclude: " + "; ".join(c.exclude))
        if c.examples:
            lines.append("Examples: " + " | ".join(c.examples))

    lines += [
        "",
        "RULES",
        f"- Use only these code ids: {', '.join(codebook.ids)}.",
        "- For every code, copy the shortest phrase from the comment that justifies it into \"evidence\".",
        "  Copy it exactly, character for character. Do not paraphrase or translate.",
        "- \"confidence\" is a number from 0 to 1: how sure you are that the code applies.",
        "- If no code applies, return an empty \"codes\" list.",
        f"- \"sentiment\" is one of: {', '.join(codebook.sentiments)} (mixed = clearly positive AND negative).",
        "- Return one result for every comment id you receive, and no other ids.",
        "",
        "Answer with JSON only, in exactly this shape:",
        '{"results": [{"id": "...", "codes": [{"code": "...", "evidence": "...", "confidence": 0.0}], "sentiment": "..."}]}',
    ]
    return "\n".join(lines)


def user_prompt(comments: Sequence[Comment], error_hint: str | None = None) -> str:
    payload = [{"id": c.id, "text": c.text} for c in comments]
    msg = "Code these comments:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    if error_hint:
        # Used on a retry: tell the model exactly what was wrong last time.
        msg += (
            "\n\nYour previous answer could not be used because of this error:\n"
            f"{error_hint}\nReturn corrected JSON only."
        )
    return msg
