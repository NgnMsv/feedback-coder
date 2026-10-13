"""The exact JSON shape every coder (LLM or offline) must return.

LLMs return text. Text can be malformed, missing fields, or have a
confidence of 1.7. Pydantic turns "hopefully the right JSON" into either a
typed Python object or a clear error message – and that error message is
sent back to the model when we retry (see pipeline.py).

Example of a valid answer for one batch:

    {"results": [
      {"id": "F001",
       "codes": [{"code": "bug_crash",
                  "evidence": "The app crashes every time I open the reports tab",
                  "confidence": 0.95}],
       "sentiment": "negative"}
    ]}
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "negative", "mixed", "neutral"]


class CodeAssignment(BaseModel):
    code: str
    # Words copied exactly from the comment that justify the code.
    evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


class CodedComment(BaseModel):
    id: str
    codes: list[CodeAssignment] = Field(default_factory=list)
    sentiment: Sentiment


class BatchResult(BaseModel):
    results: list[CodedComment]
