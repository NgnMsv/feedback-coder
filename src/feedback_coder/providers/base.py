"""The coder interface and the shared LLM coder.

A *chat client* knows how to talk to one API (OpenAI or Anthropic).
The *LLMCoder* knows how to code feedback: it builds the prompt and asks a
chat client for an answer. Splitting the two keeps each file small, and lets
the tests replace the network with a fake client.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Protocol

from ..codebook import Codebook
from ..data import Comment
from ..prompts import system_prompt, user_prompt


class Coder(Protocol):
    name: str
    model: str

    def code_batch(self, comments: Sequence[Comment], codebook: Codebook, error_hint: str | None = None) -> str: ...


class ChatClient(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str) -> str: ...


class LLMCoder:
    def __init__(self, client: ChatClient):
        self.client = client
        self.name = client.name
        self.model = client.model

    def code_batch(self, comments: Sequence[Comment], codebook: Codebook, error_hint: str | None = None) -> str:
        return self.client.complete(system_prompt(codebook), user_prompt(comments, error_hint))


def post_json(url: str, headers: dict[str, str], body: dict, timeout: float = 60.0, retries: int = 3) -> dict:
    """POST a JSON body and return the JSON response.

    Uses only the standard library, so you can see the whole HTTP request:
    a URL, some headers (including the API key) and a JSON body.
    Rate limits (429) and server errors (5xx) are retried with a growing pause.
    """
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method="POST", headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if (e.code == 429 or e.code >= 500) and attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"API error {e.code}: {detail}") from e
    raise RuntimeError("unreachable")
