"""Step 4 – the "coders": things that turn a batch of comments into JSON text.

Every coder has the same method:

    code_batch(comments, codebook, error_hint=None) -> str   (raw JSON text)

so the pipeline does not care whether the answer came from OpenAI, Claude or
the offline keyword baseline.
"""

from __future__ import annotations

import os

from .base import Coder, LLMCoder
from .offline import KeywordCoder

DEFAULT_MODELS = {
    # Check the providers' model lists for the current names; any chat model works.
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-5-5",
}


def make_coder(provider: str, model: str | None = None, *, require_key: bool = True) -> Coder:
    """Create the coder for a provider name used on the command line."""
    if provider == "offline":
        return KeywordCoder()
    if provider == "openai":
        from .openai_client import OpenAIClient

        key = os.environ.get("OPENAI_API_KEY", "")
        client = OpenAIClient(api_key=key, model=model or DEFAULT_MODELS["openai"])
    elif provider == "anthropic":
        from .anthropic_client import AnthropicClient

        key = os.environ.get("ANTHROPIC_API_KEY", "")
        client = AnthropicClient(api_key=key, model=model or DEFAULT_MODELS["anthropic"])
    else:
        raise ValueError(f"unknown provider: {provider!r} (use offline, openai or anthropic)")
    if require_key and not key:
        raise RuntimeError(
            f"No API key for {provider}. Set {'OPENAI_API_KEY' if provider == 'openai' else 'ANTHROPIC_API_KEY'} "
            "in your environment or .env file – or run with a cached run (see README)."
        )
    return LLMCoder(client)


__all__ = ["DEFAULT_MODELS", "Coder", "KeywordCoder", "LLMCoder", "make_coder"]
