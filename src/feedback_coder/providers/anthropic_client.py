"""Chat client for the Anthropic Messages API (Claude), plain HTTP, no SDK."""

from __future__ import annotations

from .base import post_json

URL = "https://api.anthropic.com/v1/messages"


class AnthropicClient:
    name = "anthropic"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": 0,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"}
        data = post_json(URL, headers, body)
        # The answer is a list of content blocks; join the text blocks.
        return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
