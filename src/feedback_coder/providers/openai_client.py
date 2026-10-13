"""Chat client for the OpenAI Chat Completions API (plain HTTP, no SDK)."""

from __future__ import annotations

from .base import post_json

URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient:
    name = "openai"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def complete(self, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "temperature": 0,  # as repeatable as possible
            "response_format": {"type": "json_object"},  # forces syntactically valid JSON
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        data = post_json(URL, {"authorization": f"Bearer {self.api_key}"}, body)
        return data["choices"][0]["message"]["content"]
