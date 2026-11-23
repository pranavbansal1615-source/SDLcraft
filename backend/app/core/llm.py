from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI


_client: Optional[OpenAI] = None


def get_llm_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1"),
            api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        )
    return _client


def invoke_llm(prompt: str, max_tokens: int = 4000) -> str:
    response = get_llm_client().chat.completions.create(
        model=os.getenv("LLM_MODEL", "local-model"),
        messages=[{"role": "user", "content": prompt}],
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        max_tokens=max_tokens,
    )
    content = response.choices[0].message.content
    return content.strip() if content else ""
