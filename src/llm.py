"""Shared OpenAI client helpers."""

from __future__ import annotations

from openai import OpenAI, RateLimitError, APIConnectionError, APITimeoutError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.config import openai_api_key

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMS = 1536
CHAT_MODEL = "gpt-4o-mini"


def get_client() -> OpenAI:
    key = openai_api_key()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")
    return OpenAI(api_key=key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    wait=wait_exponential_jitter(initial=2, max=60),
    stop=stop_after_attempt(12),
    reraise=True,
)
def embed_texts(texts: list[str], *, client: OpenAI | None = None) -> list[list[float]]:
    """Embed a batch of texts. Retries on rate limits / transient errors."""
    if not texts:
        return []
    client = client or get_client()
    cleaned = [t if t and t.strip() else " " for t in texts]
    resp = client.embeddings.create(model=EMBED_MODEL, input=cleaned)
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]


def chat_json(system: str, user: str, *, temperature: float = 0.0) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or "{}"


def chat_text(system: str, user: str, *, temperature: float = 0.2) -> str:
    client = get_client()
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
