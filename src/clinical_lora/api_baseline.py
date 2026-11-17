"""Optional hosted-LLM baseline via the Anthropic Messages API (plain HTTPS, no extra SDK).

Enabled only when ``ANTHROPIC_API_KEY`` is set. The model id and the per-million-token prices are
configuration, not defaults baked into results: pass the prices you are billed at to get a cost
estimate (otherwise cost is reported as unavailable).
"""

from __future__ import annotations

import os
import time

import httpx

from .data import Example
from .inference import Prediction
from .parsing import parse_extraction
from .prompts import build_messages

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-haiku-4-5"


def api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class AnthropicExtractor:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 512,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ):
        self.retry_backoff_s = retry_backoff_s
        self.model = model or os.environ.get("CLE_API_MODEL", DEFAULT_MODEL)
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key and client is None:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.max_tokens = max_tokens
        self.client = client or httpx.Client(timeout=timeout)
        self.max_retries = max_retries

    def _call(self, text: str, shots: list[Example] | None) -> tuple[str, int, int, float]:
        msgs = build_messages(text, shots)
        system, chat = msgs[0]["content"], msgs[1:]
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": chat,
            "temperature": 0,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = time.perf_counter()
            try:
                r = self.client.post(API_URL, json=body, headers=headers)
                if r.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                dt = time.perf_counter() - t0
                data = r.json()
                out = "".join(b.get("text", "") for b in data.get("content", []))
                usage = data.get("usage", {})
                return (
                    out,
                    int(usage.get("input_tokens", 0)),
                    int(usage.get("output_tokens", 0)),
                    dt,
                )
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_backoff_s * 2**attempt)
        raise RuntimeError(f"API call failed after {self.max_retries} attempts: {last}")

    def predict(
        self, examples: list[Example], shots: list[Example] | None = None
    ) -> list[Prediction]:
        preds = []
        for ex in examples:
            text, pt, nt, dt = self._call(ex.text, shots)
            preds.append(Prediction(text, parse_extraction(text), dt, pt, nt))
        return preds
