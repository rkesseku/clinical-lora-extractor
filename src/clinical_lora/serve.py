"""FastAPI service: returns schema-validated JSON, with retry/repair on invalid model output.

Run (after training):

    CLE_BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct CLE_ADAPTER=adapters/clinical-lora \
        uvicorn clinical_lora.serve:create_app_from_env --factory --port 8000
"""

from __future__ import annotations

import os
from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .parsing import parse_extraction
from .prompts import build_prompt
from .schema import Extraction


class ExtractRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


class ExtractResponse(BaseModel):
    extraction: Extraction
    valid_json: bool = Field(description="True if the first attempt was strict valid JSON")
    repaired: bool = Field(description="True if the answer needed the repair heuristics")
    attempts: int


class TextGenerator(Protocol):
    def __call__(self, prompt: str, attempt: int) -> str:
        """Return raw model text for ``prompt``; ``attempt`` starts at 0 (retries may sample)."""


class Extractor:
    """Wraps a text generator with parse -> repair -> retry logic."""

    def __init__(self, generate: TextGenerator, tokenizer=None, max_attempts: int = 3):
        self.generate = generate
        self.tokenizer = tokenizer
        self.max_attempts = max_attempts

    def extract(self, text: str) -> ExtractResponse:
        prompt = build_prompt(self.tokenizer, text)
        best_repaired = None
        for attempt in range(self.max_attempts):
            res = parse_extraction(self.generate(prompt, attempt))
            if res.valid:
                return ExtractResponse(
                    extraction=res.extraction,
                    valid_json=attempt == 0,
                    repaired=False,
                    attempts=attempt + 1,
                )
            if res.repaired and best_repaired is None:
                best_repaired = (res.extraction, attempt + 1)
        if best_repaired is not None:
            return ExtractResponse(
                extraction=best_repaired[0],
                valid_json=False,
                repaired=True,
                attempts=self.max_attempts,
            )
        raise ValueError(f"model produced no parseable JSON after {self.max_attempts} attempts")


def hf_text_generator(model, tokenizer, max_new_tokens: int = 256) -> TextGenerator:
    """First attempt greedy; retries sample at temperature 0.7."""
    from .inference import Generator

    gen = Generator(model, tokenizer, max_new_tokens=max_new_tokens, batch_size=1)

    def _call(prompt: str, attempt: int) -> str:
        return gen.generate([prompt], do_sample=attempt > 0)[0][0]

    return _call


def create_app(extractor: Extractor, model_info: dict | None = None) -> FastAPI:
    app = FastAPI(title="clinical-lora-extractor", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", **(model_info or {})}

    @app.post("/extract", response_model=ExtractResponse)
    def extract(req: ExtractRequest) -> ExtractResponse:
        try:
            return extractor.extract(req.text)
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def create_app_from_env() -> FastAPI:
    """Build the app from CLE_BASE_MODEL / CLE_ADAPTER / CLE_MERGE env vars (loads weights)."""
    from .inference import load_base_model, load_tokenizer, load_with_adapter

    base = os.environ.get("CLE_BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    adapter = os.environ.get("CLE_ADAPTER")
    merge = os.environ.get("CLE_MERGE", "0") == "1"
    tokenizer = load_tokenizer(base)
    model = load_with_adapter(base, adapter, merge=merge) if adapter else load_base_model(base)
    extractor = Extractor(hf_text_generator(model, tokenizer), tokenizer)
    return create_app(extractor, {"base_model": base, "adapter": adapter, "merged": merge})
