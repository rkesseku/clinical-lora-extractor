"""Batched generation, model loading and per-example prediction records."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .data import Example
from .parsing import ParseResult, parse_extraction
from .prompts import build_prompt


@dataclass
class Prediction:
    raw: str
    parse: ParseResult
    latency_s: float  # batch wall-clock / batch size (amortised)
    prompt_tokens: int
    new_tokens: int

    @property
    def entities(self) -> dict[str, list[str]] | None:
        return self.parse.extraction.as_dict() if self.parse.extraction else None


def load_base_model(
    name: str,
    load_in_4bit: bool = False,
    dtype: str | None = "auto",
    device_map: str | None = None,
):
    """Load a causal LM. ``dtype='auto'`` -> fp16 on CUDA, fp32 on CPU. 4-bit needs bitsandbytes."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs: dict = {}
    if load_in_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "load_in_4bit=True requires `pip install bitsandbytes` (Linux/CUDA)"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = device_map or {"": 0}
    else:
        if dtype == "auto":
            dtype = "float16" if torch.cuda.is_available() else "float32"
        if dtype:
            kwargs["dtype"] = getattr(torch, dtype)
        if device_map:
            kwargs["device_map"] = device_map
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except TypeError:  # older transformers used `torch_dtype`
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if not load_in_4bit and not device_map and torch.cuda.is_available():
        model.to("cuda")
    return model


def load_tokenizer(name: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_with_adapter(base_name: str, adapter_dir: str, merge: bool = False, **kw):
    """Load base weights + LoRA adapter (optionally merged into the base weights)."""
    from peft import PeftModel

    base = load_base_model(base_name, **kw)
    model = PeftModel.from_pretrained(base, adapter_dir)
    if merge:
        model = model.merge_and_unload()
    return model


class Generator:
    """Greedy (or sampled) batched generation with left padding."""

    def __init__(self, model, tokenizer, max_new_tokens: int = 256, batch_size: int = 8):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def generate(
        self, prompts: list[str], do_sample: bool = False, temperature: float = 0.7
    ) -> list[tuple[str, int, int, float]]:
        """Return ``(text, prompt_tokens, new_tokens, latency_s)`` per prompt."""
        tok = self.tokenizer
        results: list[tuple[str, int, int, float]] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
            enc = {k: v.to(self.device) for k, v in enc.items()}
            gen_kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tok.pad_token_id,
                "eos_token_id": tok.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = temperature
            else:
                gen_kwargs.update(temperature=None, top_p=None, top_k=None)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = self.model.generate(**enc, **gen_kwargs)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / len(chunk)
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                new_ids = out[j, plen:].tolist()
                n_new = len(new_ids)
                if tok.eos_token_id in new_ids:
                    n_new = new_ids.index(tok.eos_token_id) + 1
                text = tok.decode(new_ids[:n_new], skip_special_tokens=True)
                n_prompt = int(enc["attention_mask"][j].sum().item())
                results.append((text, n_prompt, n_new, dt))
        return results


def predict(
    generator: Generator, examples: list[Example], shots: list[Example] | None = None
) -> list[Prediction]:
    """Run the extraction prompt (zero/few-shot depending on ``shots``) over ``examples``."""
    prompts = [build_prompt(generator.tokenizer, e.text, shots) for e in examples]
    return [
        Prediction(raw=t, parse=parse_extraction(t), latency_s=dt, prompt_tokens=pt, new_tokens=nt)
        for (t, pt, nt, dt) in generator.generate(prompts)
    ]
