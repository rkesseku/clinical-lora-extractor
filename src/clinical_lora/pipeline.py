"""End-to-end experiment: few-shot baseline, LoRA fine-tune, (optional) API baseline, report.

``run_experiment`` accepts pre-built ``model``/``tokenizer`` and ``splits`` so the same code path
is exercised in the CPU smoke test with a tiny random model and synthetic data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .data import Example, load_bc5cdr, subsample
from .evaluate import SystemRun, build_results, save_predictions, write_outputs
from .inference import Generator, load_base_model, load_tokenizer, predict
from .prompts import select_shots
from .train import TrainConfig, train_lora


@dataclass
class ExperimentConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    n_train: int | None = 2000  # ~25-30 min on a free T4 at 1 epoch with the defaults
    n_test: int | None = 200
    n_shots: int = 3
    seed: int = 0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = False
    max_new_tokens: int = 256
    eval_batch_size: int = 8
    n_boot: int = 1000
    use_api: bool = False
    api_model: str | None = None
    api_price_in: float | None = None
    api_price_out: float | None = None
    out_dir: str = "results"
    adapter_dir: str = "adapters/clinical-lora"


def run_experiment(
    cfg: ExperimentConfig,
    model=None,
    tokenizer=None,
    splits: dict[str, list[Example]] | None = None,
) -> dict:
    splits = splits or load_bc5cdr()
    train = subsample(splits["train"], cfg.n_train, cfg.seed)
    test = subsample(splits["test"], cfg.n_test, cfg.seed)  # same test set for every system
    shots = select_shots(splits["train"], cfg.n_shots)

    tokenizer = tokenizer or load_tokenizer(cfg.base_model)
    model = model or load_base_model(cfg.base_model, load_in_4bit=cfg.load_in_4bit)
    runs: list[SystemRun] = []

    # 1) few-shot prompting of the *base* model (before any LoRA weights are attached)
    gen = Generator(model, tokenizer, cfg.max_new_tokens, cfg.eval_batch_size)
    runs.append(SystemRun(f"base few-shot (k={cfg.n_shots})", predict(gen, test, shots)))

    # 2) optional hosted API few-shot baseline
    if cfg.use_api:
        from .api_baseline import AnthropicExtractor

        api = AnthropicExtractor(model=cfg.api_model)
        price = (
            (cfg.api_price_in, cfg.api_price_out)
            if cfg.api_price_in is not None and cfg.api_price_out is not None
            else None
        )
        runs.append(
            SystemRun(
                f"API few-shot ({api.model})",
                api.predict(test, shots),
                price,
                note="hosted model; latency includes network",
            )
        )

    # 3) LoRA fine-tune (zero-shot at test time: the task is baked into the adapter)
    tcfg = TrainConfig(
        output_dir=cfg.adapter_dir,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_length=cfg.max_length,
        seed=cfg.seed,
    )
    peft_model, train_metrics = train_lora(
        model, tokenizer, train, tcfg, quantized=cfg.load_in_4bit
    )
    gen = Generator(peft_model, tokenizer, cfg.max_new_tokens, cfg.eval_batch_size)
    runs.append(
        SystemRun(
            "LoRA fine-tuned (0-shot)",
            predict(gen, test),
            extra={
                "train_metrics": {
                    k: float(v) for k, v in train_metrics.items() if isinstance(v, int | float)
                }
            },
        )
    )

    meta = {
        "base_model": cfg.base_model,
        "dataset": "BC5CDR (biomedical literature; proxy for clinical text, no PHI)",
        "n_train": len(train),
        "lora": {"r": cfg.lora_r, "alpha": cfg.lora_alpha, "4bit": cfg.load_in_4bit},
        "config": asdict(cfg),
    }
    results = build_results(runs, test, meta=meta, n_boot=cfg.n_boot, seed=cfg.seed)
    write_outputs(results, cfg.out_dir)
    save_predictions(runs, test, cfg.out_dir)
    return results
