"""Training-budget ablation: epochs x training-set size, chosen on the validation split.

Every configuration is trained from a fresh copy of the base model (so a 2-epoch run is a real
2-epoch run with its own learning-rate schedule, not a checkpoint of a longer run) and scored on a
fixed sample of BC5CDR *validation* documents. The best configuration is picked on validation
strict F1 only. The test set is touched once at the end, for the selected configuration and for
the reference recipe (2,000 documents, 1 epoch), so the comparison is not tuned on test data.
"""

from __future__ import annotations

import gc
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .data import Example, load_bc5cdr, subsample
from .evaluate import SystemRun, build_results
from .inference import Generator, load_base_model, load_tokenizer, predict
from .train import TrainConfig, train_lora


@dataclass
class AblationConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    epochs: tuple[float, ...] = (1.0, 2.0, 3.0)
    n_train: tuple[int | None, ...] = (2000, None)  # None = the full training split
    reference: tuple[int | None, float] = (2000, 1.0)  # the recipe reported in the README
    n_val: int | None = 300
    n_test: int | None = 200
    seed: int = 0
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 1024
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    max_steps: int = -1  # > 0 only for smoke tests
    load_in_4bit: bool = False
    max_new_tokens: int = 256
    eval_batch_size: int = 8
    n_boot: int = 1000
    out_dir: str = "results_ablation"
    adapter_root: str = "adapters/ablation"


def config_name(n_train: int | None, epochs: float) -> str:
    size = "all" if n_train is None else str(n_train)
    return f"n={size}, epochs={epochs:g}"


def _slug(n_train: int | None, epochs: float) -> str:
    return f"n{'all' if n_train is None else n_train}_e{epochs:g}"


def _f1(summary: dict, mode: str) -> dict:
    return summary["entity"][mode]["overall"]["f1"]


def _release_memory() -> None:
    """Collect garbage and return cached GPU memory (call after dropping model references)."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def select_best(rows: list[dict]) -> dict:
    """Highest validation strict F1; ties go to the cheaper run (fewer docs x epochs)."""
    return max(rows, key=lambda r: (r["val_strict_f1"]["value"], -r["cost_units"]))


def run_ablation(
    cfg: AblationConfig,
    model_factory: Callable[[], object] | None = None,
    tokenizer=None,
    splits: dict[str, list[Example]] | None = None,
    load_adapter: Callable[[object, str], object] | None = None,
) -> dict:
    """Run the grid, pick the best configuration on validation, then score it once on test.

    ``model_factory`` returns a fresh base model each call; ``load_adapter(base, path)`` attaches a
    saved adapter. Both default to the real Hugging Face loaders and are injectable for tests.
    """
    splits = splits or load_bc5cdr()
    if "validation" not in splits:
        raise ValueError("the ablation needs a validation split to choose on")
    val = subsample(splits["validation"], cfg.n_val, cfg.seed)
    test = subsample(splits["test"], cfg.n_test, cfg.seed)
    tokenizer = tokenizer or load_tokenizer(cfg.base_model)
    if model_factory is None:

        def model_factory():
            return load_base_model(cfg.base_model, load_in_4bit=cfg.load_in_4bit)

    if load_adapter is None:

        def load_adapter(base, path):
            from peft import PeftModel

            return PeftModel.from_pretrained(base, path)

    grid = [(n, e) for n in cfg.n_train for e in cfg.epochs]
    n_candidates = len(grid)  # the reference is a candidate only if it is part of the grid
    if cfg.reference not in grid:
        grid.append(cfg.reference)
    rows: list[dict] = []
    for n_train, epochs in grid:
        train = subsample(splits["train"], n_train, cfg.seed)
        adapter_dir = str(Path(cfg.adapter_root) / _slug(n_train, epochs))
        tcfg = TrainConfig(
            output_dir=adapter_dir,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            learning_rate=cfg.learning_rate,
            num_train_epochs=epochs,
            max_steps=cfg.max_steps,
            per_device_train_batch_size=cfg.per_device_train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            max_length=cfg.max_length,
            seed=cfg.seed,
        )
        model = model_factory()
        peft_model, train_metrics = train_lora(
            model, tokenizer, train, tcfg, quantized=cfg.load_in_4bit
        )
        gen = Generator(peft_model, tokenizer, cfg.max_new_tokens, cfg.eval_batch_size)
        name = config_name(n_train, epochs)
        summary = build_results(
            [SystemRun(name, predict(gen, val))], val, n_boot=cfg.n_boot, seed=cfg.seed
        )["systems"][name]
        rows.append(
            {
                "name": name,
                "n_train": len(train),
                "n_train_requested": n_train,
                "epochs": epochs,
                "cost_units": len(train) * epochs,
                "adapter_dir": adapter_dir,
                "train_loss": float(train_metrics.get("train_loss", float("nan"))),
                "train_runtime_s": float(train_metrics.get("train_runtime", float("nan"))),
                "val_strict_f1": _f1(summary, "strict"),
                "val_relaxed_f1": _f1(summary, "relaxed"),
                "val_strict_json_rate": summary["validity"]["strict_json_rate"],
            }
        )
        print(f"[ablation] {name}: val strict F1 {rows[-1]['val_strict_f1']['value']:.3f}")
        del peft_model, model, gen
        _release_memory()

    best = select_best(rows[:n_candidates])
    ref = next(r for r in rows if (r["n_train_requested"], r["epochs"]) == cfg.reference)

    # One test evaluation each for the selected and the reference configuration.
    test_runs = []
    for row in [ref] if best is ref else [ref, best]:
        base = model_factory()
        model = load_adapter(base, row["adapter_dir"])
        gen = Generator(model, tokenizer, cfg.max_new_tokens, cfg.eval_batch_size)
        test_runs.append(SystemRun(row["name"], predict(gen, test)))
        del model, base, gen
        _release_memory()
    test_results = build_results(
        test_runs,
        test,
        meta={"base_model": cfg.base_model, "note": "selected on validation; test used once"},
        n_boot=cfg.n_boot,
        seed=cfg.seed,
        compare=(best["name"], ref["name"]) if best is not ref else None,
    )

    results = {
        "meta": {
            "base_model": cfg.base_model,
            "n_val": len(val),
            "n_test": len(test),
            "selection": "highest validation strict F1; ties -> fewer documents x epochs",
            "config": asdict(cfg),
        },
        "grid": rows,
        "selected": best["name"],
        "reference": ref["name"],
        "test": test_results,
    }
    write_ablation_outputs(results, cfg.out_dir)
    return results


def render_ablation_markdown(results: dict) -> str:
    m = results["meta"]
    lines = [
        "# Training-budget ablation",
        "",
        f"Base model `{m['base_model']}`. Each configuration was trained from the base model and "
        f"scored on {m['n_val']} validation documents; the selection rule was: {m['selection']}.",
        "",
        "| Configuration | Train docs | Epochs | Train loss | Train time (min) | "
        "Val strict F1 [95% CI] | Val relaxed F1 [95% CI] |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["grid"]:
        s, x = r["val_strict_f1"], r["val_relaxed_f1"]
        mark = " (selected)" if r["name"] == results["selected"] else ""
        mark += " (reference)" if r["name"] == results["reference"] else ""
        lines.append(
            f"| {r['name']}{mark} | {r['n_train']} | {r['epochs']:g} | {r['train_loss']:.3f} | "
            f"{r['train_runtime_s'] / 60:.1f} | {s['value']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}] | "
            f"{x['value']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}] |"
        )
    t = results["test"]
    lines += [
        "",
        f"## Test set ({m['n_test']} documents, evaluated once)",
        "",
        "| Configuration | Strict F1 [95% CI] | Relaxed F1 [95% CI] |",
        "|---|---|---|",
    ]
    for name, s in t["systems"].items():
        a, b = _f1(s, "strict"), _f1(s, "relaxed")
        lines.append(
            f"| {name} | {a['value']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | "
            f"{b['value']:.3f} [{b['lo']:.3f}, {b['hi']:.3f}] |"
        )
    if "paired_f1_difference" in t:
        d = t["paired_f1_difference"]
        lines += ["", f"Paired bootstrap, {d['a']} minus {d['b']}:"]
        for mode in ("strict", "relaxed"):
            x = d[mode]
            lines.append(
                f"- {mode}: {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}], "
                f"two-sided p = {x['p_two_sided']:.3f}"
            )
    else:
        lines += ["", "The reference recipe was also the best on validation."]
    return "\n".join(lines) + "\n"


def plot_ablation(results: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    sizes = sorted({r["n_train"] for r in results["grid"]})
    for n in sizes:
        rs = sorted((r for r in results["grid"] if r["n_train"] == n), key=lambda r: r["epochs"])
        x = [r["epochs"] for r in rs]
        y = [r["val_strict_f1"]["value"] for r in rs]
        err = [
            [v - r["val_strict_f1"]["lo"] for v, r in zip(y, rs, strict=True)],
            [r["val_strict_f1"]["hi"] - v for v, r in zip(y, rs, strict=True)],
        ]
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=f"{n} training docs")
    ax.set_xlabel("epochs")
    ax.set_ylabel("validation strict F1")
    ax.set_xticks(sorted({r["epochs"] for r in results["grid"]}))
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def write_ablation_outputs(results: dict, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out / "ablation.json",
        "report": out / "ablation.md",
        "plot": out / "ablation_val_f1.png",
    }
    paths["json"].write_text(json.dumps(results, indent=2), encoding="utf-8")
    paths["report"].write_text(render_ablation_markdown(results), encoding="utf-8")
    plot_ablation(results, paths["plot"])
    return paths
