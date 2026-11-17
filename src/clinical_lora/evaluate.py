"""Evaluation + reporting: metrics.json, Markdown report and plots for any number of systems.

Every system is scored on the *same* held-out test examples. Reporting ideas (per-system summary
table, explicit validity/cost/latency columns, machine-readable JSON next to a human report) follow
the sibling ``llm-eval-harness`` project.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import Example
from .inference import Prediction
from .metrics import doc_counts, evaluate_entities, paired_bootstrap_f1_diff
from .schema import ENTITY_TYPES

SECTIONS = (*ENTITY_TYPES, "overall")


@dataclass
class SystemRun:
    name: str
    predictions: list[Prediction]
    price_per_mtok: tuple[float, float] | None = None  # (input, output) USD per 1M tokens
    note: str = ""
    extra: dict = field(default_factory=dict)


def _lat_stats(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {"mean": math.nan, "p50": math.nan, "p95": math.nan}
    a = np.asarray(xs)
    return {"mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.quantile(a, 0.95))}


def summarize_system(
    run: SystemRun, golds: list[dict[str, list[str]]], n_boot: int = 1000, seed: int = 0
) -> dict:
    preds = run.predictions
    if len(preds) != len(golds):
        raise ValueError(f"{run.name}: {len(preds)} predictions for {len(golds)} gold examples")
    n = len(preds)
    entities = [p.entities for p in preds]  # None for unparseable -> scored as empty
    out = {
        "n": n,
        "note": run.note,
        "validity": {
            "strict_json_rate": sum(p.parse.valid for p in preds) / n if n else math.nan,
            "repaired_rate": sum(p.parse.repaired for p in preds) / n if n else math.nan,
            "parsed_rate": sum(p.parse.parsed for p in preds) / n if n else math.nan,
        },
        "entity": evaluate_entities(golds, entities, n_boot=n_boot, seed=seed),
        "latency_s": _lat_stats([p.latency_s for p in preds]),
        "tokens": {
            "prompt_mean": float(np.mean([p.prompt_tokens for p in preds])) if n else math.nan,
            "new_mean": float(np.mean([p.new_tokens for p in preds])) if n else math.nan,
            "prompt_total": int(sum(p.prompt_tokens for p in preds)),
            "new_total": int(sum(p.new_tokens for p in preds)),
        },
        "cost_usd": None,
        **run.extra,
    }
    if run.price_per_mtok is not None:
        pin, pout = run.price_per_mtok
        total = (out["tokens"]["prompt_total"] * pin + out["tokens"]["new_total"] * pout) / 1e6
        out["cost_usd"] = {
            "total": total,
            "per_1k_examples": total / n * 1000 if n else math.nan,
            "price_per_mtok_in_out": [pin, pout],
        }
    return out


def build_results(
    runs: list[SystemRun],
    test: list[Example],
    meta: dict | None = None,
    n_boot: int = 1000,
    seed: int = 0,
    compare: tuple[str, str] | None = None,
) -> dict:
    """Score all systems on ``test`` and return the full metrics dict (JSON-serialisable)."""
    golds = [e.entities for e in test]
    systems = {r.name: summarize_system(r, golds, n_boot, seed) for r in runs}
    results = {
        "meta": {"n_test": len(test), "n_boot": n_boot, "seed": seed, **(meta or {})},
        "systems": systems,
    }
    if compare is None and len(runs) >= 2:
        compare = (runs[-1].name, runs[0].name)
    if compare and all(c in systems for c in compare):
        a, b = (next(r for r in runs if r.name == c) for c in compare)
        paired = {}
        for mode in ("strict", "relaxed"):
            ca = doc_counts(golds, [p.entities for p in a.predictions], mode)["overall"]
            cb = doc_counts(golds, [p.entities for p in b.predictions], mode)["overall"]
            paired[mode] = paired_bootstrap_f1_diff(ca, cb, n_boot=n_boot, seed=seed)
        results["paired_f1_difference"] = {"a": compare[0], "b": compare[1], **paired}
    return results


def _f(x: dict, key: str = "f1") -> str:
    d = x[key]
    return f"{d['value']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def render_markdown(results: dict) -> str:
    m = results["meta"]
    lines = ["# Evaluation report", ""]
    lines.append(
        f"Held-out test set: **{m['n_test']}** documents (identical for every system). "
        f"95% percentile-bootstrap CIs over documents ({m['n_boot']} resamples)."
    )
    for k in ("base_model", "dataset", "n_train", "lora"):
        if k in m:
            lines.append(f"- {k}: `{m[k]}`")
    lines += ["", "## Entity-level F1 (micro) with 95% CI", ""]
    for mode in ("strict", "relaxed"):
        lines += [
            f"### {mode.capitalize()} match",
            "",
            "| System | Chemicals F1 | Diseases F1 | Overall P | Overall R | Overall F1 |",
            "|---|---|---|---|---|---|",
        ]
        for name, s in results["systems"].items():
            e = s["entity"][mode]
            lines.append(
                f"| {name} | {_f(e['chemicals'])} | {_f(e['diseases'])} | "
                f"{e['overall']['precision']['value']:.3f} | "
                f"{e['overall']['recall']['value']:.3f} | "
                f"{_f(e['overall'])} |"
            )
        lines.append("")
    lines += [
        "## Output validity, latency, tokens, cost",
        "",
        "| System | Strict-JSON rate | Parsed after repair | Latency mean / p95 (s) | "
        "Prompt tok | New tok | Cost / 1k ex (USD) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in results["systems"].items():
        v, lat, tk, c = s["validity"], s["latency_s"], s["tokens"], s["cost_usd"]
        cost = f"{c['per_1k_examples']:.4f}" if c else "n/a"
        lines.append(
            f"| {name} | {v['strict_json_rate']:.3f} | {v['parsed_rate']:.3f} | "
            f"{lat['mean']:.3f} / {lat['p95']:.3f} | {tk['prompt_mean']:.0f} | "
            f"{tk['new_mean']:.0f} | {cost} |"
        )
    pd_ = results.get("paired_f1_difference")
    if pd_:
        lines += ["", f"## Paired bootstrap: overall F1 of `{pd_['a']}` minus `{pd_['b']}`", ""]
        for mode in ("strict", "relaxed"):
            d = pd_[mode]
            lines.append(
                f"- {mode}: diff = {d['diff']:+.3f}, 95% CI [{d['lo']:+.3f}, {d['hi']:+.3f}], "
                f"two-sided bootstrap p = {d['p_two_sided']:.3f}"
            )
    lines += [
        "",
        "## Notes",
        "",
        "- Invalid/unparseable outputs are scored as empty predictions (all gold entities missed).",
        "- Latency is wall-clock generation time divided by batch size (amortised); "
        "compare systems only when run on the same hardware and batch size.",
        "- Local models have no per-token cost; `n/a` means no price was supplied.",
    ]
    return "\n".join(lines) + "\n"


def plot_results(results: dict, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(results["systems"])
    paths: list[Path] = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    width = 0.8 / max(1, len(names))
    for ax, mode in zip(axes, ("strict", "relaxed"), strict=True):
        for i, name in enumerate(names):
            e = results["systems"][name]["entity"][mode]
            vals = [e[t]["f1"]["value"] for t in SECTIONS]
            err = [
                [e[t]["f1"]["value"] - e[t]["f1"]["lo"] for t in SECTIONS],
                [e[t]["f1"]["hi"] - e[t]["f1"]["value"] for t in SECTIONS],
            ]
            xs = np.arange(len(SECTIONS)) + i * width - 0.4 + width / 2
            ax.bar(xs, vals, width, yerr=err, capsize=3, label=name)
        ax.set_xticks(range(len(SECTIONS)), SECTIONS)
        ax.set_title(f"{mode.capitalize()} F1 (95% bootstrap CI)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("F1")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "f1_by_type.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    lat = [results["systems"][n]["latency_s"]["mean"] for n in names]
    val = [results["systems"][n]["validity"]["strict_json_rate"] for n in names]
    axes[0].bar(names, lat)
    axes[0].set_ylabel("mean latency / example (s)")
    axes[1].bar(names, val)
    axes[1].set_ylabel("strict JSON validity rate")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    p = out_dir / "latency_validity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def write_outputs(results: dict, out_dir: str | Path = "results") -> dict[str, Path]:
    """Write ``metrics.json``, ``report.md`` and plots into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mpath, rpath = out / "metrics.json", out / "report.md"
    mpath.write_text(json.dumps(results, indent=2, allow_nan=True), encoding="utf-8")
    rpath.write_text(render_markdown(results), encoding="utf-8")
    plots = plot_results(results, out)
    return {"metrics": mpath, "report": rpath, **{p.stem: p for p in plots}}


def save_predictions(runs: list[SystemRun], test: list[Example], out_dir: str | Path) -> Path:
    """Dump per-example raw outputs for error analysis (JSON Lines)."""
    path = Path(out_dir) / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in runs:
            for ex, p in zip(test, r.predictions, strict=True):
                fh.write(
                    json.dumps(
                        {
                            "system": r.name,
                            "text": ex.text,
                            "gold": ex.entities,
                            "raw": p.raw,
                            "pred": p.entities,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return path
