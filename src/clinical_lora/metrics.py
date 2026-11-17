"""Entity-level metrics (strict / relaxed), micro-averaged, with percentile-bootstrap CIs.

Entities are compared as *strings* per document and type (BC5CDR mentions deduplicated per
document), since the model emits mention strings rather than character offsets.

* strict  : exact string match after whitespace normalisation (case-sensitive).
* relaxed : case/punctuation-insensitive; additionally a predicted mention matches a gold mention
            when the token set of one contains the token set of the other (partial boundary
            errors such as "hepatic necrosis" vs "necrosis" count). One-to-one greedy matching,
            exact matches first.

Invalid/unparseable model output is scored as an empty prediction (all gold entities become FN).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from .schema import ENTITY_TYPES

_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm_strict(s: str) -> str:
    return " ".join(s.split())


def norm_relaxed(s: str) -> str:
    return " ".join(_PUNCT.sub(" ", s.lower()).split())


def _dedupe(items: Sequence[str], norm) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        n = norm(it)
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out


def count_strict(gold: Sequence[str], pred: Sequence[str]) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` under strict matching."""
    g, p = set(_dedupe(gold, norm_strict)), set(_dedupe(pred, norm_strict))
    tp = len(g & p)
    return tp, len(p) - tp, len(g) - tp


def count_relaxed(gold: Sequence[str], pred: Sequence[str]) -> tuple[int, int, int]:
    """Return ``(tp, fp, fn)`` under relaxed matching (one-to-one, exact first)."""
    g = _dedupe(gold, norm_relaxed)
    p = _dedupe(pred, norm_relaxed)
    g_left, p_left = list(g), list(p)
    tp = 0
    for x in list(p_left):
        if x in g_left:
            g_left.remove(x)
            p_left.remove(x)
            tp += 1
    for x in list(p_left):
        xt = set(x.split())
        for y in g_left:
            yt = set(y.split())
            if xt <= yt or yt <= xt:
                g_left.remove(y)
                p_left.remove(x)
                tp += 1
                break
    return tp, len(p) - tp, len(g) - tp


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def doc_counts(
    golds: Sequence[dict[str, list[str]]],
    preds: Sequence[dict[str, list[str]] | None],
    mode: str,
) -> dict[str, np.ndarray]:
    """Per-document ``(tp, fp, fn)`` arrays, shape ``(n_docs, 3)``, per type plus ``overall``."""
    if len(golds) != len(preds):
        raise ValueError("golds and preds must have the same length")
    fn_count = {"strict": count_strict, "relaxed": count_relaxed}[mode]
    arr = {t: np.zeros((len(golds), 3), dtype=np.int64) for t in ENTITY_TYPES}
    for i, (g, p) in enumerate(zip(golds, preds, strict=True)):
        p = p or {}
        for t in ENTITY_TYPES:
            arr[t][i] = fn_count(g.get(t, []), p.get(t, []))
    arr["overall"] = sum(arr[t] for t in ENTITY_TYPES)
    return arr


def _micro(counts: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn = counts.sum(axis=0)
    return prf(float(tp), float(fp), float(fn))


def _batched_prf(sums: np.ndarray) -> np.ndarray:
    """Vectorised P/R/F1 for an array of summed counts, shape ``(B, 3)`` -> ``(B, 3)``."""
    tp, fp, fn = sums[:, 0].astype(float), sums[:, 1].astype(float), sums[:, 2].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        r = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f = np.where(p + r > 0, 2 * p * r / (p + r), 0.0)
    return np.stack([p, r, f], axis=1)


def bootstrap_prf(
    counts: np.ndarray, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> dict[str, dict[str, float]]:
    """Micro P/R/F1 with a percentile bootstrap CI resampling *documents* with replacement."""
    n = len(counts)
    p, r, f = _micro(counts)
    out = {
        "precision": {"value": p},
        "recall": {"value": r},
        "f1": {"value": f},
    }
    if n == 0:
        for k in out:
            out[k].update(lo=float("nan"), hi=float("nan"))
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = _batched_prf(counts[idx].sum(axis=1))
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2], axis=0)
    for j, k in enumerate(("precision", "recall", "f1")):
        out[k].update(lo=float(lo[j]), hi=float(hi[j]))
    return out


def paired_bootstrap_f1_diff(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Paired bootstrap of ``F1(a) - F1(b)`` over documents (same test set for both systems)."""
    if counts_a.shape != counts_b.shape:
        raise ValueError("count arrays must have identical shape")
    n = len(counts_a)
    diff = _micro(counts_a)[2] - _micro(counts_b)[2]
    if n == 0:
        return {"diff": diff, "lo": float("nan"), "hi": float("nan"), "p_two_sided": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = (
        _batched_prf(counts_a[idx].sum(axis=1))[:, 2]
        - _batched_prf(counts_b[idx].sum(axis=1))[:, 2]
    )
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    # Bootstrap sign-flip style p-value: share of resamples on the other side of zero, doubled.
    p = min(1.0, 2 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi), "p_two_sided": p}


def evaluate_entities(
    golds: Sequence[dict[str, list[str]]],
    preds: Sequence[dict[str, list[str]] | None],
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """``{mode: {type|overall: {precision|recall|f1: {value, lo, hi}}}}`` for strict and relaxed."""
    result: dict = {}
    for mode in ("strict", "relaxed"):
        counts = doc_counts(golds, preds, mode)
        result[mode] = {t: bootstrap_prf(c, n_boot=n_boot, seed=seed) for t, c in counts.items()}
    return result
