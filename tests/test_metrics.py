import numpy as np
import pytest

from clinical_lora.metrics import (
    bootstrap_prf,
    count_relaxed,
    count_strict,
    doc_counts,
    evaluate_entities,
    paired_bootstrap_f1_diff,
    prf,
)


def test_count_strict_known_cases():
    assert count_strict(["a", "b"], ["a", "b"]) == (2, 0, 0)
    assert count_strict(["a", "b"], ["a", "c"]) == (1, 1, 1)
    assert count_strict(["Aspirin"], ["aspirin"]) == (0, 1, 1)  # case-sensitive
    assert count_strict([], []) == (0, 0, 0)
    assert count_strict(["a", "a"], ["a", "a", "a"]) == (1, 0, 0)  # deduplicated
    assert count_strict(["a  b"], ["a b"]) == (1, 0, 0)  # whitespace-normalised


def test_count_relaxed_known_cases():
    assert count_relaxed(["Aspirin"], ["aspirin"]) == (1, 0, 0)
    assert count_relaxed(["hepatic necrosis"], ["necrosis"]) == (1, 0, 0)  # containment
    assert count_relaxed(["necrosis"], ["hepatic necrosis"]) == (1, 0, 0)
    assert count_relaxed(["a-b"], ["a b"]) == (1, 0, 0)  # punctuation-insensitive
    assert count_relaxed(["liver", "kidney"], ["liver kidney"]) == (1, 0, 1)  # one-to-one
    assert count_relaxed(["x"], ["y"]) == (0, 1, 1)


def test_prf():
    assert prf(2, 2, 2) == pytest.approx((0.5, 0.5, 0.5))
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(3, 1, 0)
    assert (p, r) == (0.75, 1.0) and f == pytest.approx(2 * 0.75 / 1.75)


GOLD = [
    {"chemicals": ["a", "b"], "diseases": ["x"]},
    {"chemicals": ["c"], "diseases": []},
    {"chemicals": [], "diseases": ["y", "z"]},
]
PRED = [
    {"chemicals": ["a"], "diseases": ["x", "w"]},  # chem 1tp 0fp 1fn; dis 1tp 1fp 0fn
    None,  # unparseable -> chem 0/0/1; dis nothing
    {"chemicals": ["q"], "diseases": ["y", "z"]},  # chem 0tp 1fp; dis 2tp
]


def test_doc_counts_and_micro_values():
    c = doc_counts(GOLD, PRED, "strict")
    assert c["chemicals"].tolist() == [[1, 0, 1], [0, 0, 1], [0, 1, 0]]
    assert c["diseases"].tolist() == [[1, 1, 0], [0, 0, 0], [2, 0, 0]]
    ov = evaluate_entities(GOLD, PRED, n_boot=50)["strict"]["overall"]
    # tp=4, fp=2, fn=2
    assert ov["precision"]["value"] == pytest.approx(4 / 6)
    assert ov["recall"]["value"] == pytest.approx(4 / 6)
    assert ov["f1"]["value"] == pytest.approx(4 / 6)


def test_bootstrap_ci_contains_point_and_is_deterministic():
    rng = np.random.default_rng(1)
    counts = np.stack([rng.integers(0, 4, 60), rng.integers(0, 2, 60), rng.integers(0, 2, 60)], 1)
    a = bootstrap_prf(counts, n_boot=500, seed=3)
    b = bootstrap_prf(counts, n_boot=500, seed=3)
    assert a == b
    f = a["f1"]
    assert f["lo"] <= f["value"] <= f["hi"] and f["hi"] - f["lo"] > 0


def test_bootstrap_degenerate_perfect_has_zero_width():
    counts = np.tile([2, 0, 0], (20, 1))
    f = bootstrap_prf(counts, n_boot=100)["f1"]
    assert f["value"] == f["lo"] == f["hi"] == 1.0


def test_bootstrap_interval_narrows_with_more_documents():
    base = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0], [1, 0, 1]])
    small = bootstrap_prf(np.tile(base, (5, 1)), n_boot=400)["f1"]
    large = bootstrap_prf(np.tile(base, (50, 1)), n_boot=400)["f1"]
    assert (large["hi"] - large["lo"]) < (small["hi"] - small["lo"])


def test_paired_difference_detects_clear_winner():
    better = np.tile([3, 0, 0], (40, 1))
    worse = np.tile([1, 2, 2], (40, 1))
    d = paired_bootstrap_f1_diff(better, worse, n_boot=200)
    assert d["diff"] > 0.4 and d["lo"] > 0 and d["p_two_sided"] < 0.05
    same = paired_bootstrap_f1_diff(worse, worse, n_boot=200)
    assert same["diff"] == 0 and same["lo"] == same["hi"] == 0
    with pytest.raises(ValueError):
        paired_bootstrap_f1_diff(better, worse[:5])
