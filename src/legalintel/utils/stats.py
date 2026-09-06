"""Significance testing. Exists to stop you writing 'A beats B' when the gap
is noise."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(
    per_example, stat_fn=np.mean, n_boot: int = 10_000, alpha: float = 0.05, seed: int = 0
):
    """Percentile bootstrap CI over per-example scores. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(per_example, dtype=float)
    n = len(x)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = stat_fn(x[idx], axis=1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(stat_fn(x)), float(lo), float(hi)


def paired_randomization_test(a, b, n_perm: int = 10_000, seed: int = 0) -> dict:
    """Two-sided paired permutation test on per-query scores.

    The right test for retrieval comparisons: the same queries are scored by both
    systems, so the pairing carries most of the signal.
    """
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired test requires aligned score vectors")
    diff = a - b
    observed = float(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = (signs * diff).mean(axis=1)
    p = float((np.abs(null) >= abs(observed) - 1e-12).mean())
    return {"mean_diff": observed, "p_value": p, "n": int(len(diff))}
def bootstrap_macro_f1(y_true, y_pred, n_boot=1000, alpha=0.05, seed=0):
    """Resample examples and recompute macro-F1 each time."""
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(
            f1_score(
                y_true[idx],
                y_pred[idx],
                average="macro",
                zero_division=0,
            )
        )

    lo, hi = np.percentile(
        boots,
        [100 * alpha / 2, 100 * (1 - alpha / 2)],
    )

    point = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    return float(point), float(lo), float(hi)
