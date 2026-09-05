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
