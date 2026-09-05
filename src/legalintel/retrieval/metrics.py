"""Retrieval metrics, all returning PER-QUERY vectors.

Per-query is the important design choice: it is what lets you run paired
significance tests and bootstrap CIs later. A function that returns only a mean
throws away the information you need to say whether a difference is real.
"""
from __future__ import annotations

import numpy as np


def recall_at_k(ranked: list[list[str]], relevant: list[set[str]], k: int) -> np.ndarray:
    out = []
    for r, rel in zip(ranked, relevant):
        if not rel:
            out.append(np.nan)
            continue
        out.append(len(set(r[:k]) & rel) / len(rel))
    return np.asarray(out, dtype=float)


def reciprocal_rank(ranked: list[list[str]], relevant: list[set[str]]) -> np.ndarray:
    out = []
    for r, rel in zip(ranked, relevant):
        rr = 0.0
        for i, d in enumerate(r, start=1):
            if d in rel:
                rr = 1.0 / i
                break
        out.append(rr)
    return np.asarray(out, dtype=float)


def ndcg_at_k(
    ranked: list[list[str]],
    gains: list[dict[str, float]],
    k: int,
) -> np.ndarray:
    """Graded nDCG. `gains` maps doc_id -> relevance grade (0 if absent).

    Binary judgments are just grades in {0, 1}; keeping the graded signature means
    you can upgrade the judgments later without changing the metric code.
    """
    out = []
    for r, g in zip(ranked, gains):
        dcg = sum(
            g.get(d, 0.0) / np.log2(i + 1) for i, d in enumerate(r[:k], start=1)
        )
        ideal = sorted(g.values(), reverse=True)[:k]
        idcg = sum(val / np.log2(i + 1) for i, val in enumerate(ideal, start=1))
        out.append(dcg / idcg if idcg > 0 else np.nan)
    return np.asarray(out, dtype=float)
