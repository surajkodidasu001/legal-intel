"""Split construction.

Three rules, all enforced here rather than left to discipline:

1. Group-aware: near-duplicate clusters never straddle a split boundary.
2. Four-way: train / dev / calib / test. Dev drives every design decision.
   Calib is reserved for probability calibration so that isotonic/Platt are never
   fit on data the model was selected against. Test is opened once, after freeze.
3. Temporal mode: train on older documents, evaluate on newer ones, to measure
   drift rather than assume it away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SPLITS = ("train", "dev", "calib", "test")


@dataclass
class SplitConfig:
    mode: str = "random"                       # "random" | "temporal"
    fractions: tuple = (0.70, 0.10, 0.10, 0.10)
    group_col: str = "dup_cluster"
    date_col: str = "date"
    seed: int = 20260905


def assign_splits(df: pd.DataFrame, cfg: SplitConfig | None = None) -> pd.Series:
    cfg = cfg or SplitConfig()
    if abs(sum(cfg.fractions) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    if cfg.group_col not in df.columns:
        raise ValueError(
            f"missing {cfg.group_col!r}: run dedup.cluster_near_duplicates first"
        )

    if cfg.mode == "random":
        groups = df[cfg.group_col].unique()
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(groups)
    elif cfg.mode == "temporal":
        # order clusters by their earliest document date; oldest -> train
        order = df.groupby(cfg.group_col)[cfg.date_col].min().sort_values()
        groups = order.index.to_numpy()
    else:
        raise ValueError(f"unknown split mode {cfg.mode!r}")

    # Assign whole clusters, walking the ordered cluster list and filling one
    # split at a time. Sequential (not round-robin) filling is what makes the
    # temporal mode meaningful: train gets the oldest block, test the newest.
    sizes = df.groupby(cfg.group_col).size().to_dict()
    total = len(df)
    quotas = [f * total for f in cfg.fractions]
    group_to_split: dict = {}

    split_i, filled = 0, 0
    for g in groups:
        # advance to the next split once this one has met its quota
        while split_i < len(SPLITS) - 1 and filled >= quotas[split_i]:
            split_i += 1
            filled = 0
        group_to_split[g] = SPLITS[split_i]
        filled += sizes[g]

    return df[cfg.group_col].map(group_to_split).rename("split")


def split_report(df: pd.DataFrame, label_col: str | None = None) -> pd.DataFrame:
    """Sanity table to paste into the README: sizes and label balance per split."""
    rows = []
    for s in SPLITS:
        sub = df[df["split"] == s]
        row = {"split": s, "n_docs": len(sub), "n_clusters": sub["dup_cluster"].nunique()}
        if label_col:
            vc = sub[label_col].value_counts(normalize=True)
            row["majority_class_share"] = float(vc.iloc[0]) if len(vc) else float("nan")
            row["n_classes"] = int(sub[label_col].nunique())
        rows.append(row)
    return pd.DataFrame(rows)
