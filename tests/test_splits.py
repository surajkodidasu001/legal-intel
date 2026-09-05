"""These tests exist to make leakage impossible to reintroduce by accident."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalintel.data.splits import SPLITS, SplitConfig, assign_splits  # noqa: E402


@pytest.fixture
def df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "doc_id": [f"d{i}" for i in range(200)],
            "dup_cluster": [i // 4 for i in range(200)],   # clusters of 4
            "date": pd.date_range("2000-01-01", periods=200, freq="D"),
            "label": [i % 5 for i in range(200)],
        }
    )


def test_no_cluster_straddles_a_split(df):
    df["split"] = assign_splits(df, SplitConfig(mode="random"))
    per_cluster = df.groupby("dup_cluster")["split"].nunique()
    assert (per_cluster == 1).all(), "a near-duplicate cluster leaked across splits"


def test_all_splits_present_and_sized(df):
    df["split"] = assign_splits(df, SplitConfig(mode="random"))
    counts = df["split"].value_counts()
    assert set(counts.index) == set(SPLITS)
    assert counts["train"] > counts["test"]


def test_temporal_mode_puts_older_docs_in_train(df):
    df["split"] = assign_splits(df, SplitConfig(mode="temporal"))
    assert df[df["split"] == "train"]["date"].mean() < df[df["split"] == "test"]["date"].mean()


def test_missing_group_column_is_an_error(df):
    with pytest.raises(ValueError, match="dedup"):
        assign_splits(df.drop(columns=["dup_cluster"]))
