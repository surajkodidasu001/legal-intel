"""Confirm cluster_near_duplicates_dask produces identical output to
cluster_near_duplicates before trusting any Phase 2 benchmark numbers.

Cluster IDs themselves may differ (both assign IDs in first-seen order, which
depends on iteration order -- doc_ids list order is the same in both, so IDs
should actually match too, but we only assert on the partition structure to be
safe against that detail).

Run: python -m pytest tests/test_dedup_dask_equivalence.py -v
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from legalintel.data.dedup import DedupConfig, cluster_near_duplicates
from legalintel.data.dedup_dask import cluster_near_duplicates_dask

REAL_DATA_PATH = Path("data/processed/ledgar_60000.parquet")


def _clusters_to_partition(clusters: dict[str, int]) -> set[frozenset[str]]:
    groups: dict[int, set[str]] = {}
    for doc_id, cid in clusters.items():
        groups.setdefault(cid, set()).add(doc_id)
    return {frozenset(g) for g in groups.values()}


def test_dask_matches_sequential_on_synthetic_duplicates():
    doc_ids = [f"d{i}" for i in range(40)]
    base = "this agreement shall be governed by the laws of the state of delaware"
    other = "the parties agree that any dispute arising under this contract"
    texts = []
    for i in range(40):
        if i % 4 == 0:
            texts.append(base)
        elif i % 4 == 1:
            texts.append(base + " and nothing else shall apply")
        else:
            texts.append(other + f" clause number {i}")

    cfg = DedupConfig(threshold=0.6, num_perm=64, shingle_k=3)
    seq = cluster_near_duplicates(doc_ids, texts, cfg)
    seq_partition = _clusters_to_partition(seq)

    par = cluster_near_duplicates_dask(doc_ids, texts, cfg, n_workers=2, npartitions=5)
    assert _clusters_to_partition(par) == seq_partition, (
        "Dask minhash computation (db.from_sequence path) changed clustering "
        "output -- do not trust benchmark numbers until this passes."
    )

    par_scattered = cluster_near_duplicates_dask(
        doc_ids, texts, cfg, n_workers=2, npartitions=5, use_scatter=True
    )
    assert _clusters_to_partition(par_scattered) == seq_partition, (
        "Dask minhash computation (client.scatter path) changed clustering "
        "output -- do not trust scatter-variant benchmark numbers until this passes."
    )


@pytest.mark.skipif(
    not REAL_DATA_PATH.exists(),
    reason=f"{REAL_DATA_PATH} not present -- run scripts/prepare_data.py first",
)
def test_dask_matches_sequential_on_real_corpus_sample():
    """The synthetic test above proves the parallelization logic is correct in
    principle, using 40 hand-written sentences. Real LEDGAR text can have
    things synthetic text doesn't -- odd whitespace, very short or empty
    provisions, unusual punctuation, non-ASCII characters -- any of which
    could behave differently across the sequential and Dask code paths (e.g.
    an empty-string shingle edge case). This runs the same equivalence check
    against a real sample to close that gap. Uses milestone1.yaml's actual
    dedup config (not the loosened synthetic-test thresholds) so it reflects
    real production settings, not just a convenient hypothetical.
    """
    import yaml

    cfg_yaml = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    cfg = DedupConfig(**cfg_yaml["dedup"])

    df = pd.read_parquet(REAL_DATA_PATH, columns=["doc_id", "text"])
    sample = df.sample(n=min(300, len(df)), random_state=0)
    doc_ids = sample["doc_id"].astype(str).tolist()
    texts = sample["text"].astype(str).tolist()

    seq = cluster_near_duplicates(doc_ids, texts, cfg)
    seq_partition = _clusters_to_partition(seq)

    par = cluster_near_duplicates_dask(doc_ids, texts, cfg, n_workers=2, npartitions=8)
    assert _clusters_to_partition(par) == seq_partition, (
        "Dask minhash computation (db.from_sequence path) diverged from sequential "
        "on a real 300-document sample -- synthetic-data test passing was not enough."
    )

    par_scattered = cluster_near_duplicates_dask(
        doc_ids, texts, cfg, n_workers=2, npartitions=8, use_scatter=True
    )
    assert _clusters_to_partition(par_scattered) == seq_partition, (
        "Dask minhash computation (client.scatter path) diverged from sequential "
        "on a real 300-document sample -- synthetic-data test passing was not enough."
    )
