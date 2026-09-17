"""Profile cluster_near_duplicates to find the real bottleneck before parallelizing.

Splits the algorithm into its two phases and times them separately:
  1. shingle + minhash computation per document (hypothesized: embarrassingly parallel)
  2. LSH insert/query + union-find merge (hypothesized: inherently sequential)

Phase 1 is the only thing Phase 2's Dask work should touch. If phase 2 turns out
to dominate wall time instead, that changes the whole plan -- so confirm this
before writing any Dask code, not after.

Usage:
    python scripts/profile_dedup.py --input data/processed/ledgar_60000.parquet \
        --id-col doc_id --text-col text --sample 5000
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.data.dedup import DedupConfig, minhash


@dataclass
class PhaseTimings:
    n_docs: int
    minhash_seconds: float
    lsh_unionfind_seconds: float
    n_clusters: int
    largest_cluster: int

    @property
    def minhash_docs_per_sec(self) -> float:
        return self.n_docs / self.minhash_seconds if self.minhash_seconds > 0 else float("inf")

    @property
    def lsh_docs_per_sec(self) -> float:
        return self.n_docs / self.lsh_unionfind_seconds if self.lsh_unionfind_seconds > 0 else float("inf")

    def report(self) -> None:
        total = self.minhash_seconds + self.lsh_unionfind_seconds
        print(f"docs: {self.n_docs}")
        print(f"  minhash phase:      {self.minhash_seconds:8.3f}s  "
              f"({self.minhash_docs_per_sec:8.1f} docs/sec)  "
              f"{100 * self.minhash_seconds / total:5.1f}% of wall time")
        print(f"  lsh+unionfind phase:{self.lsh_unionfind_seconds:8.3f}s  "
              f"({self.lsh_docs_per_sec:8.1f} docs/sec)  "
              f"{100 * self.lsh_unionfind_seconds / total:5.1f}% of wall time")
        print(f"  total:              {total:8.3f}s")
        print(f"  clusters found:     {self.n_clusters}  (largest: {self.largest_cluster})")


def profile_cluster_near_duplicates(
    doc_ids: list[str], texts: list[str], cfg: DedupConfig | None = None
) -> PhaseTimings:
    cfg = cfg or DedupConfig()

    # --- phase 1: shingle + minhash ---
    t0 = time.perf_counter()
    sigs: dict[str, MinHash] = {}
    for did, text in zip(doc_ids, texts):
        sigs[did] = minhash(text, num_perm=cfg.num_perm, k=cfg.shingle_k)
    minhash_seconds = time.perf_counter() - t0

    # --- phase 2: LSH insert/query + union-find ---
    t0 = time.perf_counter()
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    parent = {d: d for d in doc_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for did in doc_ids:
        m = sigs[did]
        for other in lsh.query(m):
            union(did, other)
        lsh.insert(did, m)

    roots: dict[str, int] = {}
    cluster_sizes: dict[int, int] = {}
    for did in doc_ids:
        r = find(did)
        if r not in roots:
            roots[r] = len(roots)
        cid = roots[r]
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
    lsh_unionfind_seconds = time.perf_counter() - t0

    return PhaseTimings(
        n_docs=len(doc_ids),
        minhash_seconds=minhash_seconds,
        lsh_unionfind_seconds=lsh_unionfind_seconds,
        n_clusters=len(cluster_sizes),
        largest_cluster=max(cluster_sizes.values()) if cluster_sizes else 0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to processed parquet")
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sample", type=int, default=None, help="Optional row sample for a quick pass")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_parquet(args.input, columns=[args.id_col, args.text_col])
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    doc_ids = df[args.id_col].astype(str).tolist()
    texts = df[args.text_col].astype(str).tolist()

    timings = profile_cluster_near_duplicates(doc_ids, texts)
    timings.report()


if __name__ == "__main__":
    main()
