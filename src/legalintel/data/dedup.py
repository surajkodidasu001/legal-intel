"""Near-duplicate detection via MinHash + LSH.

Why this module exists: legal text is boilerplate-heavy. The same opinion appears
in multiple reporters; contract clauses repeat verbatim across filings. A random
train/test split therefore leaks near-identical documents across the boundary and
inflates every downstream metric.

The protocol is: cluster near-duplicates FIRST, then split by cluster so that all
members of a cluster land on the same side. Report metrics under both the naive
and the deduplicated split -- the gap between them is a finding, not an
embarrassment.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

_WS = re.compile(r"\s+")


def shingles(text: str, k: int = 5) -> set[str]:
    """Word-level k-shingles. Word shingles beat character shingles here because
    legal boilerplate differs in whitespace/casing but not word order."""
    words = _WS.sub(" ", text.lower()).strip().split(" ")
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def minhash(text: str, num_perm: int = 128, k: int = 5) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for sh in shingles(text, k=k):
        m.update(sh.encode("utf8"))
    return m


@dataclass
class DedupConfig:
    threshold: float = 0.8   # Jaccard similarity above which docs are "the same"
    num_perm: int = 128
    shingle_k: int = 5


def cluster_near_duplicates(
    doc_ids: list[str], texts: list[str], cfg: DedupConfig | None = None
) -> dict[str, int]:
    """Return {doc_id: cluster_id}. Singletons get their own cluster.

    Uses union-find over LSH candidate pairs, so transitive duplicates
    (a~b, b~c) end up in one cluster even if a and c are not directly similar.
    """
    cfg = cfg or DedupConfig()
    lsh = MinHashLSH(threshold=cfg.threshold, num_perm=cfg.num_perm)
    sigs: dict[str, MinHash] = {}

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

    for did, text in zip(doc_ids, texts):
        m = minhash(text, num_perm=cfg.num_perm, k=cfg.shingle_k)
        sigs[did] = m
        for other in lsh.query(m):
            union(did, other)
        lsh.insert(did, m)

    roots = {}
    out = {}
    for did in doc_ids:
        r = find(did)
        if r not in roots:
            roots[r] = len(roots)
        out[did] = roots[r]
    return out
