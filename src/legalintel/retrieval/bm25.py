"""BM25 index.

Note on honesty: PostgreSQL full-text search ranks with ts_rank, which is NOT
BM25. If the README claims BM25, the index has to actually be BM25. `bm25s` is
used here because it is a real BM25 implementation, is fast enough for 100K
documents on a laptop, and keeps the dependency surface small.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import bm25s


@dataclass
class BM25Config:
    k1: float = 1.5
    b: float = 0.75
    method: str = "lucene"          # bm25s variant; state it in the README
    stopwords: str = "en"
    stemmer: bool = False           # ablation: legal terms are often better unstemmed


@dataclass
class BM25Index:
    cfg: BM25Config = field(default_factory=BM25Config)
    doc_ids: list[str] = field(default_factory=list)
    _retriever: object | None = None
    build_seconds: float = 0.0

    def build(self, doc_ids: list[str], texts: list[str]) -> "BM25Index":
        t0 = time.perf_counter()
        self.doc_ids = list(doc_ids)
        tokens = bm25s.tokenize(texts, stopwords=self.cfg.stopwords)
        r = bm25s.BM25(k1=self.cfg.k1, b=self.cfg.b, method=self.cfg.method)
        r.index(tokens)
        self._retriever = r
        self.build_seconds = time.perf_counter() - t0
        return self

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._retriever is None:
            raise RuntimeError("index not built")
        q = bm25s.tokenize([query], stopwords=self.cfg.stopwords)
        idx, scores = self._retriever.retrieve(q, k=min(k, len(self.doc_ids)))
        return [
            (self.doc_ids[int(i)], float(s)) for i, s in zip(idx[0], scores[0])
        ]

    def save(self, path: str | Path) -> None:
        if self._retriever is None:
            raise RuntimeError("index not built")
        self._retriever.save(str(path))

    def index_size_mb(self, path: str | Path) -> float:
        p = Path(path)
        if not p.exists():
            return float("nan")
        total = sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size
        return total / (1024 * 1024)
