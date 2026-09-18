"""Dask-parallelized near-duplicate clustering.

Only the minhash computation is parallelized here, deliberately. LSH insert/query
is stateful and incremental (each query checks against everything inserted so
far), and union-find is a shared mutable structure -- neither parallelizes
cleanly without changing the algorithm's semantics. MinHash computation per
document has no such dependency: shingling + hashing one document doesn't touch
any other document's state, which is what makes it embarrassingly parallel.

So the shape of this module is:
    docs -> [Dask: minhash in parallel across partitions/workers] -> signatures
    signatures -> [same sequential LSH+union-find loop as dedup.py] -> clusters

This should produce IDENTICAL clusters to dedup.cluster_near_duplicates for the
same input and config, since the sequential merge logic is unchanged -- only
where the signatures come from differs. Worth asserting that in a test before
trusting any benchmark numbers from this module.
"""
from __future__ import annotations

import threading
import time
from typing import Union

import dask.bag as db
import psutil
from dask.distributed import Client, LocalCluster
from datasketch import MinHash, MinHashLSH

from legalintel.data.dedup import DedupConfig, minhash


def _compute_signature(item: tuple[str, str], num_perm: int, k: int) -> tuple[str, MinHash]:
    """Top-level (picklable) worker function: (doc_id, text) -> (doc_id, MinHash)."""
    did, text = item
    return did, minhash(text, num_perm=num_perm, k=k)


def _compute_signature_batch(
    items: list[tuple[str, str]], num_perm: int, k: int
) -> tuple[list[tuple[str, MinHash]], float]:
    """Batch version for the scatter path: one task per chunk instead of one
    task per document, so per-task scheduling overhead doesn't scale with
    n_docs.

    Returns (signatures, compute_seconds) -- the timer runs on the worker and
    stops before the return value is serialized, so compute_seconds measures
    pure minhash work only, excluding result pickling/transfer back to the
    driver. Comparing sum(compute_seconds across all batches) / n_workers
    (the "ideal" parallel wall time if transfer were free) against the
    driver's actual gather_seconds isolates how much of gather's wall time is
    genuinely waiting on compute vs. moving data.
    """
    t0 = time.perf_counter()
    sigs = [(did, minhash(text, num_perm=num_perm, k=k)) for did, text in items]
    elapsed = time.perf_counter() - t0
    return sigs, elapsed


def _chunk(items: list, n_chunks: int) -> list[list]:
    """Split into n_chunks near-equal contiguous slices (last chunk absorbs
    the remainder)."""
    n = len(items)
    size = -(-n // n_chunks)  # ceil division
    return [items[i : i + size] for i in range(0, n, size)]


def _worker_rss() -> int:
    """Runs ON a worker process via client.run(). Top-level so it's picklable."""
    return psutil.Process().memory_info().rss


class _ClusterRSSSampler:
    """Polls every worker's RSS (via client.run, which fans out to all workers)
    on a background thread and tracks the peak total across the cluster.

    This is a real, if imperfect, picture: it's a poll, not a continuous trace,
    so a spike shorter than `interval` between samples can be missed. It does
    NOT include the driver process's own memory -- see the separate
    _PeakRSSSampler in benchmark_dask_dedup.py for that half.
    """

    def __init__(self, client: Client, interval: float = 0.1):
        self.client = client
        self.interval = interval
        self._peak_total_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                per_worker = self.client.run(_worker_rss)
                total = sum(per_worker.values())
                self._peak_total_bytes = max(self._peak_total_bytes, total)
            except Exception:
                # Workers may not be up yet, or the cluster may be tearing
                # down concurrently -- a missed sample is fine, a crashed
                # benchmark run is not.
                pass
            self._stop.wait(self.interval)

    def __enter__(self) -> "_ClusterRSSSampler":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_total_mb(self) -> float:
        return self._peak_total_bytes / (1024 * 1024)


def _start_cluster_with_retry(
    n_workers: int, threads_per_worker: int, max_attempts: int = 2, backoff_seconds: float = 2.0
) -> tuple[LocalCluster, Client]:
    """Rapidly creating and tearing down LocalClusters in a loop (as the worker
    sweep in benchmark_dask_dedup.py does -- 8 create/destroy cycles in one
    process run) occasionally hits transient comm failures during startup
    (observed: CommClosedError / heartbeat failures on macOS). One retry after
    a short pause has been sufficient in practice; if it fails twice, that's a
    real problem worth surfacing, not swallowing.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            cluster = LocalCluster(
                n_workers=n_workers,
                threads_per_worker=threads_per_worker,
                processes=True,
            )
            client = Client(cluster)
            return cluster, client
        except Exception as e:  # noqa: BLE001 -- genuinely want to catch broadly here
            last_error = e
            if attempt < max_attempts:
                time.sleep(backoff_seconds)
    raise RuntimeError(
        f"Failed to start Dask LocalCluster after {max_attempts} attempts"
    ) from last_error


def cluster_near_duplicates_dask(
    doc_ids: list[str],
    texts: list[str],
    cfg: DedupConfig | None = None,
    n_workers: int = 4,
    threads_per_worker: int = 1,
    npartitions: int | None = None,
    use_scatter: bool = False,
    return_timing: bool = False,
) -> Union[dict[str, int], tuple[dict[str, int], dict[str, float]]]:
    """Same contract and return type as dedup.cluster_near_duplicates, unless
    return_timing=True (see below).

    n_workers / threads_per_worker: process-based workers, since minhash
    computation is CPU-bound and Python's GIL would otherwise serialize it
    under threads. This mirrors the n_jobs=-1 vs n_jobs=2 finding from Phase 1:
    be deliberate about worker/thread counts rather than defaulting to "all cores".

    npartitions: independent of n_workers so you can vary chunk granularity
    without conflating it with worker count when interpreting the scaling
    curve. Defaults to 4x n_workers if not given -- enough partitions per
    worker to smooth out stragglers without per-partition overhead dominating.

    use_scatter: False (default) uses db.from_sequence, which embeds every
    (doc_id, text) pair directly into the task graph and re-serializes the
    whole thing to the cluster on every call -- cheap to write, but Dask
    itself warns about this ("Sending large graph") once the corpus is large,
    and the cost is roughly fixed per run rather than shrinking as workers
    increase, which erodes scaling efficiency at higher worker counts.
    True switches to client.scatter(): the corpus is chunked and pushed to
    workers once as futures, and the graph carries references instead of the
    data itself. Run both and diff the scaling curves -- if scatter recovers
    efficiency at 8 workers, the graph-serialization cost was the bottleneck;
    if it doesn't, look elsewhere (e.g. per-task scheduling overhead, memory
    pressure).

    return_timing: False (default) preserves the original contract -- just the
    {doc_id: cluster_id} dict, so existing callers/tests are unaffected. True
    additionally returns a second dict breaking wall time into:
      - cluster_startup_seconds: LocalCluster + Client construction
      - compute_seconds: submitting + gathering the minhash work
      - teardown_seconds: client.close() + cluster.close()
      - merge_seconds: the sequential LSH+union-find phase
    When use_scatter=True, compute_seconds is further split into:
      - scatter_seconds: client.scatter(chunks) -- pushing input data to workers
      - submit_seconds: client.map(...) -- task graph submission (should be ~0)
      - gather_seconds: client.gather(...) -- waiting on compute + pulling
        results back. This conflates worker compute time with result-transfer
        time, which is why each batch self-reports its own pure compute time
        (see _compute_signature_batch) -- summed and divided by n_workers,
        that gives ideal_parallel_compute_seconds: the best-case gather_seconds
        if transfer were free. gather_seconds - ideal_parallel_compute_seconds
        is a lower bound on transfer/scheduling overhead on the output side.
        These four are None when use_scatter=False, since db.from_sequence
        doesn't expose an equivalent scatter/submit/gather split.
    worker_peak_rss_mb: peak total RSS summed across all worker subprocesses
        (polled via client.run() every 0.1s during compute) -- NOT including
        the driver process's own memory. Combine with a driver-side sampler
        for a full picture; see _PeakRSSSampler in benchmark_dask_dedup.py.
    Cluster startup/teardown is a fixed cost per call, independent of corpus
    size or worker count in the sense that it doesn't shrink as either grows --
    which makes it a prime suspect for scaling-efficiency loss that neither
    more workers nor scatter() can fix, since both leave this cost untouched.
    """
    cfg = cfg or DedupConfig()
    npartitions = npartitions or (4 * n_workers)

    t0 = time.perf_counter()
    cluster, client = _start_cluster_with_retry(n_workers, threads_per_worker)
    cluster_startup_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    scatter_seconds = submit_seconds = gather_seconds = None
    ideal_parallel_compute_seconds = None
    with _ClusterRSSSampler(client) as worker_rss_sampler:
        try:
            items = list(zip(doc_ids, texts))
            if use_scatter:
                chunks = _chunk(items, npartitions)

                t_scatter = time.perf_counter()
                futures = client.scatter(chunks)
                scatter_seconds = time.perf_counter() - t_scatter

                t_submit = time.perf_counter()
                batch_futures = client.map(
                    _compute_signature_batch, futures, num_perm=cfg.num_perm, k=cfg.shingle_k
                )
                submit_seconds = time.perf_counter() - t_submit

                t_gather = time.perf_counter()
                batches = client.gather(batch_futures)
                gather_seconds = time.perf_counter() - t_gather

                sig_pairs = [pair for batch, _ in batches for pair in batch]
                # sum of each batch's own reported compute time / n_workers is the
                # wall time you'd see if all workers ran flat-out in parallel with
                # zero transfer cost -- the best case gather_seconds could reach.
                total_batch_compute = sum(elapsed for _, elapsed in batches)
                ideal_parallel_compute_seconds = total_batch_compute / n_workers
            else:
                bag = db.from_sequence(items, npartitions=npartitions)
                sig_pairs = bag.map(
                    _compute_signature, num_perm=cfg.num_perm, k=cfg.shingle_k
                ).compute()
            compute_seconds = time.perf_counter() - t0
        finally:
            t0 = time.perf_counter()
            client.close()
            cluster.close()
            teardown_seconds = time.perf_counter() - t0
    worker_peak_rss_mb = worker_rss_sampler.peak_total_mb

    sigs: dict[str, MinHash] = dict(sig_pairs)

    # --- sequential merge: identical to dedup.cluster_near_duplicates from here ---
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
    out: dict[str, int] = {}
    for did in doc_ids:
        r = find(did)
        if r not in roots:
            roots[r] = len(roots)
        out[did] = roots[r]
    merge_seconds = time.perf_counter() - t0

    if not return_timing:
        return out

    timings = {
        "cluster_startup_seconds": cluster_startup_seconds,
        "compute_seconds": compute_seconds,
        "teardown_seconds": teardown_seconds,
        "merge_seconds": merge_seconds,
        "scatter_seconds": scatter_seconds,
        "submit_seconds": submit_seconds,
        "gather_seconds": gather_seconds,
        "ideal_parallel_compute_seconds": ideal_parallel_compute_seconds,
        "worker_peak_rss_mb": worker_peak_rss_mb,
    }
    return out, timings
