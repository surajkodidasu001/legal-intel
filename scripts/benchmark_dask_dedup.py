"""Phase 2 core experiment: does Dask actually improve dedup throughput, and
where does scaling efficiency break down?

Measures docs/sec, wall time, and peak RSS for:
  - a plain pandas/sequential baseline (dedup.cluster_near_duplicates)
  - dedup_dask.cluster_near_duplicates_dask at 1, 2, 4, 8 workers

Each run writes one ExperimentResult to results/dask.scaling/, matching the
existing format so build_report.py / make report picks it up automatically.
No hand-typed numbers -- the scaling table in the README should be generated
from these files, same as everything else in the decision table.

Usage:
    python scripts/benchmark_dask_dedup.py --input data/processed/ledgar_60000.parquet
    python scripts/benchmark_dask_dedup.py --input data/processed/ledgar_60000.parquet \
        --sample 5000 --workers 1 2 4    # quick pass before the full run
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import psutil
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.data.dedup import DedupConfig, cluster_near_duplicates
from legalintel.data.dedup_dask import cluster_near_duplicates_dask
from legalintel.utils.results import ExperimentResult


class _PeakRSSSampler:
    """Samples this process's RSS on a background thread and tracks the max.

    Dask's LocalCluster spawns worker subprocesses, so this only captures the
    driver process's memory, not total across workers -- noted in the result's
    `notes` field rather than silently reported as if it were the whole picture.
    """

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._peak_bytes = 0
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                self._peak_bytes = max(self._peak_bytes, rss)
            except psutil.NoSuchProcess:
                break
            time.sleep(self.interval)

    def __enter__(self) -> "_PeakRSSSampler":
        self._peak_bytes = self._proc.memory_info().rss
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    @property
    def peak_mb(self) -> float:
        return self._peak_bytes / (1024 * 1024)


def run_baseline(
    doc_ids: list[str], texts: list[str], cfg: DedupConfig, n_docs: int, split: str = "full_corpus"
) -> ExperimentResult:
    print("running pandas/sequential baseline...")
    with _PeakRSSSampler() as sampler:
        t0 = time.perf_counter()
        cluster_near_duplicates(doc_ids, texts, cfg)
        wall = time.perf_counter() - t0

    result = ExperimentResult(
        experiment="dask.scaling",
        variant="baseline_pandas",
        split=split,
        config={"n_workers": 1, "backend": "sequential", **vars(cfg)},
        metrics={"docs_per_sec": n_docs / wall, "wall_seconds": wall},
        timings={"wall_seconds": wall},
        resources={
            "driver_peak_rss_mb": sampler.peak_mb,
            "worker_peak_rss_mb": 0.0,
            "total_peak_rss_mb": sampler.peak_mb,
        },
        n_examples=n_docs,
        notes="Sequential baseline: no Dask, single process -- driver and total are the same.",
    )
    result.save()
    print(f"  {n_docs} docs in {wall:.2f}s  ({n_docs / wall:.1f} docs/sec)  "
          f"peak RSS {sampler.peak_mb:.0f}MB")
    return result


def run_dask(
    doc_ids: list[str],
    texts: list[str],
    cfg: DedupConfig,
    n_docs: int,
    n_workers: int,
    use_scatter: bool = False,
    split: str = "full_corpus",
) -> ExperimentResult:
    label = f"dask, {n_workers} worker(s){' [scatter]' if use_scatter else ''}"
    print(f"running {label}...")
    with _PeakRSSSampler() as sampler:
        t0 = time.perf_counter()
        _, phase_timings = cluster_near_duplicates_dask(
            doc_ids, texts, cfg, n_workers=n_workers, use_scatter=use_scatter, return_timing=True
        )
        wall = time.perf_counter() - t0

    worker_peak_rss_mb = phase_timings.pop("worker_peak_rss_mb")
    driver_peak_rss_mb = sampler.peak_mb
    total_peak_rss_mb = driver_peak_rss_mb + worker_peak_rss_mb

    variant = f"dask_workers_{n_workers}" + ("_scattered" if use_scatter else "")
    result = ExperimentResult(
        experiment="dask.scaling",
        variant=variant,
        split=split,
        config={"n_workers": n_workers, "backend": "dask", "use_scatter": use_scatter, **vars(cfg)},
        metrics={"docs_per_sec": n_docs / wall, "wall_seconds": wall},
        timings={"wall_seconds": wall, **phase_timings},
        resources={
            "driver_peak_rss_mb": driver_peak_rss_mb,
            "worker_peak_rss_mb": worker_peak_rss_mb,
            "total_peak_rss_mb": total_peak_rss_mb,
        },
        n_examples=n_docs,
        notes=(
            "Dask LocalCluster, process-based workers, minhash computation only "
            "(LSH+union-find sequential in driver, per profile_dedup.py finding). "
            "total_peak_rss_mb = driver_peak_rss_mb (sampled on the benchmark process) "
            "+ worker_peak_rss_mb (polled across all worker subprocesses via "
            "client.run() every 0.1s during compute). Both are max-observed via "
            "polling, not a continuous trace, so a short-lived spike between polls "
            "can be missed; also driver and worker peaks may not have occurred at "
            "the same instant, so total_peak_rss_mb is an upper bound, not a "
            "simultaneous snapshot. "
            "timings.cluster_startup_seconds + teardown_seconds is fixed overhead per "
            "call, independent of corpus size -- does not shrink with more workers. "
            + (
                "Uses client.scatter() to push data to workers once as futures, "
                "avoiding embedding the full corpus in the task graph."
                if use_scatter
                else "Uses db.from_sequence(), which embeds the full corpus in the task graph "
                "and re-serializes it on every call -- see dedup_dask.py docstring."
            )
        ),
    )
    result.save()
    startup = phase_timings["cluster_startup_seconds"]
    teardown = phase_timings["teardown_seconds"]
    compute = phase_timings["compute_seconds"]
    merge = phase_timings["merge_seconds"]
    print(f"  {n_docs} docs in {wall:.2f}s  ({n_docs / wall:.1f} docs/sec)  "
          f"driver peak RSS {driver_peak_rss_mb:.0f}MB  worker peak RSS {worker_peak_rss_mb:.0f}MB  "
          f"(total ~{total_peak_rss_mb:.0f}MB)")
    print(f"    startup {startup:.2f}s  compute {compute:.2f}s  "
          f"merge {merge:.2f}s  teardown {teardown:.2f}s  "
          f"(fixed overhead: {startup + teardown:.2f}s)")
    if use_scatter:
        scatter_s = phase_timings["scatter_seconds"]
        submit_s = phase_timings["submit_seconds"]
        gather_s = phase_timings["gather_seconds"]
        ideal_s = phase_timings["ideal_parallel_compute_seconds"]
        transfer_overhead = gather_s - ideal_s
        print(f"    -> scatter {scatter_s:.2f}s  submit {submit_s:.3f}s  "
              f"gather {gather_s:.2f}s  (ideal parallel compute: {ideal_s:.2f}s, "
              f"est. transfer/scheduling overhead: {transfer_overhead:.2f}s)")
    return result


def print_scaling_summary(baseline: ExperimentResult, dask_runs: list[ExperimentResult], label: str) -> None:
    base_rate = baseline.metrics["docs_per_sec"]
    print(f"\nscaling summary [{label}] (vs pandas baseline):")
    print(f"{'workers':>8}  {'docs/sec':>10}  {'speedup':>8}  {'efficiency':>10}")
    print(f"{'baseline':>8}  {base_rate:10.1f}  {'--':>8}  {'--':>10}")
    for r in dask_runs:
        n = r.config["n_workers"]
        rate = r.metrics["docs_per_sec"]
        speedup = rate / base_rate
        efficiency = speedup / n  # 1.0 = perfect linear scaling
        print(f"{n:8d}  {rate:10.1f}  {speedup:7.2f}x  {efficiency:9.1%}")


def run_fixed_cost_sweep(
    df: pd.DataFrame, id_col: str, text_col: str, cfg: DedupConfig, workers: list[int], seed: int
) -> None:
    """Tests two separate claims about per-call Dask overhead across corpus size:

      1. cluster_lifecycle_seconds (startup + teardown): expected to be FIXED,
         independent of corpus size -- spinning up N worker processes costs the
         same whether they'll process 5K or 60K docs.
      2. scatter_seconds (pushing document text to workers): expected to SCALE
         with corpus size, since more documents means more bytes to transfer --
         this should NOT be lumped in with #1 as "fixed", and its share of
         total time is not guaranteed to shrink at larger corpora (more data to
         move takes proportionally longer, not proportionally less).

    An earlier version of this sweep summed both into one "fixed_overhead"
    number and that was a mistake: it made scatter's cost look like it would
    amortize away at scale the same way cluster lifecycle does, which testing
    at 5K/20K/60K showed is false -- scatter's share of total time crept up
    slightly (3.4% -> 4.7% -> 5.1% at 1 worker), not down. Reporting them
    separately here so that mistake doesn't get baked into the README too.

    Only runs the scatter variant (the recommended path from the earlier
    comparison), not db.from_sequence, to keep total runtime reasonable --
    that path's overhead is already understood and isn't in question here.
    """
    sizes = sorted({min(n, len(df)) for n in [5_000, 20_000, len(df)]})
    print(f"\nfixed-cost-vs-corpus-size sweep at n_docs = {sizes}")

    rows = []
    for n in sizes:
        sub = df if n == len(df) else df.sample(n=n, random_state=seed)
        doc_ids = sub[id_col].astype(str).tolist()
        texts = sub[text_col].astype(str).tolist()
        split = f"n{n}docs"

        run_baseline(doc_ids, texts, cfg, n, split=split)
        for w in workers:
            result = run_dask(doc_ids, texts, cfg, n, w, use_scatter=True, split=split)
            t = result.timings
            rows.append({
                "n_docs": n,
                "workers": w,
                "cluster_lifecycle_seconds": t["cluster_startup_seconds"] + t["teardown_seconds"],
                "scatter_seconds": t["scatter_seconds"],
                "scatter_pct_of_compute": 100 * t["scatter_seconds"] / t["compute_seconds"],
            })

    print(f"\n{'n_docs':>8}  {'workers':>8}  {'cluster lifecycle':>18}  {'scatter':>9}  {'scatter % of compute':>21}")
    for r in rows:
        print(f"{r['n_docs']:8d}  {r['workers']:8d}  {r['cluster_lifecycle_seconds']:17.2f}s  "
              f"{r['scatter_seconds']:8.2f}s  {r['scatter_pct_of_compute']:20.1f}%")
    print(
        "\ncluster_lifecycle should stay roughly flat across n_docs (fixed cost per call, "
        "amortizes better on larger corpora). scatter should grow roughly proportionally with "
        "n_docs (transferring more text takes more time) -- scatter_pct_of_compute shows "
        "whether that share is shrinking, flat, or growing as corpus size increases; growing "
        "means scatter cost does NOT amortize away at scale the way cluster lifecycle does."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to processed parquet")
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sample", type=int, default=None, help="Optional row sample for a quick pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--dedup-config",
        default="configs/milestone1.yaml",
        help="Read dedup: block from this config, same as prepare_data.py",
    )
    ap.add_argument(
        "--compare-scatter",
        action="store_true",
        help="Also run the client.scatter() variant at each worker count, to test "
        "whether task-graph serialization overhead explains the efficiency drop-off.",
    )
    ap.add_argument(
        "--fixed-cost-sweep",
        action="store_true",
        help="Test whether Dask's per-call fixed overhead (startup+teardown+scatter) "
        "actually stays constant across corpus sizes (5K/20K/full), instead of just "
        "assuming it. Runs in addition to the normal single-size sweep above.",
    )
    args = ap.parse_args()

    cfg_yaml = yaml.safe_load(Path(args.dedup_config).read_text())
    cfg = DedupConfig(**cfg_yaml["dedup"])

    df_full = pd.read_parquet(args.input, columns=[args.id_col, args.text_col])
    df = df_full
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    doc_ids = df[args.id_col].astype(str).tolist()
    texts = df[args.text_col].astype(str).tolist()
    n_docs = len(doc_ids)

    baseline = run_baseline(doc_ids, texts, cfg, n_docs)
    dask_runs = [run_dask(doc_ids, texts, cfg, n_docs, n) for n in args.workers]
    print_scaling_summary(baseline, dask_runs, label="db.from_sequence")

    if args.compare_scatter:
        scattered_runs = [
            run_dask(doc_ids, texts, cfg, n_docs, n, use_scatter=True) for n in args.workers
        ]
        print_scaling_summary(baseline, scattered_runs, label="client.scatter")

    if args.fixed_cost_sweep:
        run_fixed_cost_sweep(df_full, args.id_col, args.text_col, cfg, args.workers, args.seed)


if __name__ == "__main__":
    main()
