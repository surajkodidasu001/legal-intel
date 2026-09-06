"""Ingest LEDGAR -> dedup -> four-way group-aware split -> parquet."""
import sys, hashlib, yaml
from pathlib import Path
import pandas as pd
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.data.dedup import DedupConfig, cluster_near_duplicates
from legalintel.data.splits import SplitConfig, assign_splits, split_report
from legalintel.utils.seeds import set_seeds


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])


    # 1. load LEDGAR train -> pandas, keep label names not just ints

    ds = load_dataset("coastalcph/lex_glue", "ledgar", split="train")
    names = ds.features["label"].names

    df = ds.to_pandas()

    df["label_name"] = df["label"].map(lambda i: names[i])

    # 2. sample cfg["data"]["n_docs"] with random_state=cfg["seed"]

    df = df.sample(

        n=cfg["data"]["n_docs"],

        random_state=cfg["seed"]

    ).reset_index(drop=True)

    print(df.shape)
    print(df.head())

    # 2. sample cfg["data"]["n_docs"] with random_state=cfg["seed"]
    # 3. doc_id = sha1 of text, first 16 chars
    # 4. dup_cluster via cluster_near_duplicates(ids, texts, DedupConfig(**cfg["dedup"]))
    # 5. split via assign_splits(df, SplitConfig(**cfg["split"], seed=cfg["seed"]))
    # 6. print split_report + the per-class dev support checks
    # 7. write data/processed/ledgar_5k.parquet


if __name__ == "__main__":
    main()
