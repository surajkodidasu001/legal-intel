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
    # 3. create deterministic document ID
    df["doc_id"] = df["text"].map(
        lambda t: hashlib.sha1(t.encode()).hexdigest()[:16]
    )

    # 2. sample cfg["data"]["n_docs"] with random_state=cfg["seed"]
    # 3. doc_id = sha1 of text, first 16 chars
    # 4. cluster near-duplicates, then map back onto the frame
    clusters = cluster_near_duplicates(

            df["doc_id"].tolist(),
	    df["text"].tolist(),
            DedupConfig(**cfg["dedup"])

    )

    df["dup_cluster"] = df["doc_id"].map(clusters)

    sizes = df.groupby("dup_cluster").size()

    print("clusters:", len(sizes), "of", len(df), "docs")

    print("docs in a cluster of >1:", (df["dup_cluster"].map(sizes) > 1).sum())

    print(sizes.sort_values(ascending=False).head())
    # 5. assign leakage-safe splits
    df["split"] = assign_splits(
        df,
        SplitConfig(**cfg["split"], seed=cfg["seed"])
    )

    print(split_report(df, label_col="label_name"))

    per_split = df.pivot_table(
        index="label_name",
        columns="split",
        aggfunc="size",
        fill_value=0,
    )

    print("classes absent from dev:", (per_split["dev"] == 0).sum())
    print("classes with dev support >= 10:", (per_split["dev"] >= 10).sum())
    # 6. print split_report + the per-class dev support checks
    # 7. write data/processed/ledgar_5k.parquet
    out = Path(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")

    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out)

    print("wrote", out, df.shape)


if __name__ == "__main__":
    main()
