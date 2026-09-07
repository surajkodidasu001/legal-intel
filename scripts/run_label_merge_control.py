"""Control for the synonym merge: merge six random unrelated pairs instead.

Merging any labels raises macro F1 mechanically, by reducing class count and
raising per class support. This measures that mechanical effect so the synonym
merge can be compared against it.
"""
import sys, yaml, random
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")

    rng = random.Random(cfg["seed"])
    labels = sorted(df["label_name"].unique())
    picked = rng.sample(labels, 12)
    merges = {picked[i]: picked[i + 1] for i in range(0, 12, 2)}
    print("random merges:", merges)

    df["merged"] = df["label_name"].replace(merges)
    print("classes:", df["label_name"].nunique(), "->", df["merged"].nunique())

    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    r = fit_and_score("linear_svm", tr["text"], tr["merged"],
                      dv["text"], dv["merged"], seed=cfg["seed"])
    print(r.name, r.metrics)

    ExperimentResult(
        experiment="classify.label_merge",
        variant="linear_svm_random_control",
        split="dev",
        config={
            "merges": merges,
            "n_classes": int(df["merged"].nunique()),
            "n_docs": cfg["data"]["n_docs"],
        },
        metrics=r.metrics,
        n_examples=len(dv),
        notes="control: six random pairs merged",
    ).save()


if __name__ == "__main__":
    main()
