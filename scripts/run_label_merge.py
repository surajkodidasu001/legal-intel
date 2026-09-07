"""Does merging synonymous LEDGAR labels recover macro F1?

Error analysis showed the dominant confusions are between label pairs that are
not distinguishable from provision text. If that is label noise rather than
model error, merging those pairs should raise macro F1 substantially.
"""
import sys, yaml
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds

MERGES = {
    "Applicable Laws": "Governing Laws",
    "Defined Terms": "Definitions",
    "No Waivers": "Waivers",
    "Tax Withholdings": "Withholdings",
    "Integration": "Entire Agreements",
    "Authorizations": "Authority",
}


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")
    df["merged"] = df["label_name"].replace(MERGES)
    print("classes:", df["label_name"].nunique(), "->", df["merged"].nunique())

    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    r = fit_and_score("linear_svm", tr["text"], tr["merged"],
                      dv["text"], dv["merged"], seed=cfg["seed"])
    print(r.name, r.metrics)

    ExperimentResult(
        experiment="classify.label_merge",
        variant="linear_svm_merged",
        split="dev",
        config={"merges": MERGES, "n_classes": int(df["merged"].nunique()),
                "n_docs": cfg["data"]["n_docs"]},
        metrics=r.metrics,
        n_examples=len(dv),
        notes="six synonym pairs merged; baseline on full taxonomy is 0.8028",
    ).save()


if __name__ == "__main__":
    main()
