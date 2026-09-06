"""Run classification baselines on dev, write results/*.json."""
import sys, yaml
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet("data/processed/ledgar_60000.parquet")
    train = df[df["split"] == "train"]
    dev = df[df["split"] == "dev"]
    X_train = train["text"]
    y_train = train["label_name"]
    X_dev = dev["text"]
    y_dev = dev["label_name"]
    for name in cfg["classify"]["models"]:
        r = fit_and_score(
            name,
            X_train,
            y_train,
            X_dev,
            y_dev,
            seed=cfg["seed"],
        )

        print(r.name, r.metrics, round(r.train_seconds, 1))

        ExperimentResult(
            experiment="classify.baselines",
            variant=r.name,
            split="dev",
            config={
                **cfg["classify"],
                "n_docs": cfg["data"]["n_docs"],
            },
            metrics={
   	       **r.metrics,
   	       "zero_support_classes": int(
		   cfg["classify"]["n_classes"] - y_dev.nunique()),
},
            timings={
                "train_seconds": r.train_seconds,
                "predict_seconds_per_1k": r.predict_seconds_per_1k,
            },
            n_examples=len(X_dev),
            notes="5K plumbing run, not a reported result",
        ).save()

if __name__ == "__main__":
    main()
