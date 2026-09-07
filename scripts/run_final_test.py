"""Final evaluation on the held-out test split. Run once.

Every number reported until now has been on dev. This opens test for the first
and only time, with the model and calibration already selected by evidence.
"""
import sys, yaml
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds
from legalintel.utils.stats import bootstrap_macro_f1


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")
    tr = df[df.split == "train"]
    te = df[df.split == "test"]

    r = fit_and_score("linear_svm", tr["text"], tr["label_name"],
                      te["text"], te["label_name"], seed=cfg["seed"])
    point, lo, hi = bootstrap_macro_f1(te["label_name"], r.preds,
                                       n_boot=1000, seed=cfg["seed"])
    print(r.name, r.metrics)
    print("macro f1 CI:", round(lo, 4), round(hi, 4))

    ExperimentResult(
        experiment="classify.baselines",
        variant="linear_svm",
        split="test",
        config={**cfg["classify"], "n_docs": cfg["data"]["n_docs"]},
        metrics={**r.metrics, "f1_macro_lo": lo, "f1_macro_hi": hi},
        timings={"train_seconds": r.train_seconds},
        n_examples=len(te),
        notes="final test run, model selected on dev, test opened once",
    ).save()


if __name__ == "__main__":
    main()
