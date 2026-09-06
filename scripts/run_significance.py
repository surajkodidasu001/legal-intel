"""Paired randomization test between the top two classifiers.

Separate from run_experiment.py because it needs per-example scores from two
models fitted on the same split, and because the pairing is the whole point:
marginal bootstrap CIs on each model overlap, but the paired test does not
agree with that reading.
"""
import sys, yaml
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds
from legalintel.utils.stats import paired_randomization_test

A, B = "linear_svm", "logreg"


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])

    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")
    tr, dv = df[df.split == "train"], df[df.split == "dev"]

    a = fit_and_score(A, tr["text"], tr["label_name"], dv["text"], dv["label_name"])
    b = fit_and_score(B, tr["text"], tr["label_name"], dv["text"], dv["label_name"])

    res = paired_randomization_test(a.per_example_correct, b.per_example_correct)
    print(A, "vs", B, res)

    ExperimentResult(
        experiment="classify.significance",
        variant=f"{A}_vs_{B}",
        split="dev",
        config={"test": "paired_randomization", "n_perm": 10000, "metric": "accuracy"},
        metrics=res,
        n_examples=len(dv),
        notes=(
            "paired test on identical dev examples; marginal bootstrap CIs for "
            "these two models overlap, but the paired test rejects equality"
        ),
    ).save()


if __name__ == "__main__":
    main()
