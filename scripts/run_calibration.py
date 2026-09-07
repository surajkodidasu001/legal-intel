"""Probability calibration on the held-out calib split.

Fitting calibration on train would use data the model already memorised; fitting
on dev would use the split that selected the model. calib exists for exactly this.
"""
import sys, yaml
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.utils.results import ExperimentResult
from legalintel.utils.seeds import set_seeds

def multiclass_brier(y_true, proba, classes):
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[y] for y in y_true])

    y_onehot = np.zeros_like(proba)
    y_onehot[np.arange(len(y_idx)), y_idx] = 1.0

    return np.mean(np.sum((proba - y_onehot) ** 2, axis=1))
def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")   
    tr = df[df["split"] == "train"]
    cal = df[df["split"] == "calib"]
    dv = df[df["split"] == "dev"]

    model = Pipeline([
        (
            "vec",
            TfidfVectorizer(
                min_df=3,
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_df=0.9,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                C=1.0,
            ),
        ),
    ])

    model.fit(tr["text"], tr["label_name"])

    proba = model.predict_proba(dv["text"])
    raw_brier = multiclass_brier(dv["label_name"], proba, model.classes_)
    print("raw brier:", raw_brier)
    ExperimentResult(

        experiment="classify.calibration",
        variant="logreg_raw",
        split="dev",
        config={"model": "logreg", "method": "none",
                "calibrated_on": None, "n_docs": cfg["data"]["n_docs"]},
        metrics={"brier": raw_brier},
        n_examples=len(dv),
        notes="uncalibrated baseline for comparison",
    ).save()

    svm = Pipeline([
        ("vec", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True,
                                max_df=0.9, strip_accents="unicode")),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    svm.fit(tr["text"], tr["label_name"])

    fitted = {"logreg": model, "linear_svm": svm}

    for model_name, est in fitted.items():
        for method in ("sigmoid", "isotonic"):
            c = CalibratedClassifierCV(FrozenEstimator(est), method=method)
            c.fit(cal["text"], cal["label_name"])
            p = c.predict_proba(dv["text"])
            brier = multiclass_brier(dv["label_name"], p, c.classes_)
            variant = f"{model_name}_{method}"
            print(variant, round(brier, 4))
            ExperimentResult(
                experiment="classify.calibration",
                variant=variant,
                split="dev",
                config={"model": model_name, "method": method,
                        "calibrated_on": "calib", "n_docs": cfg["data"]["n_docs"]},
                metrics={"brier": brier},
                n_examples=len(dv),
                notes="calibration fit on held-out calib split, scored on dev",
            ).save()

if __name__ == "__main__":
    main()
