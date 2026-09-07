"""Fit and persist the production model."""
import json, sys, yaml
from pathlib import Path
import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.utils.seeds import set_seeds


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")
    tr = df[df.split == "train"]
    cal = df[df.split == "calib"]

    base = Pipeline([
        ("vec", TfidfVectorizer(min_df=3, ngram_range=(1, 2), sublinear_tf=True,
                                max_df=0.9, strip_accents="unicode")),
        ("clf", LinearSVC(C=1.0, class_weight="balanced")),
    ])
    base.fit(tr["text"], tr["label_name"])

    model = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
    model.fit(cal["text"], cal["label_name"])

    out = Path("artifacts")
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "classifier.joblib")

    manifest = {
        "model": "linear_svm",
        "calibration": "sigmoid",
        "calibrated_on": "calib",
        "n_train": len(tr),
        "n_classes": int(df["label_name"].nunique()),
        "dev_macro_f1": 0.8028,
        "dev_brier": 0.1962,
        "seed": cfg["seed"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote artifacts/")


if __name__ == "__main__":
    main()
