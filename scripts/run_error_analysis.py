"""Error analysis for the selected classifier."""
import sys, yaml
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.classify.baselines import fit_and_score
from legalintel.utils.seeds import set_seeds


def main():
    cfg = yaml.safe_load(Path("configs/milestone1.yaml").read_text())
    set_seeds(cfg["seed"])
    df = pd.read_parquet(f"data/processed/ledgar_{cfg['data']['n_docs']}.parquet")
    tr, dv = df[df.split == "train"], df[df.split == "dev"]
    r = fit_and_score("linear_svm", tr["text"], tr["label_name"],
                      dv["text"], dv["label_name"], seed=cfg["seed"])
    rep = classification_report(dv["label_name"], r.preds,
                                output_dict=True, zero_division=0)
    pc = pd.DataFrame(rep).T
    pc = pc[pc.index.isin(dv["label_name"].unique())].sort_values("f1-score")
    cols = ["precision", "recall", "f1-score", "support"]
    print("\n10 worst:")
    print(pc[cols].head(10))
    print("\n10 best:")
    print(pc[cols].tail(10))

    err = pd.DataFrame({"true": dv["label_name"].values, "pred": r.preds})
    err = err[err["true"] != err["pred"]]
    pairs = err.groupby(["true", "pred"]).size().sort_values(ascending=False)
    print("\ntop 15 confusion pairs:")
    print(pairs.head(15))

    worst_true, worst_pred = pairs.index[0]
    print(f"\nexamples of {worst_true} predicted as {worst_pred}:")
    mask = (dv["label_name"].values == worst_true) & (r.preds == worst_pred)
    for t in dv["text"].values[mask][:3]:
        print("\n---", t[:400])

    Path("results/classify.errors").mkdir(parents=True, exist_ok=True)
    pc[cols].to_csv("results/classify.errors/per_class_linear_svm_dev.csv")
    pairs.head(50).to_csv("results/classify.errors/confusion_pairs_linear_svm_dev.csv")
    print("wrote results/classify.errors/")


if __name__ == "__main__":
    main()
