"""Sample docs for hand-labeling and write them to a plain-text file that's
easy to read top-to-bottom, plus a labels CSV template to fill in alongside
it. Fixed seed so the same 30 docs come out every time -- the gold set has
to be reproducible, same as every other artifact in this project.

Usage:
    python scripts/sample_for_labeling.py --input data/processed/ledgar_60000.parquet --n 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--id-col", default="doc_id")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/ner_labeling")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input, columns=[args.id_col, args.text_col])
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)

    docs_path = out_dir / "docs_to_label.txt"
    with open(docs_path, "w") as f:
        f.write(
            "HAND-LABELING SET -- read each doc below and record entities in "
            "labels_template.csv\n"
            "For each doc, look for:\n"
            "  GOVERNING_LAW  -- a US state named as the governing jurisdiction "
            "(e.g. 'the State of Delaware', 'State of New York')\n"
            "  DEFINED_TERM   -- a capitalized term being declared, usually in "
            "parens with quotes (e.g. '(the \"Company\")', '(this \"Amendment\")')\n"
            "Some docs will have zero of either -- that's fine, just don't add a "
            "row for them. Write down the EXACT text span as it appears (e.g. "
            "'New York' not 'the State of New York').\n"
            "\n" + "=" * 70 + "\n\n"
        )
        for i, row in sample.iterrows():
            f.write(f"=== DOC {i:03d} (id={row[args.id_col]}) ===\n")
            f.write(str(row[args.text_col]).strip() + "\n\n")

    labels_path = out_dir / "labels_template.csv"
    if not labels_path.exists():
        with open(labels_path, "w") as f:
            f.write("doc_index,entity_text,label\n")
            f.write("# example rows below -- delete these two lines and add your own\n")
            f.write("000,New York,GOVERNING_LAW\n")
            f.write("000,Company,DEFINED_TERM\n")
    else:
        print(f"{labels_path} already exists, leaving it alone (not overwriting your work)")

    print(f"wrote {len(sample)} docs to {docs_path}")
    print(f"labels template at {labels_path}")
    print(f"\nnext: open {docs_path} in a text editor, read each doc, and add rows to")
    print(f"{labels_path} using the doc_index shown (e.g. '000', '001', ...)")


if __name__ == "__main__":
    main()
