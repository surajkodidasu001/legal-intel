"""Look at what pretrained spaCy actually finds in real LEDGAR text before
building anything. Same discipline as profile_dedup.py in Phase 2: profile
before you parallelize/extend, don't assume.

Usage:
    python scripts/explore_ner.py --input data/processed/ledgar_60000.parquet --sample 15
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sample", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    nlp = spacy.load(args.model)

    df = pd.read_parquet(args.input, columns=[args.text_col])
    sample = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    label_counts = Counter()
    docs_with_no_entities = 0

    for i, text in enumerate(sample[args.text_col].astype(str), 1):
        doc = nlp(text)
        print(f"\n--- doc {i} ---")
        print(text[:300] + ("..." if len(text) > 300 else ""))
        if not doc.ents:
            print("  (no entities found)")
            docs_with_no_entities += 1
        for ent in doc.ents:
            print(f"  [{ent.label_:10}] {ent.text}")
            label_counts[ent.label_] += 1

    print(f"\n\n=== summary over {len(sample)} docs ===")
    print(f"docs with zero entities: {docs_with_no_entities} ({100*docs_with_no_entities/len(sample):.0f}%)")
    print("\nentity label frequency:")
    for label, count in label_counts.most_common():
        print(f"  {label:10} {count}")


if __name__ == "__main__":
    main()
