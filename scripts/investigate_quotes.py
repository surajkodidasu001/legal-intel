"""Before fixing the defined-term regex, look at what patterns actually
precede quoted phrases in real text -- don't guess at a broader pattern,
look at the data first.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

QUOTE_RE = re.compile(r'(.{0,50})[\u201c"]([A-Z][A-Za-z0-9 /\-]{1,40})[\u201d"]')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_parquet(args.input, columns=[args.text_col])
    sample = df.sample(n=min(args.sample, len(df)), random_state=args.seed)

    count = 0
    for text in sample[args.text_col].astype(str):
        for m in QUOTE_RE.finditer(text):
            preceding, quoted = m.group(1), m.group(2)
            print(f"...{preceding.strip()[-50:]}... [QUOTE: {quoted}]")
            count += 1

    print(f"\ntotal quoted-capitalized-phrase matches: {count} over {len(sample)} docs")


if __name__ == "__main__":
    main()
