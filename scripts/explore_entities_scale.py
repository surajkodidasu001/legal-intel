"""Larger-sample entity survey: aggregate spaCy label counts plus targeted
regex checks for patterns hypothesized from the 15-doc spot check (defined
terms, governing-law jurisdictions, money amounts). Decides what's worth
building extraction rules for based on actual frequency, not a guess.

Usage:
    python scripts/explore_entities_scale.py --input data/processed/ledgar_60000.parquet --sample 200
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# the "Guarantor" / the "Company" / (the "Servicer") -- defined-term declaration
DEFINED_TERM_RE = re.compile(r'\(?\s*(?:the|this)\s+[\u201c"]([A-Z][A-Za-z0-9 /\-]{1,40})[\u201d"]\s*\)?')
# "State of Georgia" / "State of Delaware" -- governing-law jurisdiction
GOVERNING_LAW_RE = re.compile(r'State of ([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)')
# $1,000,000.00 / $50,000 -- money amounts
MONEY_RE = re.compile(r'\$[\d,]+(?:\.\d{2})?')
# ordinary date-like patterns spaCy might miss (MM/DD/YYYY)
SLASH_DATE_RE = re.compile(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="en_core_web_sm")
    args = ap.parse_args()

    print(f"Loading {args.model}...")
    nlp = spacy.load(args.model)

    df = pd.read_parquet(args.input, columns=[args.text_col])
    sample = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    texts = sample[args.text_col].astype(str).tolist()
    n_docs = len(texts)

    label_counts = Counter()
    docs_with_label = Counter()

    docs_with_defined_term = 0
    docs_with_governing_law = 0
    docs_with_money = 0
    docs_with_slash_date = 0
    defined_terms_found = Counter()
    governing_law_states = Counter()

    print(f"Processing {n_docs} documents...")
    for text in nlp.pipe(texts, batch_size=64):
        seen_labels_this_doc = set()
        for ent in text.ents:
            label_counts[ent.label_] += 1
            seen_labels_this_doc.add(ent.label_)
        for lbl in seen_labels_this_doc:
            docs_with_label[lbl] += 1

    for text in texts:
        dt_matches = DEFINED_TERM_RE.findall(text)
        if dt_matches:
            docs_with_defined_term += 1
            for m in dt_matches:
                defined_terms_found[m] += 1

        gl_matches = GOVERNING_LAW_RE.findall(text)
        if gl_matches:
            docs_with_governing_law += 1
            for m in gl_matches:
                governing_law_states[m] += 1

        if MONEY_RE.search(text):
            docs_with_money += 1
        if SLASH_DATE_RE.search(text):
            docs_with_slash_date += 1

    print(f"\n=== spaCy entity labels over {n_docs} docs ===")
    print(f"{'label':12} {'total mentions':>15} {'docs containing':>17} {'% of docs':>11}")
    for label, count in label_counts.most_common():
        pct = 100 * docs_with_label[label] / n_docs
        print(f"{label:12} {count:15} {docs_with_label[label]:17} {pct:10.1f}%")

    print(f"\n=== regex pattern checks over {n_docs} docs ===")
    print(f"defined-term pattern (the \"X\"):  {docs_with_defined_term:4} docs ({100*docs_with_defined_term/n_docs:.1f}%), "
          f"{len(defined_terms_found)} unique terms, {sum(defined_terms_found.values())} total mentions")
    print(f"governing-law (State of X):       {docs_with_governing_law:4} docs ({100*docs_with_governing_law/n_docs:.1f}%)")
    print(f"money ($X,XXX):                    {docs_with_money:4} docs ({100*docs_with_money/n_docs:.1f}%)")
    print(f"slash-format dates (MM/DD/YYYY):   {docs_with_slash_date:4} docs ({100*docs_with_slash_date/n_docs:.1f}%)")

    print("\ntop 15 defined terms found:")
    for term, count in defined_terms_found.most_common(15):
        print(f"  {term:30} {count}")

    print("\ngoverning-law states found:")
    for state, count in governing_law_states.most_common(15):
        print(f"  {state:20} {count}")


if __name__ == "__main__":
    main()
