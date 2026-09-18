"""NER evaluation: precision/recall/F1 for GOVERNING_LAW and DEFINED_TERM
extraction against the 100-doc hand-labeled gold set
(data/ner_labeling/labels_template.csv), matching the standard Phase 1 held
itself to for classification (real computed metrics, not eyeballing).

Matching rule: an extracted entity counts as a true positive if its (label,
text) pair -- text compared case-insensitively -- appears in that doc's gold
set. This is text-match, not character-offset-match: simpler and defensible
at this scale, since every gold label was hand-verified against the actual
doc text already. A char-offset-based eval would be more rigorous but adds
real complexity for marginal benefit on a 100-doc set with no ambiguous
duplicate mentions per doc.

Usage:
    python scripts/evaluate_ner.py \
        --gold data/ner_labeling/labels_template.csv \
        --doc-files data/ner_labeling/docs_to_label.txt data/ner_labeling/review_batch_docs.txt
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legalintel.ner.extract import extract_entities, load_pipeline

DOC_HEADER_RE = re.compile(r"^=== DOC (\d{3}) \(id=.*?\) ===\s*$", re.MULTILINE)
TARGET_LABELS = ("GOVERNING_LAW", "DEFINED_TERM")


def load_docs(paths: list[str]) -> dict[str, str]:
    """Parse one or more '=== DOC NNN (id=...) ===' formatted files into
    {doc_index: text}. Strips any '[CANDIDATES]: ...' annotation line so the
    eval runs against the real contract text only, not our own tooling
    artifacts."""
    docs: dict[str, str] = {}
    for path in paths:
        content = Path(path).read_text()
        headers = list(DOC_HEADER_RE.finditer(content))
        for i, m in enumerate(headers):
            idx = m.group(1)
            start = m.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            body = content[start:end]
            lines = [
                ln for ln in body.splitlines()
                if not ln.strip().startswith("[CANDIDATES]:")
            ]
            text = "\n".join(lines).strip()
            if idx in docs:
                raise ValueError(f"doc index {idx} appears in more than one input file")
            docs[idx] = text
    return docs


def load_gold(path: str) -> dict[str, set[tuple[str, str]]]:
    """{doc_index: {(label, entity_text.lower()), ...}}"""
    gold: dict[str, set[tuple[str, str]]] = defaultdict(set)
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row["doc_index"].strip()
            text = row["entity_text"].strip()
            label = row["label"].strip()
            gold[idx].add((label, text.lower()))
    return dict(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--doc-files", nargs="+", required=True)
    ap.add_argument("--save-results", action="store_true", help="write an ExperimentResult to results/")
    args = ap.parse_args()

    docs = load_docs(args.doc_files)
    gold = load_gold(args.gold)

    missing = set(gold) - set(docs)
    if missing:
        raise ValueError(f"gold labels reference doc indices not found in --doc-files: {sorted(missing)}")

    print(f"loaded {len(docs)} docs, gold labels for {len(gold)} of them")
    print("loading spaCy pipeline...")
    nlp = load_pipeline()

    tp = fp = fn = 0
    per_label = {lbl: {"tp": 0, "fp": 0, "fn": 0} for lbl in TARGET_LABELS}
    false_positives_detail = []
    false_negatives_detail = []

    for idx in sorted(docs):
        text = docs[idx]
        gold_set = gold.get(idx, set())

        predicted = extract_entities(text, nlp=nlp)
        pred_set = {
            (e.label, e.text.lower())
            for e in predicted
            if e.label in TARGET_LABELS
        }

        doc_tp = pred_set & gold_set
        doc_fp = pred_set - gold_set
        doc_fn = gold_set - pred_set

        tp += len(doc_tp)
        fp += len(doc_fp)
        fn += len(doc_fn)

        for label, etext in doc_tp:
            per_label[label]["tp"] += 1
        for label, etext in doc_fp:
            per_label[label]["fp"] += 1
            false_positives_detail.append((idx, label, etext))
        for label, etext in doc_fn:
            per_label[label]["fn"] += 1
            false_negatives_detail.append((idx, label, etext))

    def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 and precision == precision and recall == recall
            else float("nan")
        )
        return precision, recall, f1

    overall_p, overall_r, overall_f1 = prf(tp, fp, fn)

    print(f"\n=== overall (GOVERNING_LAW + DEFINED_TERM combined) ===")
    print(f"TP={tp}  FP={fp}  FN={fn}")
    print(f"precision={overall_p:.3f}  recall={overall_r:.3f}  f1={overall_f1:.3f}")

    per_label_prf = {}
    print(f"\n=== per-label ===")
    for label in TARGET_LABELS:
        counts = per_label[label]
        p, r, f1 = prf(counts["tp"], counts["fp"], counts["fn"])
        per_label_prf[label] = {"precision": p, "recall": r, "f1": f1, **counts}
        print(f"{label:15} TP={counts['tp']:2}  FP={counts['fp']:2}  FN={counts['fn']:2}  "
              f"precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}")

    if false_positives_detail:
        print(f"\n=== false positives ({len(false_positives_detail)}) ===")
        for idx, label, etext in false_positives_detail:
            print(f"  doc {idx}: [{label}] predicted {etext!r} but not in gold")

    if false_negatives_detail:
        print(f"\n=== false negatives ({len(false_negatives_detail)}) ===")
        for idx, label, etext in false_negatives_detail:
            print(f"  doc {idx}: [{label}] gold has {etext!r} but not predicted")

    if not false_positives_detail and not false_negatives_detail:
        print("\nno errors -- perfect precision and recall on this gold set")

    if args.save_results:
        from legalintel.utils.results import ExperimentResult
        result = ExperimentResult(
            experiment="ner.extraction",
            variant="rule_based_governing_law_defined_term",
            split="hand_labeled_100docs",
            config={"target_labels": list(TARGET_LABELS), "matching": "case_insensitive_text_match"},
            metrics={
                "precision": overall_p,
                "recall": overall_r,
                "f1": overall_f1,
                **{f"{lbl.lower()}_precision": per_label_prf[lbl]["precision"] for lbl in TARGET_LABELS},
                **{f"{lbl.lower()}_recall": per_label_prf[lbl]["recall"] for lbl in TARGET_LABELS},
                **{f"{lbl.lower()}_f1": per_label_prf[lbl]["f1"] for lbl in TARGET_LABELS},
            },
            n_examples=len(docs),
            notes=(
                f"Gold set: {len(gold)} of {len(docs)} docs have at least one label, "
                f"{sum(len(v) for v in gold.values())} total gold entities. "
                "Text-match (case-insensitive) not char-offset-match -- see script docstring."
            ),
        )
        path = result.save()
        print(f"\nwrote results to {path}")


if __name__ == "__main__":
    main()
