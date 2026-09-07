# Project status

Legal Intelligence Research & ML Platform. Repo at ~/projects/legal-intel.
Conda env `legal`, Python 3.11, sklearn 1.9.

## Done

- **Data pipeline** (`scripts/prepare_data.py`). LEDGAR from
  `coastalcph/lex_glue`, 60,000 provisions, 100 classes. MinHash + LSH
  near-duplicate clustering at Jaccard 0.8 over 5-word shingles, then
  group-aware four-way split: train 42k / dev 6k / calib 6k / test 6k.
  Full-corpus duplicate rate 14.9 percent, largest cluster 99 documents.
  At 5K the rate was 2.8 percent, because sampling separated most pairs.
  Zero classes absent from dev at 60K, versus 9 at 5K.

- **Classification study** (`scripts/run_experiment.py`). TF-IDF plus
  Logistic Regression, Linear SVM, Multinomial Naive Bayes. Equal tuning
  budgets of 12 configs each, enforced in code. Dev macro F1: linear_svm
  0.8028, logreg 0.7999, naive_bayes 0.7445. Bootstrap CIs for the top two
  overlap, but a paired randomization test disagrees: mean accuracy
  difference 0.0077, p = 0.0009 (`scripts/run_significance.py`).
  Decision: Linear SVM, more accurate and 2.3x faster to train.

- **Calibration study** (`scripts/run_calibration.py`). Fit on the calib
  split only. Multiclass Brier: logreg raw 0.3433, logreg sigmoid 0.2665,
  logreg isotonic 0.2736, svm sigmoid 0.1962, svm isotonic 0.1976. Model
  choice matters roughly ten times more than calibration method. Linear
  SVM's missing predict_proba turned out not to be a limitation.

- **Infrastructure.** `ExperimentResult` writes results/*.json with git sha,
  config, metrics, timings. `scripts/build_report.py` regenerates the
  decision table and now pulls p-values from classify.significance.

- **Error analysis** (`scripts/run_error_analysis.py`). Support does not
  explain failures: Brokers with 30 dev examples scores 0.984, Applicable Laws
  with 38 scores 0.200. Dominant confusions are synonym pairs, several
  bidirectional. One provision labelled Applicable Laws reads in full "This
  Agreement shall be governed by the laws of the State of Arizona".

- **Label ambiguity** (`run_label_merge.py`, `run_label_merge_control.py`).
  Merging six synonym pairs: 0.8028 -> 0.8210 on 94 classes. Control merging
  six random pairs: 0.7981, slightly worse than baseline. The gain is semantic,
  not an artifact of fewer classes. Single random draw, no error bar.

## Known open items

- Training time is in the JSON but not in the decision table, so cost is
  not a visible decision input.
- LEDGAR's own train/val/test splits are ignored in favour of our own, so
  numbers are not directly comparable to published LexGLUE results.
- The test split has never been opened. Everything so far is dev.
- sklearn 1.9 emits a spurious stratification warning inside
  CalibratedClassifierCV even with FrozenEstimator. Verified the frozen
  model is not refit.

## Next: Phase 1

1. API and Docker. `src/legalintel/api/main.py` returns 503 on every endpoint;
   artifact persistence and loading are TODO. Docker never built.
2. Test-split freeze. Final run on test, once, after experiments are frozen.

## Later phases

2. Dask. Pandas baseline on preprocessing, profile, parallelize, benchmark
   1/2/4/8 workers. Motivated by a real finding: n_jobs=-1 was about 10x
   slower than n_jobs=2 due to memory contention.
3. NER. spaCy baseline, error analysis, domain rules for citations and
   statutes, hand-labeled test set.
4. Retrieval on CourtListener. Design the relevance protocol and hand
   validate 50 judgments before writing any retrieval code.
5. Postgres/pgvector, monitoring, final write-up.
