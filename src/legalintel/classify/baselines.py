"""Classical baselines with EQUAL tuning budgets.

The budget rule is the whole point of this module. If Logistic Regression gets a
24-point grid and Naive Bayes gets defaults, the benchmark measures tuning effort,
not model quality. Every model here gets the same number of candidate
configurations, searched on train with CV, selected on dev.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TUNING_BUDGET = 12  # candidate configs per model -- identical for all of them


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True, sublinear_tf=True, min_df=3, max_df=0.9,
        ngram_range=(1, 2), strip_accents="unicode",
    )


# Each grid is sized to TUNING_BUDGET. Keep them that way.
GRIDS: dict[str, tuple[Any, dict]] = {
    "logreg": (
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 10.0], "vec__min_df": [2, 3], "vec__ngram_range": [(1, 1), (1, 2)]},
    ),
    "naive_bayes": (
        MultinomialNB(),
        {"clf__alpha": [0.01, 0.1, 1.0], "vec__min_df": [2, 3], "vec__ngram_range": [(1, 1), (1, 2)]},
    ),
    "linear_svm": (
        LinearSVC(class_weight="balanced"),
        {"clf__C": [0.1, 1.0, 10.0], "vec__min_df": [2, 3], "vec__ngram_range": [(1, 1), (1, 2)]},
    ),
}


@dataclass
class FitResult:
    name: str
    best_params: dict
    train_seconds: float
    predict_seconds_per_1k: float
    metrics: dict[str, float]
    per_example_correct: np.ndarray = field(repr=False, default=None)
    estimator: Any = field(repr=False, default=None)


def fit_and_score(
    name: str, X_train, y_train, X_eval, y_eval, seed: int = 20260905, n_jobs: int = 2
) -> FitResult:
    est, grid = GRIDS[name]
    n_candidates = int(np.prod([len(v) for v in grid.values()]))
    if n_candidates != TUNING_BUDGET:
        raise ValueError(
            f"{name}: grid has {n_candidates} candidates, budget is {TUNING_BUDGET}. "
            "Unequal budgets invalidate the comparison."
        )

    pipe = Pipeline([("vec", _tfidf()), ("clf", est)])
    search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=3, n_jobs=n_jobs, refit=True)

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = search.best_estimator_.predict(X_eval)
    predict_seconds = time.perf_counter() - t1

    correct = (np.asarray(preds) == np.asarray(y_eval)).astype(float)
    return FitResult(
        name=name,
        best_params=search.best_params_,
        train_seconds=train_seconds,
        predict_seconds_per_1k=predict_seconds / max(len(X_eval), 1) * 1000,
        metrics={
            "f1_macro": float(f1_score(y_eval, preds, average="macro")),
            "f1_micro": float(f1_score(y_eval, preds, average="micro")),
            "accuracy": float(correct.mean()),
        },
        per_example_correct=correct,
        estimator=search.best_estimator_,
    )
