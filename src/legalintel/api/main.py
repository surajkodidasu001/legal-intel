"""Production API. Serves ONLY the variants that won their experiment.

Deliberately thin: the research value of this project is in the experiments, and
the API exists to prove the winning methods can be deployed, not to be a product.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Legal Intelligence Platform",
    description=(
        "Experimental research system. Not legal advice. Retrieval relevance "
        "judgments are partly derived from citation structure and are imperfect."
    ),
    version="0.1.0",
)

ARTIFACTS = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
_state: dict = {}


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=100)


class SearchHit(BaseModel):
    doc_id: str
    score: float


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1)


@app.on_event("startup")
def load_artifacts() -> None:
    manifest = ARTIFACTS / "manifest.json"
    if manifest.exists():
        _state["manifest"] = json.loads(manifest.read_text())
    clf_path = ARTIFACTS / "classifier.joblib"
    if clf_path.exists():
        import joblib
        _state["classifier"] = joblib.load(clf_path)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "loaded": sorted(_state.keys()),
        "manifest": _state.get("manifest", {}),
    }


@app.post("/search", response_model=list[SearchHit])
def search(req: SearchRequest) -> list[SearchHit]:
    index = _state.get("bm25")
    if index is None:
        raise HTTPException(503, "retrieval index not loaded")
    return [SearchHit(doc_id=d, score=s) for d, s in index.search(req.query, k=req.k)]


@app.post("/classify")
def classify(req: ClassifyRequest) -> dict:
    clf = _state.get("classifier")
    if clf is None:
        raise HTTPException(503, "classifier not loaded")
    pred = clf.predict([req.text])[0]
    out = {"label": str(pred)}
    if hasattr(clf, "predict_proba"):
        probs = clf.predict_proba([req.text])[0]
        out["confidence"] = float(max(probs))
        out["calibration"] = _state.get("manifest", {}).get("calibration", "none")
        out["calibration"] = _state.get("manifest", {}).get("calibration", "none")
    return out
