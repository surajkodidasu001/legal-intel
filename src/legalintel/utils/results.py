"""Every experiment writes one JSON file here. README tables are generated from
these files, never hand-copied. A number in the README with no results file
behind it is a bug."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

RESULTS_DIR = Path("results")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


def config_hash(cfg: dict[str, Any]) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


@dataclass
class ExperimentResult:
    experiment: str                       # e.g. "classify.baselines"
    variant: str                          # e.g. "logreg_tfidf"
    split: str                            # "dev" or "test" -- test only after freeze
    config: dict[str, Any]
    metrics: dict[str, float]
    timings: dict[str, float] = field(default_factory=dict)    # seconds
    resources: dict[str, float] = field(default_factory=dict)  # MB, index size, ...
    n_examples: int | None = None
    notes: str = ""
    git_sha: str = field(default_factory=_git_sha)
    created_at: float = field(default_factory=time.time)
    env: dict[str, str] = field(
        default_factory=lambda: {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
    )

    def path(self) -> Path:
        return RESULTS_DIR / self.experiment / f"{self.variant}__{self.split}.json"

    def save(self) -> Path:
        p = self.path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        return p


def load_all(experiment: str | None = None) -> list[dict[str, Any]]:
    root = RESULTS_DIR / experiment if experiment else RESULTS_DIR
    if not root.exists():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.rglob("*.json"))]
