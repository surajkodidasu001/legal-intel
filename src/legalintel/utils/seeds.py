"""Single place where randomness is pinned. Import before anything stochastic."""
from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 20260905


def set_seeds(seed: int = DEFAULT_SEED) -> int:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed
