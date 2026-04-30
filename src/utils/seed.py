"""Random seed helpers."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch if available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
