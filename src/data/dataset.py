"""Processed dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ProcessedDataset:
    """Container for processed interactions and items."""

    interactions: pd.DataFrame
    items: pd.DataFrame


def load_processed_dataset(root: str | Path = "data/processed") -> ProcessedDataset:
    """Load the processed dataset from disk."""
    root_path = Path(root)
    return ProcessedDataset(
        interactions=pd.read_csv(root_path / "interactions.csv"),
        items=pd.read_csv(root_path / "items.csv"),
    )
