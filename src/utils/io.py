"""I/O utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Write an iterable of dicts as UTF-8 JSONL."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into memory."""
    input_path = Path(path)
    if not input_path.exists():
        return []
    with input_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_dataframe(path: str | Path, frame: pd.DataFrame) -> None:
    """Write a dataframe to CSV."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    frame.to_csv(output_path, index=False, encoding="utf-8")
