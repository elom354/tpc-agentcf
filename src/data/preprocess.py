"""Preprocessing pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)

GENRE_TO_DOMAIN = [
    ("Drama", "drama"),
    ("Comedy", "comedy"),
    ("Action", "action"),
    ("Thriller", "action"),
    ("Romance", "romance"),
]


def _map_movielens_domain(genres: str) -> str:
    for token, domain in GENRE_TO_DOMAIN:
        if token in genres:
            return domain
    return "other"


def preprocess_dataset(raw_root: str | Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build processed interaction and item tables for the configured dataset."""
    dataset = config["data"]["dataset"]
    if dataset == "amazon":
        interactions, items = _preprocess_amazon(Path(raw_root), config)
    else:
        interactions, items = _preprocess_movielens(Path(raw_root), config)
    interactions = apply_cross_domain_filter(interactions, dataset, config["data"]["min_interactions_per_user"])
    interactions = limit_users_and_items(interactions, config)
    interactions = chronological_split(interactions)
    processed_dir = ensure_dir("data/processed")
    interactions.to_csv(processed_dir / "interactions.csv", index=False)
    items[items["item_id"].isin(interactions["item_id"].unique())].to_csv(processed_dir / "items.csv", index=False)
    return interactions, items


def _preprocess_movielens(raw_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(raw_root / "ratings.csv")
    movies = pd.read_csv(raw_root / "movies.csv")
    movies["domain"] = movies["genres"].fillna("").map(_map_movielens_domain)
    movies["description"] = movies["title"].fillna("") + " [" + movies["genres"].fillna("") + "]"
    items = movies.rename(columns={"movieId": "item_id"})[["item_id", "title", "domain", "description"]]
    interactions = ratings.merge(movies[["movieId", "domain"]], on="movieId", how="left").rename(
        columns={"userId": "user_id", "movieId": "item_id"}
    )
    interactions["title"] = interactions["item_id"].map(items.set_index("item_id")["title"])
    interactions["split"] = "train"
    return interactions[["user_id", "item_id", "rating", "timestamp", "domain", "split"]], items


def _preprocess_amazon(raw_root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for filename, domain in [("Books.csv", "books"), ("Movies_and_TV.csv", "movies")]:
        csv_path = raw_root / filename
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, header=None, names=["user_id", "item_id", "rating", "timestamp"])
        frame["domain"] = domain
        frame["split"] = "train"
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No Amazon CSV files available after download.")
    interactions = pd.concat(frames, ignore_index=True)
    items = (
        interactions[["item_id", "domain"]]
        .drop_duplicates()
        .assign(title=lambda df: df["item_id"], description=lambda df: df["item_id"])
        [["item_id", "title", "domain", "description"]]
    )
    return interactions[["user_id", "item_id", "rating", "timestamp", "domain", "split"]], items


def apply_cross_domain_filter(interactions: pd.DataFrame, dataset: str, min_interactions_per_user: int) -> pd.DataFrame:
    """Keep only users with sufficient interactions and cross-domain coverage."""
    domain_counts = interactions.groupby("user_id")["domain"].nunique()
    if dataset == "amazon":
        keep_users = domain_counts[domain_counts >= 2].index
    else:
        keep_users = domain_counts[domain_counts >= 2].index
    interactions = interactions[interactions["user_id"].isin(keep_users)].copy()
    support = interactions.groupby("user_id").size()
    keep_users = support[support >= min_interactions_per_user].index
    filtered = interactions[interactions["user_id"].isin(keep_users)].copy()
    LOGGER.info("Cross-domain filter kept %s users.", filtered["user_id"].nunique())
    return filtered


def limit_users_and_items(interactions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Limit users and items for a fast prototype run."""
    max_users = config["data"]["max_users"]
    max_items = config["data"]["max_items"]
    if interactions["user_id"].nunique() > max_users:
        keep_users = interactions["user_id"].drop_duplicates().head(max_users)
        interactions = interactions[interactions["user_id"].isin(keep_users)]
    item_counts = interactions["item_id"].value_counts().head(max_items)
    return interactions[interactions["item_id"].isin(item_counts.index)].copy()


def chronological_split(interactions: pd.DataFrame) -> pd.DataFrame:
    """Assign train/validation/test splits chronologically per user."""
    chunks = []
    for _, frame in interactions.groupby("user_id"):
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        n_rows = len(frame)
        if n_rows < 10:
            continue
        train_end = max(int(np.floor(n_rows * 0.8)), 1)
        valid_end = max(int(np.floor(n_rows * 0.9)), train_end + 1)
        frame.loc[:, "split"] = "test"
        frame.loc[: train_end - 1, "split"] = "train"
        frame.loc[train_end: valid_end - 1, "split"] = "validation"
        chunks.append(frame)
    return pd.concat(chunks, ignore_index=True) if chunks else interactions.iloc[0:0].copy()
