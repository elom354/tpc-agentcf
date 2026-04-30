"""Dataset download and synthetic fallback helpers."""

from __future__ import annotations

import gzip
import logging
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from src.utils.io import ensure_dir

LOGGER = logging.getLogger(__name__)


class DatasetDownloader:
    """Download and cache supported datasets."""

    MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    AMAZON_URLS = {
        "Books": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Books.csv.gz",
        "Movies_and_TV": "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/benchmark/5core/rating_only/Movies_and_TV.csv.gz",
    }

    def __init__(self, root_dir: str | Path = "data/raw", timeout: int = 60) -> None:
        self.root_dir = ensure_dir(root_dir)
        self.timeout = timeout

    def download(self, dataset: str) -> Path:
        """Download a supported dataset and return its root path."""
        if dataset == "movielens":
            return self.download_movielens()
        if dataset == "amazon":
            return self.download_amazon()
        raise ValueError(f"Unsupported dataset: {dataset}")

    def download_movielens(self) -> Path:
        """Download MovieLens Latest Small or create a synthetic fallback."""
        target_dir = self.root_dir / "ml-latest-small"
        ratings_path = target_dir / "ratings.csv"
        movies_path = target_dir / "movies.csv"
        if ratings_path.exists() and movies_path.exists():
            LOGGER.info("MovieLens already cached, skipping.")
            return target_dir
        zip_path = self.root_dir / "ml-latest-small.zip"
        ensure_dir(target_dir)
        try:
            self._download_file(self.MOVIELENS_URL, zip_path)
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(self.root_dir)
            zip_path.unlink(missing_ok=True)
            return target_dir
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("MovieLens download failed (%s). Using synthetic fallback.", exc)
            zip_path.unlink(missing_ok=True)
            self._write_synthetic_movielens(target_dir)
            return target_dir

    def download_amazon(self) -> Path:
        """Download Amazon Books and Movies rating-only files."""
        amazon_dir = ensure_dir(self.root_dir / "amazon")
        failures = 0
        for domain, url in self.AMAZON_URLS.items():
            csv_path = amazon_dir / f"{domain}.csv"
            if csv_path.exists():
                LOGGER.info("%s already cached, skipping.", csv_path.name)
                continue
            gz_path = amazon_dir / f"{domain}.csv.gz"
            try:
                self._download_file(url, gz_path)
                with gzip.open(gz_path, "rb") as src, csv_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                gz_path.unlink(missing_ok=True)
            except Exception as exc:
                failures += 1
                LOGGER.warning("Amazon download failed for %s (%s).", domain, exc)
                gz_path.unlink(missing_ok=True)
        if failures:
            LOGGER.warning("Falling back to MovieLens because Amazon download did not fully succeed.")
            return self.download_movielens()
        return amazon_dir

    def _download_file(self, url: str, target_path: Path) -> None:
        """Download a file with progress reporting."""
        response = requests.get(url, stream=True, timeout=self.timeout)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with target_path.open("wb") as handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {target_path.name}",
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 64):
                if not chunk:
                    continue
                handle.write(chunk)
                progress.update(len(chunk))

    def _write_synthetic_movielens(self, target_dir: Path) -> None:
        """Create a tiny deterministic fallback dataset."""
        ensure_dir(target_dir)
        movies = pd.DataFrame(
            [
                {"movieId": idx, "title": f"Movie {idx}", "genres": genre}
                for idx, genre in enumerate(
                    ["Drama", "Comedy", "Action", "Romance", "Thriller", "Comedy|Drama"] * 4,
                    start=1,
                )
            ]
        )
        rows: list[dict[str, int | float]] = []
        timestamp = 1_700_000_000
        for user_id in range(1, 9):
            for offset, movie_id in enumerate(range(1, 13)):
                rows.append(
                    {
                        "userId": user_id,
                        "movieId": ((movie_id + user_id - 1) % len(movies)) + 1,
                        "rating": 4.0 if offset % 3 != 0 else 5.0,
                        "timestamp": timestamp + user_id * 100 + offset,
                    }
                )
        pd.DataFrame(rows).to_csv(target_dir / "ratings.csv", index=False)
        movies.to_csv(target_dir / "movies.csv", index=False)
