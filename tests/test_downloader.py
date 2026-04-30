from pathlib import Path

from src.data.downloader import DatasetDownloader


def test_movielens_downloader_returns_directory(tmp_path: Path) -> None:
    downloader = DatasetDownloader(tmp_path)
    root = downloader.download_movielens()
    assert root.exists()
    assert (root / "ratings.csv").exists()


def test_movielens_downloader_uses_cache(tmp_path: Path) -> None:
    downloader = DatasetDownloader(tmp_path)
    root = downloader.download_movielens()
    first_mtime = (root / "ratings.csv").stat().st_mtime
    downloader.download_movielens()
    second_mtime = (root / "ratings.csv").stat().st_mtime
    assert first_mtime == second_mtime


def test_amazon_downloader_gracefully_fails(tmp_path: Path) -> None:
    downloader = DatasetDownloader(tmp_path)
    downloader.AMAZON_URLS = {"Books": "https://invalid.example.com/missing.csv.gz"}
    root = downloader.download_amazon()
    assert root.exists()
