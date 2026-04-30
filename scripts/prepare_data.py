"""Prepare datasets end-to-end."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.downloader import DatasetDownloader
from src.data.preprocess import preprocess_dataset
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.utils.seed import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    config = load_config(args.config, {"data": {"dataset": args.dataset}} if args.dataset else None)
    set_global_seed(config["project"]["seed"])
    downloader = DatasetDownloader(ROOT / "data" / "raw")
    raw_root = downloader.download(config["data"]["dataset"])
    interactions, items = preprocess_dataset(raw_root, config)
    logger = logging.getLogger(__name__)
    logger.info(
        "Prepared dataset with %s users, %s items, %s interactions, %s cross-domain users.",
        interactions["user_id"].nunique(),
        interactions["item_id"].nunique(),
        len(interactions),
        interactions["user_id"].nunique(),
    )


if __name__ == "__main__":
    main()
