"""Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


class BPRMF:
    """Small NumPy BPR-MF implementation for CPU-only prototype runs."""

    def __init__(self, num_users: int, num_items: int, latent_dim: int = 32, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.user_emb = rng.normal(0.0, 0.05, size=(max(num_users, 1), latent_dim))
        self.item_emb = rng.normal(0.0, 0.05, size=(max(num_items, 1), latent_dim))

    def score(self, user_idx: int, item_idx: int) -> float:
        """Dot-product score."""
        return float(np.dot(self.user_emb[user_idx], self.item_emb[item_idx]))


def train_bpr_mf(interactions: pd.DataFrame, config: dict):
    """Train BPR-MF on positive train interactions with NumPy SGD."""
    train_cfg = {"latent_dim": 32, "lr": 0.01, "weight_decay": 1e-4, "epochs": 20, "batch_size": 256}
    train_cfg.update(config.get("bpr_mf", {}))
    positives = interactions[(interactions["split"] == "train") & (interactions["rating"] >= 4.0)]
    user_ids = sorted(map(str, positives["user_id"].unique()))
    item_ids = sorted(map(str, positives["item_id"].unique()))
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    model = BPRMF(len(user_to_idx), len(item_to_idx), train_cfg["latent_dim"], config["project"]["seed"])
    if positives.empty:
        return model, user_to_idx, item_to_idx

    seen = positives.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
    item_pool = list(item_to_idx)
    rng = random.Random(config["project"]["seed"])
    triples: list[tuple[int, int, int]] = []
    for row in positives.itertuples(index=False):
        user_id = str(row.user_id)
        pos_item = str(row.item_id)
        negatives = [item for item in item_pool if item not in seen[row.user_id]]
        if not negatives:
            continue
        neg_item = rng.choice(negatives)
        triples.append((user_to_idx[user_id], item_to_idx[pos_item], item_to_idx[neg_item]))
    if not triples:
        return model, user_to_idx, item_to_idx

    lr = train_cfg["lr"]
    reg = train_cfg["weight_decay"]
    for _ in range(train_cfg["epochs"]):
        rng.shuffle(triples)
        for user_idx, pos_idx, neg_idx in triples:
            user_vec = model.user_emb[user_idx]
            pos_vec = model.item_emb[pos_idx]
            neg_vec = model.item_emb[neg_idx]
            x_uij = float(np.dot(user_vec, pos_vec - neg_vec))
            sigmoid = 1.0 / (1.0 + np.exp(x_uij))
            user_grad = sigmoid * (pos_vec - neg_vec) - reg * user_vec
            pos_grad = sigmoid * user_vec - reg * pos_vec
            neg_grad = -sigmoid * user_vec - reg * neg_vec
            model.user_emb[user_idx] += lr * user_grad
            model.item_emb[pos_idx] += lr * pos_grad
            model.item_emb[neg_idx] += lr * neg_grad
    return model, user_to_idx, item_to_idx


def score_bpr_candidates(model: BPRMF, user_to_idx: dict[str, int], item_to_idx: dict[str, int], user_id: str, candidates: list[str]) -> list[tuple[str, float]]:
    """Score a candidate set with the trained NumPy BPR model."""
    if user_id not in user_to_idx:
        return [(item_id, 0.0) for item_id in candidates]
    user_idx = user_to_idx[user_id]
    scored = [(item_id, model.score(user_idx, item_to_idx.get(item_id, 0))) for item_id in candidates]
    return sorted(scored, key=lambda pair: pair[1], reverse=True)
