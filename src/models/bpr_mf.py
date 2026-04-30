"""Bayesian Personalized Ranking matrix factorization."""

from __future__ import annotations

import random

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class BPRMF(nn.Module):
    """Simple CPU-friendly BPR matrix factorization."""

    def __init__(self, num_users: int, num_items: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        nn.init.normal_(self.user_emb.weight, std=0.05)
        nn.init.normal_(self.item_emb.weight, std=0.05)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        return (self.user_emb(user_idx) * self.item_emb(item_idx)).sum(dim=1)


def train_bpr_mf(interactions: pd.DataFrame, config: dict) -> tuple[BPRMF, dict[str, int], dict[str, int]]:
    """Train BPR-MF on positive train interactions."""
    train_cfg = {"latent_dim": 32, "lr": 0.01, "weight_decay": 1e-4, "epochs": 20, "batch_size": 256}
    train_cfg.update(config.get("bpr_mf", {}))
    positives = interactions[(interactions["split"] == "train") & (interactions["rating"] >= 4.0)]
    user_ids = sorted(map(str, positives["user_id"].unique()))
    item_ids = sorted(map(str, positives["item_id"].unique()))
    user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    triples: list[tuple[int, int, int]] = []
    item_pool = list(item_to_idx.values())
    seen = positives.groupby("user_id")["item_id"].apply(lambda s: set(map(str, s))).to_dict()
    rng = random.Random(config["project"]["seed"])
    for row in positives.itertuples(index=False):
        user_id = str(row.user_id)
        pos_item = str(row.item_id)
        negatives = [idx for item, idx in item_to_idx.items() if item not in seen[row.user_id]]
        if not negatives:
            continue
        triples.append((user_to_idx[user_id], item_to_idx[pos_item], rng.choice(negatives)))
    model = BPRMF(max(len(user_to_idx), 1), max(len(item_to_idx), 1), train_cfg["latent_dim"])
    if not triples:
        return model, user_to_idx, item_to_idx
    dataset = TensorDataset(torch.tensor(triples, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    for _ in range(train_cfg["epochs"]):
        for (batch,) in loader:
            user_idx, pos_idx, neg_idx = batch[:, 0], batch[:, 1], batch[:, 2]
            pos_score = model(user_idx, pos_idx)
            neg_score = model(user_idx, neg_idx)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, user_to_idx, item_to_idx


def score_bpr_candidates(model: BPRMF, user_to_idx: dict[str, int], item_to_idx: dict[str, int], user_id: str, candidates: list[str]) -> list[tuple[str, float]]:
    """Score a candidate set with a trained BPR model."""
    if user_id not in user_to_idx:
        return [(item_id, 0.0) for item_id in candidates]
    user_idx = torch.tensor([user_to_idx[user_id]] * len(candidates), dtype=torch.long)
    item_idx = torch.tensor([item_to_idx.get(item_id, 0) for item_id in candidates], dtype=torch.long)
    with torch.no_grad():
        scores = model(user_idx, item_idx).tolist()
    return sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)
