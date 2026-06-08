"""Dataset index sampling for holdout eval (no openpi / lerobot imports)."""

from __future__ import annotations

import numpy as np


def pick_eval_indices(dataset_len: int, num_samples: int, seed: int = 42) -> list[int]:
    n = min(num_samples, dataset_len)
    rng = np.random.default_rng(seed)
    if n >= dataset_len:
        return list(range(dataset_len))
    return sorted(rng.choice(dataset_len, size=n, replace=False).tolist())


def pick_eval_holdout_indices(
    dataset_len: int,
    num_samples: int,
    seed: int = 42,
    val_ratio: float = 0.15,
) -> list[int]:
    start = int(dataset_len * (1.0 - val_ratio))
    pool = list(range(start, dataset_len))
    if not pool:
        return pick_eval_indices(dataset_len, num_samples, seed=seed)
    n = min(num_samples, len(pool))
    rng = np.random.default_rng(seed)
    if n >= len(pool):
        return pool
    return sorted(rng.choice(pool, size=n, replace=False).tolist())
