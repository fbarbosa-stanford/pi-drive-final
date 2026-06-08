"""Group-relative advantage computation for GRPO."""

from __future__ import annotations

import numpy as np


def compute_grpo_advantages(
    rewards: list[float] | np.ndarray,
    *,
    eps: float = 1e-8,
    normalize: bool = True,
) -> np.ndarray:
    """Compute GRPO advantages: (r_i - mean) / (std + eps).

    Returns shape (G,). When std ≈ 0 (all rewards equal), returns zeros.
    """
    r = np.asarray(rewards, dtype=np.float64)
    if r.size == 0:
        return r
    mean = float(np.mean(r))
    adv = r - mean
    if normalize:
        std = float(np.std(r))
        if std > eps:
            adv = adv / (std + eps)
        else:
            adv = np.zeros_like(adv)
    return adv.astype(np.float32)


def summarize_group(rewards: list[float], advantages: np.ndarray) -> dict[str, float]:
    r = np.asarray(rewards, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    best = int(np.argmax(r)) if r.size else -1
    worst = int(np.argmin(r)) if r.size else -1
    return {
        "group_size": float(r.size),
        "reward_mean": float(np.mean(r)) if r.size else 0.0,
        "reward_std": float(np.std(r)) if r.size else 0.0,
        "reward_min": float(np.min(r)) if r.size else 0.0,
        "reward_max": float(np.max(r)) if r.size else 0.0,
        "advantage_std": float(np.std(adv)) if adv.size else 0.0,
        "best_index": float(best),
        "worst_index": float(worst),
    }
