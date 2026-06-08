"""Sample groups of trajectories for GRPO ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SampleGroupConfig:
    group_size: int = 12
    noise_scale: float = 0.15  # for synthetic perturbation smoke tests
    seed: int = 0


def perturb_actions(reference: np.ndarray, rng: np.random.Generator, noise_scale: float) -> np.ndarray:
    """Create a synthetic candidate by adding scaled noise to a reference trajectory."""
    ref = np.asarray(reference, dtype=np.float32)
    scale = np.std(ref, axis=0, keepdims=True) + 1e-3
    noise = rng.normal(0.0, 1.0, size=ref.shape).astype(np.float32)
    return ref + noise_scale * scale * noise


def make_synthetic_group(
    reference: np.ndarray,
    *,
    config: SampleGroupConfig | None = None,
) -> list[np.ndarray]:
    """Build G trajectories around a reference for ranker smoke tests."""
    cfg = config or SampleGroupConfig()
    rng = np.random.default_rng(cfg.seed)
    ref = np.asarray(reference, dtype=np.float32)
    group = [ref.copy()]
    while len(group) < cfg.group_size:
        group.append(perturb_actions(ref, rng, cfg.noise_scale))
    return group[: cfg.group_size]


def sample_policy_group(policy, batch: dict, *, group_size: int = 12) -> list[np.ndarray]:
    """Sample G action chunks from a LeRobot PI05Policy (requires torch + lerobot).

    Uses different RNG seeds per sample. Flow-SDE sampling should be wired here
    once RL training starts; for now uses standard predict_action_chunk.
    """
    import torch

    actions: list[np.ndarray] = []
    for i in range(group_size):
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(i)
        with torch.no_grad():
            pred = policy.predict_action_chunk(batch)
        actions.append(pred[0].detach().cpu().numpy().astype(np.float32))
    return actions
