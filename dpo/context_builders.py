"""RewardContext helpers for DPO (AR1 label cache + smoke tests)."""

from __future__ import annotations

import numpy as np

from rewards.reward_context import RewardContext


def make_straight_route(
    length_m: float = 80.0,
    num_points: int = 20,
) -> np.ndarray:
    xs = np.linspace(0.0, length_m, num_points, dtype=np.float32)
    ys = np.zeros_like(xs)
    return np.stack([xs, ys], axis=1)


def context_from_label_record(record) -> RewardContext:
    """``LabelRecord`` from AR1 labeling → ranking context."""
    ar1 = None
    if record.expert_actions is not None:
        ar1 = np.asarray(record.expert_actions, dtype=np.float32)
        if ar1.ndim == 2:
            ar1 = ar1[np.newaxis, ...]
    return RewardContext(
        coc_tags=[],
        coc_text=record.coc_text,
        ar1_trajs=ar1,
        route_polyline=make_straight_route(),
        initial_speed=record.initial_speed,
        initial_yaw=record.initial_yaw,
    )


def make_smoke_context(
    reference_traj: np.ndarray,
    *,
    num_ar1_modes: int = 3,
    initial_speed: float = 8.0,
) -> RewardContext:
    """Synthetic AR1 modes around a reference traj (local smoke only)."""
    ref = np.asarray(reference_traj, dtype=np.float32)
    rng = np.random.default_rng(42)
    modes = [ref.copy()]
    for i in range(1, num_ar1_modes):
        noise = rng.normal(0, 0.05, size=ref.shape).astype(np.float32)
        modes.append(ref + noise * np.array([0.02, 0.003], dtype=np.float32) * (i + 1))
    return RewardContext(
        coc_tags=["continue_straight"],
        coc_text="continue straight",
        ar1_trajs=np.stack(modes, axis=0),
        route_polyline=make_straight_route(),
        initial_speed=initial_speed,
        initial_yaw=0.0,
    )
