"""Build RewardContext for smoke tests and (later) AR1 label cache."""

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


def make_synthetic_smoke_context(
    reference_traj: np.ndarray,
    *,
    horizon: int | None = None,
    coc_tags: list[str] | None = None,
    coc_text: str = "",
    num_ar1_modes: int = 3,
    lead_vehicle_xy: tuple[float, float] | None = (25.0, 0.0),
    lead_vehicle_speed_mps: float = 5.0,
    initial_speed: float = 8.0,
    initial_yaw: float = 0.0,
    drivable_half_width_m: float = 6.0,
    dt: float = 0.1,
) -> RewardContext:
    """Synthetic context for composite-reward smoke tests."""
    ref = np.asarray(reference_traj, dtype=np.float32)
    if horizon is not None:
        ref = ref[:horizon]

    rng = np.random.default_rng(42)
    modes = [ref.copy()]
    for i in range(1, num_ar1_modes):
        noise = rng.normal(0, 0.05, size=ref.shape).astype(np.float32)
        scale = np.array([0.02, 0.003], dtype=np.float32)
        modes.append(ref + noise * scale * (i + 1))
    ar1_trajs = np.stack(modes, axis=0)

    tags = coc_tags or ["continue_straight"]
    lead = None if lead_vehicle_xy is None else np.array(lead_vehicle_xy, dtype=np.float32)

    return RewardContext(
        coc_tags=tags,
        coc_text=coc_text,
        ar1_trajs=ar1_trajs,
        route_polyline=make_straight_route(),
        drivable_half_width_m=drivable_half_width_m,
        lead_vehicle_xy=lead,
        lead_vehicle_speed_mps=lead_vehicle_speed_mps,
        initial_speed=initial_speed,
        initial_yaw=initial_yaw,
        dt=dt,
    )


def context_from_label_record(record) -> RewardContext:
    """Convert LabelRecord → RewardContext when AR1 cache is wired."""
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
