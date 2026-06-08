"""Action ↔ trajectory conversions for GRPO scoring."""

from __future__ import annotations

from typing import Literal

import numpy as np

ActionFormat = Literal["ego_delta", "accel_curvature"]


def integrate_accel_curvature(
    actions: np.ndarray,
    *,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
    initial_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Integrate (accel, curvature) actions into ego-frame XYZ waypoints.

    Matches the unicycle-style layout used by Alpamayo AR1 pseudo-labels.
    actions: (T, 2) with columns [acceleration m/s², curvature rad/m].
    Returns: (T, 3) XYZ in ego frame at t0 (x forward, y left, z up).
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] < 2:
        raise ValueError(f"expected (T, 2) accel/curvature actions, got {actions.shape}")

    t_steps = actions.shape[0]
    xyz = np.zeros((t_steps, 3), dtype=np.float64)
    x, y, z = initial_xyz
    v = float(initial_speed)
    yaw = float(initial_yaw)

    for i in range(t_steps):
        accel = float(actions[i, 0])
        curvature = float(actions[i, 1])
        v_next = max(v + accel * dt, 0.0)
        v_mid = 0.5 * (v + v_next)
        yaw_next = yaw + v_mid * curvature * dt

        yaw_mid = 0.5 * (yaw + yaw_next)
        dx = v_mid * np.cos(yaw_mid) * dt
        dy = v_mid * np.sin(yaw_mid) * dt
        x += dx
        y += dy
        # z stays 0 unless a third action dim is added later
        xyz[i] = (x, y, z)
        v, yaw = v_next, yaw_next

    return xyz.astype(np.float32)


def actions_to_xyz(
    actions: np.ndarray,
    *,
    action_format: ActionFormat,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
    yaw_series: np.ndarray | None = None,
) -> np.ndarray:
    """Convert policy actions to XYZ waypoints for metric computation."""
    actions = np.asarray(actions, dtype=np.float64)
    if action_format == "accel_curvature":
        return integrate_accel_curvature(
            actions,
            dt=dt,
            initial_speed=initial_speed,
            initial_yaw=initial_yaw,
        )

    if action_format == "ego_delta":
        from rewards.trajectory_metrics import integrate_ego_deltas

        if yaw_series is None:
            yaw_series = np.zeros(actions.shape[0], dtype=np.float64)
        return integrate_ego_deltas(actions, yaw_series, dt=dt)

    raise ValueError(f"unknown action_format: {action_format!r}")
