"""Comfort metrics inspired by Alpamayo RL aggregated_reward."""

from __future__ import annotations

import numpy as np


def _fraction_in_bounds(values: np.ndarray, low: float, high: float) -> float:
    if values.size == 0:
        return 0.0
    ok = (values >= low) & (values <= high)
    return float(np.mean(ok))


def comfort_from_xyz(
    xyz: np.ndarray,
    *,
    dt: float = 0.1,
    max_accel: float = 3.0,
    max_jerk: float = 5.0,
    max_yaw_rate: float = 0.5,
) -> dict[str, float]:
    """Compute comfort sub-scores from a waypoint path.

    Returns dict with keys in [0, 1] (higher = more comfortable) plus ``score`` mean.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.shape[0] < 3:
        return {"accel": 0.0, "jerk": 0.0, "yaw_rate": 0.0, "score": 0.0}

    vel = np.diff(xyz[:, :2], axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    accel = np.diff(speed) / dt
    jerk = np.diff(accel) / dt if accel.size > 1 else np.array([])

    heading = np.arctan2(vel[:, 1], vel[:, 0] + 1e-9)
    yaw_rate = np.diff(heading) / dt
    yaw_rate = (yaw_rate + np.pi) % (2 * np.pi) - np.pi

    accel_score = _fraction_in_bounds(accel, -max_accel, max_accel)
    jerk_score = _fraction_in_bounds(jerk, -max_jerk, max_jerk) if jerk.size else 1.0
    yaw_score = _fraction_in_bounds(yaw_rate, -max_yaw_rate, max_yaw_rate)

    score = float(np.mean([accel_score, jerk_score, yaw_score]))
    return {
        "accel": accel_score,
        "jerk": jerk_score,
        "yaw_rate": yaw_score,
        "score": score,
    }


def comfort_penalty(comfort: dict[str, float]) -> float:
    """Map comfort dict to Alpamayo-style penalty in [-1, 0]."""
    return float(comfort["score"] - 1.0)
