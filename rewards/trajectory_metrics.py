"""Geometric trajectory metrics (ADE, FDE, integration)."""

from __future__ import annotations

import numpy as np


def integrate_ego_deltas(
    deltas: np.ndarray,
    yaw_rad: np.ndarray,
    *,
    dt: float = 0.1,
) -> np.ndarray:
    """Integrate ego-frame (dx, dy, dz) deltas into world-frame XYZ path.

    deltas: (T, 3) — forward/lateral/vertical step in ego frame at each step.
    yaw_rad: (T,) — heading at each step (world frame).
    Returns: (T, 3) cumulative position in world frame starting at origin.
    """
    deltas = np.asarray(deltas, dtype=np.float64)
    yaw_rad = np.asarray(yaw_rad, dtype=np.float64)
    if deltas.ndim != 2 or deltas.shape[1] < 2:
        raise ValueError(f"expected (T, >=2) deltas, got {deltas.shape}")
    if yaw_rad.shape[0] != deltas.shape[0]:
        raise ValueError("yaw_rad length must match number of delta steps")

    t_steps = deltas.shape[0]
    xyz = np.zeros((t_steps, 3), dtype=np.float64)
    x, y, z = 0.0, 0.0, 0.0

    for i in range(t_steps):
        dx_e, dy_e = float(deltas[i, 0]), float(deltas[i, 1])
        dz = float(deltas[i, 2]) if deltas.shape[1] > 2 else 0.0
        yaw = float(yaw_rad[i])
        c, s = np.cos(yaw), np.sin(yaw)
        wx = c * dx_e - s * dy_e
        wy = s * dx_e + c * dy_e
        x += wx
        y += wy
        z += dz
        xyz[i] = (x, y, z)

    return xyz.astype(np.float32)


def ade(pred_xyz: np.ndarray, ref_xyz: np.ndarray, *, xy_only: bool = True) -> float:
    """Average Displacement Error between two waypoint sequences."""
    pred = np.asarray(pred_xyz, dtype=np.float64)
    ref = np.asarray(ref_xyz, dtype=np.float64)
    n = min(pred.shape[0], ref.shape[0])
    if n == 0:
        return float("inf")
    pred = pred[:n]
    ref = ref[:n]
    if xy_only:
        pred = pred[:, :2]
        ref = ref[:, :2]
    return float(np.mean(np.linalg.norm(pred - ref, axis=-1)))


def fde(pred_xyz: np.ndarray, ref_xyz: np.ndarray, *, xy_only: bool = True) -> float:
    """Final Displacement Error (last matched waypoint)."""
    pred = np.asarray(pred_xyz, dtype=np.float64)
    ref = np.asarray(ref_xyz, dtype=np.float64)
    n = min(pred.shape[0], ref.shape[0])
    if n == 0:
        return float("inf")
    p = pred[n - 1, :2] if xy_only else pred[n - 1]
    r = ref[n - 1, :2] if xy_only else ref[n - 1]
    return float(np.linalg.norm(p - r))
