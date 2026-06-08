"""Open-loop ADE / action MSE for (64,2) accel/curvature trajectories."""

from __future__ import annotations

import numpy as np

from rewards.action_space import actions_to_xyz
from rewards.flat_actions import flatten_actions, unflatten_actions
from rewards.trajectory_metrics import ade, fde


def action_mse(pred_chunk: np.ndarray, ref_chunk: np.ndarray) -> float:
    """MSE in action space — pred/ref shape (64, 2) or flat (128,)."""
    p = np.asarray(pred_chunk, dtype=np.float64).reshape(-1)
    r = np.asarray(ref_chunk, dtype=np.float64).reshape(-1)
    n = min(p.size, r.size)
    if n == 0:
        return float("inf")
    return float(np.mean((p[:n] - r[:n]) ** 2))


def trajectory_ade_fde(
    pred_chunk: np.ndarray,
    ref_chunk: np.ndarray,
    *,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
) -> dict[str, float]:
    """ADE/FDE after integrating accel/curvature to XYZ."""
    pred = unflatten_actions(pred_chunk) if np.asarray(pred_chunk).size == 128 else pred_chunk
    ref = unflatten_actions(ref_chunk) if np.asarray(ref_chunk).size == 128 else ref_chunk
    pred_xyz = actions_to_xyz(
        pred, action_format="accel_curvature", dt=dt,
        initial_speed=initial_speed, initial_yaw=initial_yaw,
    )
    ref_xyz = actions_to_xyz(
        ref, action_format="accel_curvature", dt=dt,
        initial_speed=initial_speed, initial_yaw=initial_yaw,
    )
    return {
        "ade_m": ade(pred_xyz, ref_xyz),
        "fde_m": fde(pred_xyz, ref_xyz),
        "action_mse": action_mse(pred, ref),
    }


def eval_candidate_vs_references(
    pred_flat: np.ndarray,
    *,
    gt_flat: np.ndarray | None = None,
    ar1_chunks: list[np.ndarray] | None = None,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
) -> dict[str, float]:
    """Metrics dict for one policy sample vs GT and/or AR1 modes."""
    pred = unflatten_actions(pred_flat)
    out: dict[str, float] = {}

    if gt_flat is not None:
        gt = unflatten_actions(gt_flat)
        m = trajectory_ade_fde(pred, gt, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw)
        out.update({f"gt_{k}": v for k, v in m.items()})

    if ar1_chunks:
        ades = []
        for ref in ar1_chunks:
            ref_u = unflatten_actions(ref) if np.asarray(ref).size == 128 else ref
            ades.append(
                trajectory_ade_fde(
                    pred, ref_u, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw
                )["ade_m"]
            )
        out["ar1_min_ade_m"] = float(min(ades))
        out["ar1_mean_ade_m"] = float(np.mean(ades))

    return out
