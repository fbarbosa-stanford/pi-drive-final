"""PDMS-style driving sub-rewards for open-loop trajectory ranking."""

from __future__ import annotations

import numpy as np

from rewards.action_space import actions_to_xyz
from rewards.comfort import comfort_from_xyz


def _candidate_xyz(
    actions: np.ndarray,
    *,
    dt: float,
    initial_speed: float,
    initial_yaw: float,
) -> np.ndarray:
    return actions_to_xyz(
        actions,
        action_format="accel_curvature",
        dt=dt,
        initial_speed=initial_speed,
        initial_yaw=initial_yaw,
    )


def _arc_length_xy(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    diffs = np.diff(xyz[:, :2], axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def _lateral_offsets(xyz: np.ndarray, route: np.ndarray) -> np.ndarray:
    """Signed lateral distance from each waypoint to piecewise-linear route."""
    pts = xyz[:, :2]
    route = np.asarray(route, dtype=np.float64)
    if route.shape[0] < 2:
        origin = route[0] if route.size else np.zeros(2)
        seg = np.array([1.0, 0.0])
        rel = pts - origin
        lateral = -seg[0] * rel[:, 1] + seg[1] * rel[:, 0]
        return lateral.astype(np.float32)

    lateral = np.zeros(pts.shape[0], dtype=np.float64)
    for i, p in enumerate(pts):
        best = float("inf")
        best_lat = 0.0
        for j in range(route.shape[0] - 1):
            a, b = route[j], route[j + 1]
            ab = b - a
            denom = float(np.dot(ab, ab)) + 1e-9
            t = np.clip(float(np.dot(p - a, ab) / denom), 0.0, 1.0)
            proj = a + t * ab
            seg = ab / (np.linalg.norm(ab) + 1e-9)
            lat = -seg[0] * (p[1] - proj[1]) + seg[1] * (p[0] - proj[0])
            dist = float(np.linalg.norm(p - proj))
            if dist < best:
                best = dist
                best_lat = lat
        lateral[i] = best_lat
    return lateral.astype(np.float32)


def ego_progress_along_route(
    candidate_traj: np.ndarray,
    route_polyline: np.ndarray,
    *,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
) -> float:
    """Ego progress (EP): fraction of route arc length covered by horizon end."""
    xyz = _candidate_xyz(
        candidate_traj, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw
    )
    route = np.asarray(route_polyline, dtype=np.float64)
    traveled = _arc_length_xy(xyz)
    route_len = _arc_length_xy(
        np.column_stack([route[:, 0], route[:, 1], np.zeros(route.shape[0])])
    )
    if route_len < 1e-3:
        return 0.0
    return float(np.clip(traveled / route_len, 0.0, 1.0))


def drivable_area_compliance(
    candidate_traj: np.ndarray,
    *,
    route_polyline: np.ndarray,
    half_width_m: float,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
    strict_binary: bool = False,
) -> float:
    """DAC: 1 if trajectory stays inside drivable corridor, else 0 (or soft fraction)."""
    xyz = _candidate_xyz(
        candidate_traj, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw
    )
    lateral = np.abs(_lateral_offsets(xyz, route_polyline))
    inside = lateral <= half_width_m
    if strict_binary:
        return 1.0 if bool(np.all(inside)) else 0.0
    return float(np.mean(inside))


def time_to_collision(
    candidate_traj: np.ndarray,
    *,
    lead_vehicle_xy: np.ndarray,
    lead_vehicle_speed_mps: float = 0.0,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
    min_ttc_s: float = 0.5,
    safe_ttc_s: float = 4.0,
) -> float:
    """TTC margin in [0, 1]; higher is safer. Uses constant-velocity lead approximation."""
    xyz = _candidate_xyz(
        candidate_traj, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw
    )
    lead = np.asarray(lead_vehicle_xy, dtype=np.float64).reshape(2)
    ego_xy = xyz[:, :2]

    if ego_xy.shape[0] < 2:
        return 1.0

    ego_vel = np.diff(ego_xy, axis=0) / dt
    ego_speed = np.linalg.norm(ego_vel, axis=1)
    ego_heading = np.arctan2(ego_vel[:, 1], ego_vel[:, 0] + 1e-9)
    lead_dir = np.array([np.cos(0.0), np.sin(0.0)])  # assume lead along +x
    lead_v = lead_dir * lead_vehicle_speed_mps

    worst_ttc = safe_ttc_s
    for i in range(ego_xy.shape[0]):
        rel = lead - ego_xy[i]
        dist = float(np.linalg.norm(rel))
        if i < ego_vel.shape[0]:
            closing = float(
                np.dot(ego_vel[i] - lead_v, rel / (np.linalg.norm(rel) + 1e-9))
            )
        else:
            closing = 0.0
        if closing > 0.1:
            ttc = dist / closing
            worst_ttc = min(worst_ttc, ttc)

    if worst_ttc >= safe_ttc_s:
        return 1.0
    if worst_ttc <= min_ttc_s:
        return 0.0
    return float((worst_ttc - min_ttc_s) / (safe_ttc_s - min_ttc_s))


def comfort_score(
    candidate_traj: np.ndarray,
    *,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
) -> float:
    """Comfort in [0, 1] from jerk/accel/yaw-rate bounds."""
    xyz = _candidate_xyz(
        candidate_traj, dt=dt, initial_speed=initial_speed, initial_yaw=initial_yaw
    )
    return float(comfort_from_xyz(xyz, dt=dt)["score"])


def pdms_driving_reward(
    candidate_traj: np.ndarray,
    context,
    *,
    use_dac_gate: bool = True,
) -> tuple[float, dict[str, float]]:
    """Composite PDMS-style term: dac * (5*ttc + 2*comfort + 5*progress) / 12."""
    route = context.route_polyline
    if route is None:
        route = np.array([[0.0, 0.0], [50.0, 0.0]], dtype=np.float32)

    r_progress = ego_progress_along_route(
        candidate_traj,
        route,
        dt=context.dt,
        initial_speed=context.initial_speed,
        initial_yaw=context.initial_yaw,
    )
    r_comfort = comfort_score(
        candidate_traj,
        dt=context.dt,
        initial_speed=context.initial_speed,
        initial_yaw=context.initial_yaw,
    )
    r_dac = drivable_area_compliance(
        candidate_traj,
        route_polyline=route,
        half_width_m=context.drivable_half_width_m,
        dt=context.dt,
        initial_speed=context.initial_speed,
        initial_yaw=context.initial_yaw,
        strict_binary=False,
    )

    if context.lead_vehicle_xy is not None:
        r_ttc = time_to_collision(
            candidate_traj,
            lead_vehicle_xy=context.lead_vehicle_xy,
            lead_vehicle_speed_mps=context.lead_vehicle_speed_mps,
            dt=context.dt,
            initial_speed=context.initial_speed,
            initial_yaw=context.initial_yaw,
        )
    else:
        r_ttc = 1.0

    inner = (5.0 * r_ttc + 2.0 * r_comfort + 5.0 * r_progress) / 12.0
    r_driving = (r_dac * inner) if use_dac_gate else inner

    return float(r_driving), {
        "progress": r_progress,
        "comfort": r_comfort,
        "dac": r_dac,
        "ttc": r_ttc,
        "inner": inner,
    }
