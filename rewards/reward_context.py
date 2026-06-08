"""Per-sample context for composite GRPO rewards."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RewardContext:
    """Cached labels and scene geometry for one (clip, t0) ranking step."""

    coc_tags: list[str] = field(default_factory=list)
    coc_text: str = ""
    # (K, T, 2) accel/curvature reference modes from AR1 (or GT stand-in for smoke)
    ar1_trajs: np.ndarray | None = None
    # Polyline in ego frame at t0: x forward, y left; shape (N, 2)
    route_polyline: np.ndarray | None = None
    # Half-width of drivable corridor around route centerline (meters)
    drivable_half_width_m: float = 6.0
    # Optional lead agent at t0 in ego frame (x, y) meters
    lead_vehicle_xy: np.ndarray | None = None
    lead_vehicle_speed_mps: float = 0.0
    initial_speed: float = 0.0
    initial_yaw: float = 0.0
    dt: float = 0.1

    def __post_init__(self) -> None:
        if self.ar1_trajs is not None:
            self.ar1_trajs = np.asarray(self.ar1_trajs, dtype=np.float32)
        if self.route_polyline is not None:
            self.route_polyline = np.asarray(self.route_polyline, dtype=np.float32)
        if self.lead_vehicle_xy is not None:
            self.lead_vehicle_xy = np.asarray(self.lead_vehicle_xy, dtype=np.float32).reshape(-1)[:2]
