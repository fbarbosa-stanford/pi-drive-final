"""CoC tags ↔ trajectory meta-action consistency."""

from __future__ import annotations

import re

import numpy as np

from rewards.action_space import actions_to_xyz

# Alpamayo-style intended behaviors from parsed CoC tags
INTENDED_SLOW = frozenset({"must_yield", "must_slow", "must_stop", "yield", "stop", "slow"})
INTENDED_LEFT = frozenset({"turn_left", "merge_left", "lane_change_left", "bear_left"})
INTENDED_RIGHT = frozenset({"turn_right", "merge_right", "lane_change_right", "bear_right"})
INTENDED_STRAIGHT = frozenset({"continue_straight", "drive_forward", "maintain_lane", "straight"})

_COC_TAG_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("must_yield", re.compile(r"\b(yield|must yield|give way)\b", re.I)),
    ("must_slow", re.compile(r"\b(slow|decelerat|reduce speed|brake)\b", re.I)),
    ("must_stop", re.compile(r"\b(stop|halt|stand still)\b", re.I)),
    ("turn_left", re.compile(r"\b(turn left|left turn|merge left|lane change left)\b", re.I)),
    ("turn_right", re.compile(r"\b(turn right|right turn|merge right|lane change right)\b", re.I)),
    ("continue_straight", re.compile(r"\b(straight|maintain lane|continue|proceed straight)\b", re.I)),
    ("pedestrian", re.compile(r"\b(pedestrian|crosswalk|walker|cyclist)\b", re.I)),
]


def parse_coc_tags(coc_tags: list[str] | None = None, coc_text: str = "") -> set[str]:
    """Normalize CoC into intended-behavior tags."""
    found: set[str] = set()
    if coc_tags:
        for t in coc_tags:
            found.add(str(t).strip().lower().replace(" ", "_"))
    if coc_text:
        for name, pat in _COC_TAG_PATTERNS:
            if pat.search(coc_text):
                found.add(name)
    return found


def trajectory_to_meta_actions(
    candidate_traj: np.ndarray,
    *,
    dt: float = 0.1,
    initial_speed: float = 0.0,
    initial_yaw: float = 0.0,
) -> set[str]:
    """Infer meta-actions from accel/curvature trajectory."""
    actions = np.asarray(candidate_traj, dtype=np.float64)
    meta: set[str] = set()

    mean_accel = float(np.mean(actions[:, 0]))
    mean_curv = float(np.mean(actions[:, 1]))
    end_accel = float(actions[-1, 0]) if actions.shape[0] else 0.0

    xyz = actions_to_xyz(
        actions,
        action_format="accel_curvature",
        dt=dt,
        initial_speed=initial_speed,
        initial_yaw=initial_yaw,
    )
    if xyz.shape[0] > 1:
        speeds = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / dt
        speed_delta = float(speeds[-1] - speeds[0]) if speeds.size > 1 else 0.0
        mean_speed = float(np.mean(speeds))
    else:
        speed_delta = 0.0
        mean_speed = initial_speed

    if mean_accel < -0.2 or end_accel < -0.3 or speed_delta < -0.5:
        meta.add("decelerating")
    if mean_accel > 0.15 and speed_delta > 0.2:
        meta.add("accelerating")
    if mean_speed < 1.5:
        meta.add("crawling")

    if mean_curv > 0.008:
        meta.add("turning_left")
    elif mean_curv < -0.008:
        meta.add("turning_right")
    else:
        meta.add("going_straight")

    return meta


def meta_actions_match(
    meta_actions: set[str],
    intended_tags: set[str],
) -> float:
    """Soft consistency score in [0, 1] between trajectory meta-actions and CoC tags."""
    if not intended_tags:
        return 0.5

    checks: list[bool] = []

    if intended_tags & INTENDED_SLOW:
        checks.append("decelerating" in meta_actions or "crawling" in meta_actions)
    if intended_tags & INTENDED_LEFT:
        checks.append("turning_left" in meta_actions)
    if intended_tags & INTENDED_RIGHT:
        checks.append("turning_right" in meta_actions)
    if intended_tags & INTENDED_STRAIGHT:
        checks.append("going_straight" in meta_actions and "decelerating" not in meta_actions)

    if not checks:
        # Generic keyword overlap fallback
        overlap = bool(
            (intended_tags & INTENDED_LEFT and "turning_left" in meta_actions)
            or (intended_tags & INTENDED_RIGHT and "turning_right" in meta_actions)
            or (intended_tags & INTENDED_STRAIGHT and "going_straight" in meta_actions)
        )
        return 1.0 if overlap else 0.5

    return float(sum(checks) / len(checks))
