"""Heuristic Chain-of-Causation consistency scoring (Tier 2 ranker)."""

from __future__ import annotations

import re

# Keywords grouped by maneuver class for lightweight consistency checks.
_MANEUVER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "slow": ("slow", "decelerate", "reduce speed", "yield", "stop", "brake", "caution"),
    "turn_left": ("turn left", "left turn", "merge left", "steer left", "lane change left"),
    "turn_right": ("turn right", "right turn", "merge right", "steer right", "lane change right"),
    "straight": ("proceed straight", "continue straight", "maintain lane", "keep lane", "follow lane"),
    "pedestrian": ("pedestrian", "crosswalk", "walker", "cyclist", "vulnerable"),
    "vehicle": ("vehicle", "car", "truck", "traffic", "lead vehicle", "following"),
}


def infer_maneuver_tags(
    *,
    mean_speed: float,
    max_curvature: float,
    speed_delta: float,
) -> set[str]:
    """Infer coarse maneuver tags from integrated trajectory statistics."""
    tags: set[str] = set()
    if speed_delta < -0.5 or mean_speed < 2.0:
        tags.add("slow")
    if max_curvature > 0.02:
        tags.add("turn_left")  # sign handled separately if needed
    elif max_curvature > 0.005:
        tags.add("turn_right")
    else:
        tags.add("straight")
    return tags


def infer_maneuver_from_actions(actions, action_format: str, dt: float = 0.1) -> set[str]:
    """Infer maneuver tags from raw actions."""
    import numpy as np

    from rewards.action_space import actions_to_xyz

    actions = np.asarray(actions, dtype=np.float64)
    if action_format == "accel_curvature":
        mean_accel = float(np.mean(actions[:, 0]))
        max_curv = float(np.max(np.abs(actions[:, 1])))
        xyz = actions_to_xyz(actions, action_format="accel_curvature", dt=dt)
        speed = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1) / dt if xyz.shape[0] > 1 else np.array([0.0])
        mean_speed = float(np.mean(speed)) if speed.size else 0.0
        speed_delta = float(speed[-1] - speed[0]) if speed.size > 1 else 0.0
        tags = infer_maneuver_tags(mean_speed=mean_speed, max_curvature=max_curv, speed_delta=speed_delta)
        if mean_accel < -0.3:
            tags.add("slow")
        if max_curv > 0.01 and float(np.mean(actions[:, 1])) > 0:
            tags.add("turn_left")
        elif max_curv > 0.01:
            tags.add("turn_right")
        return tags

    # ego_delta: use speed from forward component
    forward = actions[:, 0]
    mean_speed = float(np.mean(forward) / dt)
    speed_delta = float((forward[-1] - forward[0]) / dt) if forward.size > 1 else 0.0
    lateral = np.abs(actions[:, 1]) if actions.shape[1] > 1 else np.zeros_like(forward)
    max_lat = float(np.max(lateral))
    max_curv = max_lat / max(mean_speed, 0.5)
    return infer_maneuver_tags(mean_speed=mean_speed, max_curvature=max_curv, speed_delta=speed_delta)


def coc_consistency_score(coc_text: str, maneuver_tags: set[str]) -> float:
    """Score in [0, 1]: does CoC text mention factors consistent with the trajectory?"""
    if not coc_text or not coc_text.strip():
        return 0.5  # neutral when no reasoning available

    text = coc_text.lower()
    if not maneuver_tags:
        return 0.5

    hits = 0
    checks = 0
    for tag in maneuver_tags:
        keywords = _MANEUVER_KEYWORDS.get(tag, ())
        if not keywords:
            continue
        checks += 1
        if any(kw in text for kw in keywords):
            hits += 1

    if checks == 0:
        return 0.5
    return float(hits / checks)


def extract_coc_entities(coc_text: str) -> set[str]:
    """Extract coarse entity tokens from CoC for debugging."""
    text = coc_text.lower()
    found: set[str] = set()
    for _maneuver, keywords in _MANEUVER_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                found.add(kw)
    # strip punctuation tokens
    return {re.sub(r"[^a-z0-9 ]", "", w).strip() for w in found if w.strip()}
