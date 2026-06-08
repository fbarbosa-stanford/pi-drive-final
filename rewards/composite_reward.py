"""Composite GRPO ranking reward (PDMS + CoC consistency + AR1 guardrail)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rewards.action_space import actions_to_xyz
from rewards.meta_actions import (
    meta_actions_match,
    parse_coc_tags,
    trajectory_to_meta_actions,
)
from rewards.pdms_terms import pdms_driving_reward
from rewards.reward_context import RewardContext
from rewards.trajectory_metrics import ade


@dataclass(frozen=True)
class CompositeRewardConfig:
    w_driving: float = 0.5
    w_consistency: float = 0.3
    w_ref: float = 0.2
    ref_ade_threshold_m: float = 3.0
    use_dac_gate: bool = True


@dataclass
class RewardBreakdown:
    total: float
    r_driving: float
    r_consistency: float
    r_ref_clipped: float
    driving_terms: dict[str, float] = field(default_factory=dict)
    meta_actions: set[str] = field(default_factory=set)
    intended_tags: set[str] = field(default_factory=set)
    min_ade_ar1: float = float("nan")


def nearest_mode_ade(
    candidate_traj: np.ndarray,
    ar1_trajs: np.ndarray,
    *,
    dt: float,
    initial_speed: float,
    initial_yaw: float,
) -> float:
    """Minimum ADE to any cached AR1 mode (accel/curvature integrated to XYZ)."""
    ar1_trajs = np.asarray(ar1_trajs, dtype=np.float32)
    if ar1_trajs.ndim == 2:
        ar1_trajs = ar1_trajs[np.newaxis, ...]

    pred_xyz = actions_to_xyz(
        candidate_traj,
        action_format="accel_curvature",
        dt=dt,
        initial_speed=initial_speed,
        initial_yaw=initial_yaw,
    )

    best = float("inf")
    for k in range(ar1_trajs.shape[0]):
        ref_xyz = actions_to_xyz(
            ar1_trajs[k],
            action_format="accel_curvature",
            dt=dt,
            initial_speed=initial_speed,
            initial_yaw=initial_yaw,
        )
        best = min(best, ade(pred_xyz, ref_xyz))
    return best


def compute_reward(
    candidate_traj: np.ndarray,
    context: RewardContext,
    config: CompositeRewardConfig | None = None,
) -> RewardBreakdown:
    """Scalar reward for one candidate trajectory.

    Combines PDMS-style driving quality, CoC/meta-action consistency, and
    nearest-mode AR1 guardrail (clipped).
    """
    cfg = config or CompositeRewardConfig()
    actions = np.asarray(candidate_traj, dtype=np.float32)

    r_driving, driving_terms = pdms_driving_reward(
        actions, context, use_dac_gate=cfg.use_dac_gate
    )

    meta = trajectory_to_meta_actions(
        actions,
        dt=context.dt,
        initial_speed=context.initial_speed,
        initial_yaw=context.initial_yaw,
    )
    intended = parse_coc_tags(context.coc_tags, context.coc_text)
    r_consistency = meta_actions_match(meta, intended)

    if context.ar1_trajs is not None and context.ar1_trajs.size > 0:
        min_ade = nearest_mode_ade(
            actions,
            context.ar1_trajs,
            dt=context.dt,
            initial_speed=context.initial_speed,
            initial_yaw=context.initial_yaw,
        )
        r_ref = -min_ade
        r_ref_clipped = max(r_ref, -cfg.ref_ade_threshold_m)
    else:
        min_ade = float("nan")
        r_ref_clipped = 0.0

    total = (
        cfg.w_driving * r_driving
        + cfg.w_consistency * r_consistency
        + cfg.w_ref * r_ref_clipped
    )

    return RewardBreakdown(
        total=float(total),
        r_driving=float(r_driving),
        r_consistency=float(r_consistency),
        r_ref_clipped=float(r_ref_clipped),
        driving_terms=driving_terms,
        meta_actions=meta,
        intended_tags=intended,
        min_ade_ar1=min_ade,
    )


def compute_rewards_for_group(
    candidates: list[np.ndarray],
    context: RewardContext,
    config: CompositeRewardConfig | None = None,
) -> list[RewardBreakdown]:
    return [compute_reward(c, context, config) for c in candidates]
