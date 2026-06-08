"""Pick preferred / dispreferred trajectories using Alpamayo (AR1) references.

Lower dispreference score = closer to AR1 expert modes (preferred for DPO).
Uses cached ``LabelRecord`` expert actions/xyz or ``RewardContext.ar1_trajs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rewards.action_space import actions_to_xyz
from rewards.composite_reward import nearest_mode_ade
from rewards.flat_actions import unflatten_actions
from rewards.reward_context import RewardContext
from rewards.trajectory_metrics import ade


@dataclass(frozen=True)
class AlpamayoPreferenceConfig:
    """How candidates are scored against AR1 references."""

    dt: float = 0.1
    tie_eps_m: float = 1e-4
    # Require this ADE gap (m) between winner and loser; else no pair
    min_margin_m: float = 0.0


@dataclass
class PreferencePair:
    """Indices into the candidate list for one DPO step."""

    chosen_idx: int
    rejected_idx: int
    chosen_score: float
    rejected_score: float
    margin_m: float
    all_scores: list[float] = field(default_factory=list)
    source: str = "ar1_min_ade"


def score_candidate_vs_ar1(
    candidate: np.ndarray,
    *,
    context: RewardContext,
    expert_xyz: np.ndarray | None = None,
) -> float:
    """Dispreference score: lower = better match to AR1 (min ADE in meters)."""
    traj = np.asarray(candidate, dtype=np.float32)
    if traj.ndim == 1:
        traj = unflatten_actions(traj)

    speed = context.initial_speed
    yaw = context.initial_yaw
    dt = context.dt

    if expert_xyz is not None:
        ref = np.asarray(expert_xyz, dtype=np.float32)
        pred_xyz = actions_to_xyz(
            traj,
            action_format="accel_curvature",
            dt=dt,
            initial_speed=speed,
            initial_yaw=yaw,
        )
        ref_xy = ref[:, :2] if ref.shape[-1] >= 2 else ref
        return float(ade(pred_xyz, ref_xy))

    if context.ar1_trajs is not None and context.ar1_trajs.size > 0:
        return float(
            nearest_mode_ade(
                traj,
                context.ar1_trajs,
                dt=dt,
                initial_speed=speed,
                initial_yaw=yaw,
            )
        )

    raise ValueError(
        "Alpamayo preference needs expert_xyz or RewardContext.ar1_trajs "
        "(generate labels with modal AR1 or pass LabelRecord)."
    )


def pick_preference_pair(
    candidates: list[np.ndarray],
    *,
    context: RewardContext,
    expert_xyz: np.ndarray | None = None,
    config: AlpamayoPreferenceConfig | None = None,
) -> PreferencePair | None:
    """AR1 picks best (min ADE) vs worst (max ADE) in the group."""
    if len(candidates) < 2:
        return None

    cfg = config or AlpamayoPreferenceConfig()
    scores = [
        score_candidate_vs_ar1(c, context=context, expert_xyz=expert_xyz)
        for c in candidates
    ]
    chosen_idx = int(np.argmin(scores))
    rejected_idx = int(np.argmax(scores))
    if chosen_idx == rejected_idx:
        return None

    margin = scores[rejected_idx] - scores[chosen_idx]
    if margin < cfg.min_margin_m:
        return None

    return PreferencePair(
        chosen_idx=chosen_idx,
        rejected_idx=rejected_idx,
        chosen_score=float(scores[chosen_idx]),
        rejected_score=float(scores[rejected_idx]),
        margin_m=float(margin),
        all_scores=[float(s) for s in scores],
        source="ar1_expert_xyz" if expert_xyz is not None else "ar1_modes_min_ade",
    )


def preference_from_label_record(
    candidates: list[np.ndarray],
    record,
    *,
    config: AlpamayoPreferenceConfig | None = None,
) -> PreferencePair | None:
    """Build RewardContext from AR1 ``LabelRecord`` and pick the pair."""
    from dpo.context_builders import context_from_label_record

    ctx = context_from_label_record(record)
    expert_xyz = None
    if record.expert_xyz is not None:
        expert_xyz = np.asarray(record.expert_xyz, dtype=np.float32)
    return pick_preference_pair(
        candidates,
        context=ctx,
        expert_xyz=expert_xyz,
        config=config,
    )
