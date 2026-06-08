"""Unified Alpamayo-backed trajectory ranker for GRPO."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rewards.action_space import ActionFormat, actions_to_xyz
from rewards.coc_consistency import coc_consistency_score, infer_maneuver_from_actions
from rewards.comfort import comfort_from_xyz, comfort_penalty
from rewards.trajectory_metrics import ade, fde


@dataclass(frozen=True)
class RankerConfig:
    """Weights and gates for scalar trajectory reward."""

    w_expert_ade: float = 0.4
    w_gt_ade: float = 0.3
    w_comfort: float = 0.2
    w_coc: float = 0.1
    ade_threshold_m: float = 3.0
    ade_gate_penalty: float = -1.0
    action_format: ActionFormat = "ego_delta"
    dt: float = 0.1


@dataclass
class RankResult:
    reward: float
    ade_expert: float
    ade_gt: float
    fde_gt: float
    comfort_score: float
    coc_score: float
    gated: bool
    details: dict[str, float] = field(default_factory=dict)


class AlpamayoRanker:
    """Score candidate trajectories against cached AR1/GT labels."""

    def __init__(self, config: RankerConfig | None = None):
        self.config = config or RankerConfig()

    def score(
        self,
        actions: np.ndarray,
        *,
        expert_xyz: np.ndarray | None = None,
        gt_xyz: np.ndarray | None = None,
        coc_text: str | None = None,
        initial_speed: float = 0.0,
        initial_yaw: float = 0.0,
        yaw_series: np.ndarray | None = None,
    ) -> RankResult:
        cfg = self.config
        pred_xyz = actions_to_xyz(
            actions,
            action_format=cfg.action_format,
            dt=cfg.dt,
            initial_speed=initial_speed,
            initial_yaw=initial_yaw,
            yaw_series=yaw_series,
        )

        ade_expert = ade(pred_xyz, expert_xyz) if expert_xyz is not None else float("nan")
        ade_gt = ade(pred_xyz, gt_xyz) if gt_xyz is not None else float("nan")
        fde_gt = fde(pred_xyz, gt_xyz) if gt_xyz is not None else float("nan")

        comfort = comfort_from_xyz(pred_xyz, dt=cfg.dt)
        comfort_term = comfort_penalty(comfort)

        maneuver_tags = infer_maneuver_from_actions(actions, cfg.action_format, dt=cfg.dt)
        coc_score = coc_consistency_score(coc_text or "", maneuver_tags)

        # Primary gate: use best available ADE reference
        ref_ade = ade_gt if gt_xyz is not None else ade_expert
        gated = ref_ade >= cfg.ade_threshold_m

        if gated:
            reward = cfg.ade_gate_penalty
        else:
            expert_term = 0.0 if np.isnan(ade_expert) else -(ade_expert / cfg.ade_threshold_m)
            gt_term = 0.0 if np.isnan(ade_gt) else -(ade_gt / cfg.ade_threshold_m)
            reward = (
                cfg.w_expert_ade * expert_term
                + cfg.w_gt_ade * gt_term
                + cfg.w_comfort * comfort_term
                + cfg.w_coc * coc_score
            )

        return RankResult(
            reward=float(reward),
            ade_expert=float(ade_expert),
            ade_gt=float(ade_gt),
            fde_gt=float(fde_gt),
            comfort_score=float(comfort["score"]),
            coc_score=float(coc_score),
            gated=gated,
            details={
                "expert_term": 0.0 if np.isnan(ade_expert) else -(ade_expert / cfg.ade_threshold_m),
                "gt_term": 0.0 if np.isnan(ade_gt) else -(ade_gt / cfg.ade_threshold_m),
                "comfort_term": comfort_term,
            },
        )

    def rank_group(
        self,
        action_group: list[np.ndarray],
        **label_kwargs,
    ) -> list[RankResult]:
        return [self.score(actions, **label_kwargs) for actions in action_group]
