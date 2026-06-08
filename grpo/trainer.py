"""GRPO trainer skeleton (offline open-loop until BC checkpoint + Flow-SDE land)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grpo.advantages import compute_grpo_advantages, summarize_group
from grpo.sample_group import SampleGroupConfig, make_synthetic_group
from rewards.alpamayo_ranker import AlpamayoRanker, RankerConfig


@dataclass
class GRPOTrainerConfig:
    group_size: int = 12
    kl_coef: float = 0.01
    advantage_clip: float = 5.0
    ranker: RankerConfig = field(default_factory=RankerConfig)


@dataclass
class GRPOStepResult:
    rewards: list[float]
    advantages: np.ndarray
    summary: dict[str, float]
    rank_results: list


class GRPOTrainer:
    """Offline GRPO step orchestrator (ranking + advantages).

    Policy gradient / Flow-SDE loss is intentionally stubbed — wire to LeRobot or
    RLinf once the BC checkpoint is available.
    """

    def __init__(self, config: GRPOTrainerConfig | None = None):
        self.config = config or GRPOTrainerConfig()
        self.ranker = AlpamayoRanker(self.config.ranker)

    def run_ranking_step(
        self,
        action_group: list[np.ndarray],
        *,
        expert_xyz=None,
        gt_xyz=None,
        coc_text: str | None = None,
        yaw_series=None,
    ) -> GRPOStepResult:
        rank_results = self.ranker.rank_group(
            action_group,
            expert_xyz=expert_xyz,
            gt_xyz=gt_xyz,
            coc_text=coc_text,
            yaw_series=yaw_series,
        )
        rewards = [r.reward for r in rank_results]
        advantages = compute_grpo_advantages(rewards)
        clip = self.config.advantage_clip
        advantages = np.clip(advantages, -clip, clip)
        summary = summarize_group(rewards, advantages)
        return GRPOStepResult(
            rewards=rewards,
            advantages=advantages,
            summary=summary,
            rank_results=rank_results,
        )

    def dry_run_synthetic(
        self,
        reference_actions: np.ndarray,
        *,
        gt_xyz=None,
        expert_xyz=None,
        coc_text: str | None = None,
        yaw_series=None,
    ) -> GRPOStepResult:
        group = make_synthetic_group(
            reference_actions,
            config=SampleGroupConfig(group_size=self.config.group_size),
        )
        return self.run_ranking_step(
            group,
            expert_xyz=expert_xyz,
            gt_xyz=gt_xyz,
            coc_text=coc_text,
            yaw_series=yaw_series,
        )

    def policy_loss_stub(self, advantages: np.ndarray, flow_losses: list[float]) -> float:
        """Deprecated — use ``grpo.objective.compute_flow_grpo_loss`` instead."""
        from grpo.objective import FlowGRPOInputs, compute_flow_grpo_loss

        out = compute_flow_grpo_loss(
            FlowGRPOInputs(advantages=advantages, flow_losses=flow_losses)
        )
        return out.loss
