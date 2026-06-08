"""Flow-GRPO orchestration: sample group → composite reward → advantages.

Policy objective lives in ``grpo/objective.py`` (not implemented here).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grpo.advantages import compute_grpo_advantages, summarize_group
from grpo.objective import FlowGRPOInputs, FlowGRPOOutputs, compute_flow_grpo_loss
from grpo.sample_group import SampleGroupConfig, make_synthetic_group
from rewards.composite_reward import (
    CompositeRewardConfig,
    RewardBreakdown,
    compute_rewards_for_group,
)
from rewards.reward_context import RewardContext


@dataclass
class FlowGRPOConfig:
    group_size: int = 12
    advantage_clip: float = 5.0
    reward: CompositeRewardConfig = field(default_factory=CompositeRewardConfig)
    sample: SampleGroupConfig = field(default_factory=SampleGroupConfig)
    kl_coef: float = 0.01


@dataclass
class FlowGRPORankingResult:
    """Output of one ranking step (no policy gradient)."""

    candidates: list[np.ndarray]
    breakdowns: list[RewardBreakdown]
    rewards: list[float]
    advantages: np.ndarray
    summary: dict[str, float]


class FlowGRPOTrainer:
    """Sample candidates, rank with composite reward, compute advantages."""

    def __init__(self, config: FlowGRPOConfig | None = None):
        self.config = config or FlowGRPOConfig()

    def run_composite_ranking_step(
        self,
        candidates: list[np.ndarray],
        context: RewardContext,
    ) -> FlowGRPORankingResult:
        breakdowns = compute_rewards_for_group(
            candidates, context, self.config.reward
        )
        rewards = [b.total for b in breakdowns]
        advantages = compute_grpo_advantages(rewards)
        advantages = np.clip(advantages, -self.config.advantage_clip, self.config.advantage_clip)
        summary = summarize_group(rewards, advantages)
        return FlowGRPORankingResult(
            candidates=candidates,
            breakdowns=breakdowns,
            rewards=rewards,
            advantages=advantages,
            summary=summary,
        )

    def dry_run_synthetic(
        self,
        reference_traj: np.ndarray,
        context: RewardContext,
    ) -> FlowGRPORankingResult:
        cfg = self.config.sample
        cfg = SampleGroupConfig(
            group_size=self.config.group_size,
            noise_scale=cfg.noise_scale,
            seed=cfg.seed,
        )
        group = make_synthetic_group(reference_traj, config=cfg)
        return self.run_composite_ranking_step(group, context)

    def build_objective_inputs(
        self,
        ranking: FlowGRPORankingResult,
        *,
        log_probs: np.ndarray | None = None,
        flow_losses: list[float] | None = None,
        ref_log_probs: np.ndarray | None = None,
    ) -> FlowGRPOInputs:
        """Package ranking output for ``compute_flow_grpo_loss`` (your implementation)."""
        return FlowGRPOInputs(
            advantages=ranking.advantages,
            log_probs=log_probs,
            flow_losses=flow_losses,
            ref_log_probs=ref_log_probs,
            kl_coef=self.config.kl_coef,
        )

    def run_policy_step(
        self,
        ranking: FlowGRPORankingResult,
        *,
        log_probs: np.ndarray,
        ref_log_probs: np.ndarray | None = None,
    ) -> FlowGRPOOutputs:
        """Call after you have log_probs — delegates to ``grpo/objective.py``."""
        inputs = self.build_objective_inputs(
            ranking, log_probs=log_probs, ref_log_probs=ref_log_probs
        )
        return compute_flow_grpo_loss(inputs)
