"""DPO training step: sample group → AR1 preference pair → DPO loss."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dpo.alpamayo_preference import (
    AlpamayoPreferenceConfig,
    PreferencePair,
    pick_preference_pair,
    preference_from_label_record,
)
from dpo.objective import DPOInputs, DPOOutputs, compute_dpo_loss, flow_loss_to_log_prob
from rewards.reward_context import RewardContext


@dataclass
class DPOConfig:
    group_size: int = 8
    preference: AlpamayoPreferenceConfig = field(default_factory=AlpamayoPreferenceConfig)
    beta: float = 0.1
    flow_log_prob_scale: float = 1.0


@dataclass
class DPORankingResult:
    pair: PreferencePair
    context: RewardContext
    candidates: list[np.ndarray]


class DPOTrainer:
    """Alpamayo (AR1) picks winner/loser; objective in ``dpo/objective.py``."""

    def __init__(self, config: DPOConfig | None = None):
        self.config = config or DPOConfig()

    def select_pair(
        self,
        candidates: list[np.ndarray],
        context: RewardContext,
        *,
        expert_xyz: np.ndarray | None = None,
        label_record=None,
    ) -> PreferencePair | None:
        if label_record is not None:
            return preference_from_label_record(
                candidates, label_record, config=self.config.preference
            )
        return pick_preference_pair(
            candidates,
            context=context,
            expert_xyz=expert_xyz,
            config=self.config.preference,
        )

    def build_objective_inputs(
        self,
        pair: PreferencePair,
        *,
        flow_loss_chosen: float,
        flow_loss_rejected: float,
        log_prob_chosen: float | None = None,
        log_prob_rejected: float | None = None,
        ref_flow_loss_chosen: float | None = None,
        ref_flow_loss_rejected: float | None = None,
    ) -> DPOInputs:
        scale = self.config.flow_log_prob_scale
        lp_w = (
            log_prob_chosen
            if log_prob_chosen is not None
            else flow_loss_to_log_prob(flow_loss_chosen, scale=scale)
        )
        lp_l = (
            log_prob_rejected
            if log_prob_rejected is not None
            else flow_loss_to_log_prob(flow_loss_rejected, scale=scale)
        )
        ref_w = (
            flow_loss_to_log_prob(ref_flow_loss_chosen, scale=scale)
            if ref_flow_loss_chosen is not None
            else None
        )
        ref_l = (
            flow_loss_to_log_prob(ref_flow_loss_rejected, scale=scale)
            if ref_flow_loss_rejected is not None
            else None
        )
        return DPOInputs(
            log_prob_chosen=lp_w,
            log_prob_rejected=lp_l,
            ref_log_prob_chosen=ref_w,
            ref_log_prob_rejected=ref_l,
            beta=self.config.beta,
        )

    def run_preference_step(
        self,
        candidates: list[np.ndarray],
        context: RewardContext,
        flow_losses: list[float],
        *,
        label_record=None,
        expert_xyz: np.ndarray | None = None,
    ) -> tuple[DPORankingResult, DPOOutputs] | None:
        pair = self.select_pair(
            candidates,
            context,
            expert_xyz=expert_xyz,
            label_record=label_record,
        )
        if pair is None:
            return None

        inputs = self.build_objective_inputs(
            pair,
            flow_loss_chosen=flow_losses[pair.chosen_idx],
            flow_loss_rejected=flow_losses[pair.rejected_idx],
        )
        out = compute_dpo_loss(inputs)
        ranking = DPORankingResult(pair=pair, context=context, candidates=candidates)
        return ranking, out
