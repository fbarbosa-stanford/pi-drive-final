"""GRPO utilities for π₀.5 driving post-training."""

from grpo.advantages import compute_grpo_advantages, summarize_group
from grpo.flow_grpo import FlowGRPOConfig, FlowGRPOTrainer, FlowGRPORankingResult
from grpo.label_cache import LabelCache, LabelRecord
from grpo.objective import (
    FlowGRPOInputs,
    FlowGRPOOutputs,
    compute_flow_grpo_loss,
    flow_losses_to_log_probs,
    resolve_log_probs,
)

__all__ = [
    "FlowGRPOConfig",
    "FlowGRPOTrainer",
    "FlowGRPORankingResult",
    "FlowGRPOInputs",
    "FlowGRPOOutputs",
    "LabelCache",
    "LabelRecord",
    "compute_flow_grpo_loss",
    "flow_losses_to_log_probs",
    "resolve_log_probs",
    "compute_grpo_advantages",
    "summarize_group",
]
