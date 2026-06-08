"""DPO objective: AR1-chosen trajectory vs rejected (flow log-prob surrogate)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DPOInputs:
    """One pairwise preference step."""

    log_prob_chosen: float
    log_prob_rejected: float
    ref_log_prob_chosen: float | None = None
    ref_log_prob_rejected: float | None = None
    beta: float = 0.1


@dataclass
class DPOOutputs:
    loss: float
    metrics: dict[str, float]


def flow_loss_to_log_prob(flow_loss: float, *, scale: float = 1.0) -> float:
    """Surrogate log π(a|o) ∝ -flow_matching_mse / scale."""
    denom = max(float(scale), 1e-8)
    return float(-float(flow_loss) / denom)


def compute_dpo_loss(inputs: DPOInputs) -> DPOOutputs:
    """Bradley-Terry DPO loss with optional reference policy (π_ref).

    L = -log σ(β · ((log π_w - log π_l) - (log π_ref_w - log π_ref_l)))
    """
    beta = float(inputs.beta)
    log_w = float(inputs.log_prob_chosen)
    log_l = float(inputs.log_prob_rejected)

    logits = beta * (log_w - log_l)
    if inputs.ref_log_prob_chosen is not None and inputs.ref_log_prob_rejected is not None:
        logits -= beta * (
            float(inputs.ref_log_prob_chosen) - float(inputs.ref_log_prob_rejected)
        )

    # Stable -log sigmoid: softplus(-logits)
    loss = float(np.log1p(np.exp(-logits)))

    implicit_reward = log_w - log_l
    metrics = {
        "loss": loss,
        "dpo_logits": float(logits),
        "implicit_reward": float(implicit_reward),
        "log_prob_chosen": log_w,
        "log_prob_rejected": log_l,
        "accuracy": 1.0 if log_w > log_l else 0.0,
    }
    if inputs.ref_log_prob_chosen is not None:
        metrics["ref_implicit_reward"] = float(
            inputs.ref_log_prob_chosen - inputs.ref_log_prob_rejected
        )
    return DPOOutputs(loss=loss, metrics=metrics)
