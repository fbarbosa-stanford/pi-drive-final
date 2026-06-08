"""Flow-GRPO policy objective (group-relative policy gradient + optional KL)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FlowGRPOInputs:
    """Inputs for one Flow-GRPO policy-gradient step."""

    advantages: np.ndarray  # (G,) group-relative advantages
    # log π(a_i | o) from Flow-SDE / exact density — preferred when available
    log_probs: np.ndarray | None = None
    # Per-sample flow-matching MSE; converted to log-prob surrogate if log_probs is None
    flow_losses: list[float] | np.ndarray | None = None
    # Reference policy log-probs for KL(π || π_ref)
    ref_log_probs: np.ndarray | None = None
    kl_coef: float = 0.01
    # Scale for surrogate: log π ≈ -flow_loss / flow_log_prob_scale
    flow_log_prob_scale: float = 1.0
    # Detach advantages (treat as constant weights during backward)
    detach_advantages: bool = True


@dataclass
class FlowGRPOOutputs:
    loss: float
    metrics: dict[str, float]


def flow_losses_to_log_probs(
    flow_losses: np.ndarray | list[float],
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Gaussian surrogate: log π(a|o) ∝ -flow_matching_mse / scale.

    Lower flow loss → higher log-prob. Replace with Flow-SDE path log-density when wired.
    """
    losses = np.asarray(flow_losses, dtype=np.float64).reshape(-1)
    denom = max(float(scale), 1e-8)
    return (-losses / denom).astype(np.float64)


def resolve_log_probs(inputs: FlowGRPOInputs) -> np.ndarray:
    """Resolve per-sample log π(a|o), from explicit log_probs or flow_loss surrogate."""
    adv = np.asarray(inputs.advantages, dtype=np.float64).reshape(-1)
    g = adv.size

    if inputs.log_probs is not None:
        log_p = np.asarray(inputs.log_probs, dtype=np.float64).reshape(-1)
    elif inputs.flow_losses is not None:
        log_p = flow_losses_to_log_probs(
            inputs.flow_losses, scale=inputs.flow_log_prob_scale
        )
    else:
        raise ValueError(
            "FlowGRPOInputs requires log_probs or flow_losses "
            "(wire Flow-SDE log-density or per-sample flow MSE from the policy)."
        )

    if log_p.size != g:
        raise ValueError(f"log_probs length {log_p.size} != group size {g}")
    return log_p


def compute_flow_grpo_loss(inputs: FlowGRPOInputs) -> FlowGRPOOutputs:
    """Flow-GRPO loss: -E[A · log π] + kl_coef · E[log π - log π_ref].

    Advantages are group-normalized rewards (see ``grpo.advantages``). Minimize this scalar
    during training; higher-advantage samples get up-weighted when their log-prob is increased.

    With only ``flow_losses``, uses the flow-MSE surrogate for log π (see ``flow_losses_to_log_probs``).
    """
    adv = np.asarray(inputs.advantages, dtype=np.float64).reshape(-1)
    if inputs.detach_advantages:
        adv = adv.copy()  # caller should stop_grad on advantages in JAX/torch; numpy is already detached

    log_p = resolve_log_probs(inputs)

    # Policy-gradient term (GRPO group baseline already in advantages)
    pg_loss = -float(np.mean(adv * log_p))

    kl_loss = 0.0
    if inputs.ref_log_probs is not None:
        ref = np.asarray(inputs.ref_log_probs, dtype=np.float64).reshape(-1)
        if ref.size != log_p.size:
            raise ValueError(f"ref_log_probs length {ref.size} != group size {log_p.size}")
        kl_loss = float(inputs.kl_coef * np.mean(log_p - ref))

    loss = pg_loss + kl_loss

    metrics = {
        "loss": loss,
        "pg_loss": pg_loss,
        "kl_loss": kl_loss,
        "log_prob_mean": float(np.mean(log_p)),
        "log_prob_std": float(np.std(log_p)),
        "advantage_mean": float(np.mean(adv)),
        "used_flow_surrogate": 1.0 if inputs.log_probs is None else 0.0,
    }
    return FlowGRPOOutputs(loss=loss, metrics=metrics)
