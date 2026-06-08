"""Pure loss functions for the RLT actor-critic.

Kept free of any module/optimiser state so the TD-target arithmetic can be
unit-tested in isolation against hand-computed boundary cases. The two subtle
calls live here:

  - **Chunked C-step TD target.** ``bootstrap`` and ``gamma_pow`` are taken
    *verbatim* from the batch (computed once in the buffer). The per-step reward
    sum is discounted ``Σ_k γ^k · rewards_k · discounts_k`` (k=0 is the first
    step, undiscounted); the bootstrap adds ``bootstrap · γ^n · min_i Q'_i``.

  - **Reference-dropout asymmetry.** Ref-dropout (zeroing ã to the actor input)
    is applied **only** in the policy loss, to force the actor to learn an
    independent pathway rather than copy ã. It is **never** applied when forming
    the TD target's next action a': the deployed actor always sees ã, so the
    value backup must too. The BC anchor always pulls toward the *true* ã.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import RLTConfig
from .nets import GaussianActor, QEnsemble
from .transition import Batch


def _discounted_reward_sum(rewards: Tensor, discounts: Tensor, gamma: float) -> Tensor:
    """Σ_k γ^k · rewards_k · discounts_k  over the C step axis -> (B,)."""
    C = rewards.shape[1]
    powers = gamma ** torch.arange(C, dtype=rewards.dtype, device=rewards.device)
    return (rewards * discounts * powers).sum(dim=1)


def _ref_dropout(a_ref: Tensor, p: float, training: bool) -> Tensor:
    """Zero the whole reference chunk per-sample w.p. ``p`` (policy input only)."""
    if not training or p <= 0.0:
        return a_ref
    keep = (torch.rand(a_ref.shape[0], 1, 1, device=a_ref.device) >= p).to(a_ref.dtype)
    return a_ref * keep


@torch.no_grad()
def critic_td_target(
    batch: Batch,
    target_actor: GaussianActor,
    target_q: QEnsemble,
    cfg: RLTConfig,
) -> Tensor:
    """y = Σ_k γ^k r_k·d_k + bootstrap·γ^n·min_i Q'_i(x', a'),  a' ~ π_target(x', ã').

    No reference dropout on a' -- the value backup must match the deployed
    policy, which always sees ã'.
    """
    disc_r = _discounted_reward_sum(batch["rewards"], batch["discounts"], cfg.gamma)
    a_next = target_actor.rsample(batch["z_rl_next"], batch["s_p_next"], batch["a_ref_next"])
    min_q_next = target_q.min_q(batch["z_rl_next"], batch["s_p_next"], a_next)
    return disc_r + batch["bootstrap"] * batch["gamma_pow"] * min_q_next


def critic_loss(
    batch: Batch,
    online_q: QEnsemble,
    target_actor: GaussianActor,
    target_q: QEnsemble,
    cfg: RLTConfig,
) -> Tensor:
    """Mean-squared TD error summed across the ensemble (each critic regresses
    the same min-of-N target)."""
    y = critic_td_target(batch, target_actor, target_q, cfg)  # (B,)
    q = online_q(batch["z_rl"], batch["s_p"], batch["a"])      # (n_critics, B)
    return ((q - y.unsqueeze(0)) ** 2).mean()


def policy_loss(
    batch: Batch,
    actor: GaussianActor,
    online_q: QEnsemble,
    cfg: RLTConfig,
) -> Tensor:
    """RLT policy objective:  L = mean(-min_i Q_i(x, a) + β·||a − ã||²),
    a = rsample(μ_θ(x, ã_in), σ),  ã_in ref-dropped,  BC anchor vs *true* ã."""
    a_ref_in = _ref_dropout(batch["a_ref"], cfg.ref_dropout_p, actor.training)
    a = actor.rsample(batch["z_rl"], batch["s_p"], a_ref_in)
    q = online_q.min_q(batch["z_rl"], batch["s_p"], a)        # pessimistic (B,)
    bc = ((a - batch["a_ref"]) ** 2).flatten(1).mean(dim=1)   # anchor to true ã
    return (-q + cfg.beta * bc).mean()
