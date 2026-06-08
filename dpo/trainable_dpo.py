"""Trainable DPO gradient step (reuses the GRPO trainable core).

Mirrors ``dpo/objective.py`` but differentiable in the policy params via openpi's
``model.compute_loss`` (flow-MSE surrogate for log π, same as the GRPO path).

Pairing trick: ``compute_loss`` draws its own noise + time from the rng it is given.
Calling it for ``chosen`` and ``rejected`` with the **same rng** and identical action
shapes (1, ah, ad) yields identical internal noise/time — so the chosen-vs-rejected
flow comparison is paired/variance-reduced (the Diffusion-DPO trick) without touching
model internals. The frozen BC reference (``make_ref_logp``) is evaluated with that same
rng so the π_ref term subtracts cleanly.

Build state / save checkpoints with ``grpo.trainable_state`` (shared core)::

    from grpo.trainable_state import build_train_state, make_sampler, save_checkpoint
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Any, Callable


def make_ref_logp(config: Any, *, flow_log_prob_scale: float = 1.0) -> Callable:
    """Jitted reference log-probs (no grad): ``(logp_w, logp_l) = ref(ref_state, rng, obs1, chosen, rejected)``.

    ``ref_state`` is the frozen initial (BC) TrainState; pass the SAME rng used by the
    DPO step so noise/time match between the current policy and the reference.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    scale = max(float(flow_log_prob_scale), 1e-8)

    def _ref(ref_state, rng, obs1, chosen, rejected):
        model = nnx.merge(ref_state.model_def, ref_state.params)
        model.eval()
        flow_w = jnp.mean(model.compute_loss(rng, obs1, chosen, train=False))
        flow_l = jnp.mean(model.compute_loss(rng, obs1, rejected, train=False))
        return -flow_w / scale, -flow_l / scale

    return jax.jit(_ref)


def make_dpo_train_step(
    config: Any,
    *,
    beta: float = 30.0,
    flow_log_prob_scale: float = 1.0,
    anchor_lambda: float = 0.0,
) -> Callable:
    """Build a jitted **Diffusion-DPO** gradient step (flow-MSE log-prob surrogate).

    Signature::

        new_state, metrics = train_step(state, rng, obs1, chosen, rejected, ref_logp_w, ref_logp_l)

    ``obs1`` is a batch-1 ``Observation``; ``chosen``/``rejected`` are (1, ah, ad)
    normalized action chunks; ``ref_logp_w``/``ref_logp_l`` are scalar reference
    log-probs from ``make_ref_logp``.

    Loss (Bradley-Terry / Diffusion-DPO; log π ≈ −flow_loss/scale)::

        logits = β·((log π_w − log π_l) − (log π_ref_w − log π_ref_l))
               = β·((flow_l − flow_l_ref) − (flow_w − flow_w_ref))   # at scale=1
        L = softplus(−logits)

    **β is the effective Diffusion-DPO coefficient (the β·T·ω factor), NOT a 0.1-style
    temperature.** It must be large enough that the sigmoid SATURATES after a small flow
    margin — otherwise the loss never bounds and the policy keeps pushing the rejected
    action's flow-MSE up (degrading the shared flow field), which is what collapsed the
    earlier β=0.1 run. With meaned flow-MSE (~0.1), β≈30 saturates at margin ~0.15.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    scale = max(float(flow_log_prob_scale), 1e-8)
    beta = float(beta)
    anchor_lambda = float(anchor_lambda)

    def _step(config, state, rng, obs1, chosen, rejected, ref_logp_w, ref_logp_l):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, obs1, chosen, rejected):
            # Same rng + same shape -> identical internal noise/time (paired comparison).
            flow_w = jnp.mean(model.compute_loss(rng, obs1, chosen, train=False))
            flow_l = jnp.mean(model.compute_loss(rng, obs1, rejected, train=False))
            logp_w = -flow_w / scale
            logp_l = -flow_l / scale
            logits = beta * ((logp_w - logp_l) - (ref_logp_w - ref_logp_l))
            # DPO+NLL anchor (RPO / DPOP-style): also keep the CHOSEN action's flow-MSE low.
            # Pure flow-DPO optimizes only the *margin*, and because log π ≈ −flow_MSE is
            # UNNORMALIZED, it grows the margin by inflating the rejected action's flow-MSE,
            # which degrades the shared flow field and drags flow_chosen up too -> collapse.
            # The +λ·flow_chosen term pins flow_chosen at BC level so the field can't run away.
            loss = jax.nn.softplus(-logits) + anchor_lambda * flow_w
            aux = (logp_w, logp_l, logits)
            return loss, aux

        # Use the same rng as make_ref_logp (no fold_in) so π_ref subtraction is paired.
        diff_state = nnx.DiffState(0, config.trainable_filter)
        (loss, (logp_w, logp_l, logits)), grads = nnx.value_and_grad(
            loss_fn, argnums=diff_state, has_aux=True
        )(model, rng, obs1, chosen, rejected)

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = dataclasses.replace(
            state, step=state.step + 1, params=new_params, opt_state=new_opt_state
        )
        if state.ema_decay is not None:
            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params,
                    new_params,
                ),
            )

        metrics = {
            "loss": loss,
            "logits": logits,  # saturates (|logits| large) once the margin is learned
            "implicit_reward": logp_w - logp_l,  # log π_w − log π_l
            "flow_chosen": -logp_w * scale,       # want this DOWN (bounded ≥0)
            "flow_rejected": -logp_l * scale,     # watch: must stay BOUNDED, not explode
            "accuracy": (logp_w > logp_l).astype(jnp.float32),
            "grad_norm": optax.global_norm(grads),
        }
        return new_state, metrics

    return jax.jit(functools.partial(_step, config))
