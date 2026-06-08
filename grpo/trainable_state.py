"""Trainable openpi state for RL post-training (GRPO / DPO).

This is the piece the scaffolding was missing: a real, differentiable training
step over Mark's BC checkpoint. ``grpo/objective.py`` and ``dpo/objective.py``
compute losses in numpy from ``policy.infer()`` outputs — forward only, no
gradient reaches the weights. Here we instead:

  1. Build an openpi ``TrainState`` (params + optax opt_state) from the config +
     Mark's checkpoint, mirroring ``scripts/train.py::init_train_state`` minus FSDP
     (single-device H100).
  2. Sample candidate action chunks from the *current* policy (``sample_actions``).
  3. Run a jitted gradient step on ``model.compute_loss`` (per-sample flow-matching
     MSE) weighted by group advantages — the flow-MSE surrogate for log π
     (see ``grpo/objective.py``): log π(a|o) ≈ -flow_loss, so the GRPO policy
     gradient ``-E[A·log π]`` becomes ``E[A·flow_loss]``, differentiable in params.
  4. Save an orbax checkpoint loadable by ``policy_config.create_trained_policy``
     (same layout ``compare_fifty`` already restores).

Only the trainable filter (LoRA + action expert; see the ``pi05_driving``
``freeze_filter`` in ``openpi_patches/patch_openpi.py``) receives gradients.

All openpi/jax imports are function-local so this module is importable without
the openpi venv on sys.path (the runner prepends it first).
"""

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from typing import Any, Callable


def build_train_state(
    config: Any,
    *,
    seed: int = 0,
    peak_lr: float | None = None,
    warmup_steps: int | None = None,
    resume_from: str | None = None,
) -> tuple[Any, Any]:
    """Build a single-device openpi TrainState from ``config`` + its weight loader.

    Mirrors ``scripts/train.py::init_train_state`` but drops the FSDP mesh/sharding
    (one H100). ``config.weight_loader`` must already point at Mark's params dir
    (set via ``PI05_BC_CHECKPOINT_PARAMS`` in ``patch_openpi``); LoRA params absent
    from the checkpoint are initialized fresh, matching openpi's ``.*lora.*`` merge.

    Returns ``(train_state, tx)``.
    """
    import flax.nnx as nnx
    import flax.traverse_util as traverse_util
    import jax
    import jax.numpy as jnp

    import dataclasses as _dc

    import openpi.shared.array_typing as at
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.optimizer as _optimizer
    import openpi.training.utils as training_utils

    # The config LR (peak 3e-5) is the BC PRE-TRAINING rate — far too hot for RL/DPO
    # fine-tuning of this flow policy (it diverges even during warmup). Override with a
    # much lower peak + short warmup for stable post-training.
    lr_schedule = config.lr_schedule
    if peak_lr is not None:
        lr_schedule = _dc.replace(
            lr_schedule,
            peak_lr=float(peak_lr),
            decay_lr=float(peak_lr),  # ~constant low LR over the (short) run
            warmup_steps=int(warmup_steps if warmup_steps is not None else 50),
        )
    tx = _optimizer.create_optimizer(config.optimizer, lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # Errors if partial_params is not a subset of the model state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        # Frozen params -> bf16 (trainable LoRA/expert stay fp32), as in init_train_state.
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16))
        )
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    rng = jax.random.key(seed)
    state_shape = jax.eval_shape(init, rng)

    # Resume: restore the FULL train state (params + optimizer + step) from a prior
    # checkpoint, in a fresh process. This sidesteps the gradual GPU-memory leak — each
    # ~1k-step segment runs in a clean container. ``resume_from`` is the step dir
    # (e.g. .../dpo-5k-v2/1000).
    if resume_from is not None:
        import openpi.training.checkpoints as _checkpoints

        rp = Path(resume_from)
        base, ckpt_step = str(rp.parent), int(rp.name)
        data_config = config.data.create(config.assets_dirs, config.model)

        class _ShimDL:
            def data_config(self):
                return data_config

        mngr, _ = _checkpoints.initialize_checkpoint_dir(
            base, keep_period=None, overwrite=False, resume=True
        )
        try:
            restored = _checkpoints.restore_state(mngr, state_shape, _ShimDL(), step=ckpt_step)
        finally:
            mngr.close()
        print(f"[resume] restored train state from {resume_from} (step {ckpt_step})", flush=True)
        return restored, tx

    # Load and validate Mark's params against the model's expected pytree.
    loaded = config.weight_loader.load(state_shape.params.to_pure_dict())
    at.check_pytree_equality(
        expected=state_shape.params.to_pure_dict(), got=loaded, check_shapes=True, check_dtypes=True
    )
    partial_params = traverse_util.unflatten_dict(
        {
            k: v
            for k, v in traverse_util.flatten_dict(loaded).items()
            if not isinstance(v, jax.ShapeDtypeStruct)
        }
    )

    train_state = jax.jit(init, donate_argnums=(1,))(rng, partial_params)
    return train_state, tx


def make_ref_flow(config: Any) -> Callable:
    """Jitted per-sample flow loss under a frozen reference policy (for the GRPO KL term).

    ``flow = ref_flow(ref_state, rng, observation, actions)`` — pass the SAME rng the train
    step uses so flow_ref shares the internal noise/time with the policy's flow (paired).
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    def _ref(ref_state, rng, observation, actions):
        model = nnx.merge(ref_state.model_def, ref_state.params)
        model.eval()
        chunked = model.compute_loss(rng, observation, actions, train=False)
        return jnp.mean(chunked, axis=-1)  # (G,)

    return jax.jit(_ref)


def make_grpo_train_step(
    config: Any,
    *,
    kl_coef: float = 0.1,
    clip_eps: float = 0.2,
    flow_log_prob_scale: float = 1.0,
) -> Callable:
    """Build a jitted **faithful** GRPO step (DeepSeek-style), flow-MSE log-prob surrogate.

        new_state, metrics = train_step(state, rng, observation, candidate_actions, advantages, flow_ref)

    log π(a|o) ≈ −flow_loss / scale. Loss = PPO-clipped policy gradient + KL-to-reference::

        ratio_i = exp(log π_θ,i − log π_old,i)              # π_old = policy at sample time
        L_pg    = −mean_i min(ratio_i·Â_i, clip(ratio_i, 1±ε)·Â_i)
        Δ_i     = log π_ref,i − log π_θ,i
        L_kl    = mean_i (exp(Δ_i) − Δ_i − 1)               # DeepSeek unbiased KL est., ≥ 0
        L       = L_pg + kl_coef · L_kl

    The KL term anchors the policy to the BC reference and **bounds the (otherwise unbounded)
    flow surrogate**, which is what a bare ``mean(A·flow)`` lacks — that omission let the
    policy's flow loss explode and diverge. Only ``trainable_filter`` params get gradients.
    """
    import dataclasses as _dc

    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    import optax

    scale = max(float(flow_log_prob_scale), 1e-8)
    kl_coef = float(kl_coef)
    clip_eps = float(clip_eps)

    def _step(config, state, rng, observation, candidate_actions, advantages, flow_ref):
        model = nnx.merge(state.model_def, state.params)
        model.train()

        def loss_fn(model, rng, observation, actions, advantages, flow_ref):
            chunked = model.compute_loss(rng, observation, actions, train=False)
            flow = jnp.mean(chunked, axis=-1)  # (G,), differentiable
            adv = jax.lax.stop_gradient(advantages)
            logp = -flow / scale
            logp_old = jax.lax.stop_gradient(logp)               # single update: old == current
            logp_ref = -jax.lax.stop_gradient(flow_ref) / scale  # frozen BC reference
            # PPO-clipped surrogate (faithful; clip inactive at ratio==1, correct for multi-epoch).
            ratio = jnp.exp(logp - logp_old)
            pg = -jnp.mean(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv))
            # KL(π_θ || π_ref), DeepSeek unbiased estimator (>= 0), anchors to BC.
            # Clip before exp: unbounded flow surrogate can make delta huge and blow up KL grads.
            delta = jnp.clip(logp_ref - logp, -10.0, 10.0)
            kl = jnp.mean(jnp.exp(delta) - delta - 1.0)
            loss = pg + kl_coef * kl
            return loss, (flow, pg, kl)

        diff_state = nnx.DiffState(0, config.trainable_filter)
        (loss, (flow, pg, kl)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
            model, rng, observation, candidate_actions, advantages, flow_ref
        )

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params = nnx.state(model)

        new_state = _dc.replace(
            state, step=state.step + 1, params=new_params, opt_state=new_opt_state
        )
        if state.ema_decay is not None:
            new_state = _dc.replace(
                new_state,
                ema_params=jax.tree.map(
                    lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                    state.ema_params, new_params,
                ),
            )

        metrics = {
            "loss": loss,
            "pg_loss": pg,
            "kl": kl,
            "flow_mean": jnp.mean(flow),
            "flow_max": jnp.max(flow),
            "flow_ref_mean": jnp.mean(flow_ref),
            "adv_abs_mean": jnp.mean(jnp.abs(advantages)),
            "grad_norm": optax.global_norm(grads),
        }
        return new_state, metrics

    return jax.jit(functools.partial(_step, config))


def make_sampler(config: Any) -> Callable:
    """Build a jitted group sampler: ``actions = sample(state, rng, observation, noise)``.

    ``observation`` must already be batched to the group size G; ``noise`` is
    (G, action_horizon, action_dim). Returns normalized action chunks in model space
    (un-normalize with the data config's norm stats before reward scoring).
    """
    import flax.nnx as nnx
    import jax

    def _sample(state, rng, observation, noise):
        model = nnx.merge(state.model_def, state.params)
        model.eval()
        return model.sample_actions(rng, observation, noise=noise)

    return jax.jit(_sample)


def save_checkpoint(
    state: Any,
    config: Any,
    out_dir: str | Path,
    step: int,
    *,
    overwrite: bool = True,
) -> str:
    """Write an orbax checkpoint loadable by ``policy_config.create_trained_policy``.

    Produces ``<out_dir>/<step>/{params,train_state,assets}`` via openpi's own
    ``save_state``; the ``assets`` item carries the norm stats so the restored
    policy can normalize. ``compare_fifty`` loads ``<out_dir>/<step>`` directly.
    """
    import openpi.training.checkpoints as _checkpoints

    out_dir = Path(out_dir)
    data_config = config.data.create(config.assets_dirs, config.model)

    class _ShimDataLoader:
        def data_config(self):
            return data_config

    mngr, _ = _checkpoints.initialize_checkpoint_dir(
        out_dir, keep_period=None, overwrite=overwrite, resume=False
    )
    try:
        _checkpoints.save_state(mngr, state, _ShimDataLoader(), step)
        mngr.wait_until_finished()
    finally:
        # CRITICAL: close the manager — otherwise each periodic save leaks its async
        # buffers + a device param copy, OOMing the GPU after a few saves.
        mngr.close()
    return str(out_dir / str(step))
