"""Shared test builders for the RLT learner-core tests."""

from __future__ import annotations

import torch

from pi05.rlt.config import RLTConfig
from pi05.rlt.nets import GaussianActor, QEnsemble


def small_cfg(**overrides) -> RLTConfig:
    base = dict(
        token_dim=4,
        dim_proprio=3,
        chunk_len=3,
        action_dim=2,
        stride=2,
        n_critics=2,
        actor_hidden=(16, 16),
        critic_hidden=(16, 16),
        capacity=200,
        batch_size=8,
        learning_starts=4,
        device="cpu",
    )
    base.update(overrides)
    return RLTConfig(**base)


def make_actor(cfg: RLTConfig) -> GaussianActor:
    return GaussianActor(
        state_dim=cfg.state_dim,
        chunk_len=cfg.chunk_len,
        action_dim=cfg.action_dim,
        hidden=cfg.actor_hidden,
        action_low=(cfg.accel_min, cfg.curvature_min),
        action_high=(cfg.accel_max, cfg.curvature_max),
        std=(cfg.actor_std * (cfg.accel_max - cfg.accel_min),
             cfg.actor_std * (cfg.curvature_max - cfg.curvature_min)),
    )


def make_q(cfg: RLTConfig) -> QEnsemble:
    return QEnsemble(
        state_dim=cfg.state_dim,
        action_flat_dim=cfg.action_flat_dim,
        hidden=cfg.critic_hidden,
        n_critics=cfg.n_critics,
    )


def add_episode(buf, cfg, eid, T, *, source=None, intervention=False, now=0.0):
    """Stage a T-step episode where step ``t`` carries the scalar ``t`` in every
    state field (so tests can read back the head index from ``z_rl[...,0]`` and
    the bootstrap next-state from ``z_rl_next[...,0]``). Executed action ``a`` is
    ``t+0.5`` so it is distinguishable from the reference ``a_ref = t``."""
    from pi05.rlt.transition import SourceType
    source = SourceType.ONLINE if source is None else source
    C, d = cfg.chunk_len, cfg.action_dim
    z, p = cfg.token_dim, cfg.dim_proprio
    for t in range(T):
        buf.add_step(
            eid,
            z_rl=torch.full((z,), float(t)),
            s_p=torch.full((p,), float(t)),
            a_ref=torch.full((C, d), float(t)),
            a=torch.full((C, d), float(t) + 0.5),
            z_rl_next=torch.full((z,), float(t + 1)),
            s_p_next=torch.full((p,), float(t + 1)),
            a_ref_next=torch.full((C, d), float(t + 1)),
            reward=0.0,
            source=source,
            is_intervention=intervention,
            now=now,
        )
