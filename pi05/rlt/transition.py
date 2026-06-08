"""Transition schema for the RLT replay buffer.

Each *stored* transition is one strided C-step sub-chunk ``<x_t, a_{t:t+C}, ...>``.
Reward is sparse / terminal / binary, so a row carries a per-step ``rewards[C]``
(almost always all-zero) plus a ``discounts[C]`` mask that zeroes any step past
the effective length ``n_steps`` (which is < C only on the boundary sub-chunk
that runs into the episode end).

The bootstrap decision (timeout vs. true terminal) and the boundary length ``n``
are computed exactly once -- at insert time, in the buffer -- and stored here as
``bootstrap`` and ``gamma_pow`` (= gamma**n). The TD loss consumes these verbatim
and never re-derives them, so the classic "timeout treated as terminal" bug has
exactly one place it could live (the buffer) and is unit-tested there.

A batch is a plain ``dict[str, Tensor]`` (no tensordict dependency).
"""

from __future__ import annotations

from enum import IntEnum

import torch

Batch = dict[str, torch.Tensor]


class DoneType(IntEnum):
    """How an episode (or the step a sub-chunk reaches) terminated."""

    NONE = 0      # mid-episode: bootstrap
    SUCCESS = 1   # true terminal: do NOT bootstrap
    FAILURE = 2   # true terminal: do NOT bootstrap
    TIMEOUT = 3   # time-limit only: DO bootstrap (V(x') still meaningful)


class SourceType(IntEnum):
    """Which collection phase produced the transition."""

    DEMO = 0
    WARMUP = 1
    ONLINE = 2
    INTERVENTION = 3


def field_specs(cfg) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    """Per-row (dtype, shape-without-batch) for every stored field."""
    C, d = cfg.chunk_len, cfg.action_dim
    z, p = cfg.token_dim, cfg.dim_proprio
    f32, i64, b = torch.float32, torch.int64, torch.bool
    return {
        # state x_t
        "z_rl": (f32, (z,)),
        "s_p": (f32, (p,)),
        "a_ref": (f32, (C, d)),     # the VLA reference chunk a-tilde
        "a": (f32, (C, d)),         # executed action chunk
        # C-step return machinery
        "rewards": (f32, (C,)),     # r_{t'}, t'=1..C (mostly 0)
        "discounts": (f32, (C,)),   # 1 for real steps, 0 for padded/post-terminal
        "n_steps": (i64, ()),       # effective steps in this sub-chunk (<= C)
        # next state x'
        "z_rl_next": (f32, (z,)),
        "s_p_next": (f32, (p,)),
        "a_ref_next": (f32, (C, d)),  # a-tilde' at x', for the target actor
        # bootstrap / termination control (derived + stored)
        "done_type": (i64, ()),
        "bootstrap": (f32, ()),     # 1.0 if gamma_pow * minQ' is added, else 0.0
        "gamma_pow": (f32, ()),     # gamma ** n_steps
        # bookkeeping
        "source": (i64, ()),
        "is_intervention": (b, ()),
        "episode_id": (i64, ()),
    }


def make_storage(capacity: int, cfg, device: str | torch.device = "cpu") -> Batch:
    """Preallocate a (capacity, *shape) tensor per field for the ring buffer."""
    specs = field_specs(cfg)
    return {
        name: torch.zeros((capacity, *shape), dtype=dtype, device=device)
        for name, (dtype, shape) in specs.items()
    }


def validate_batch(batch: Batch, cfg, *, batch_dim: bool = True) -> None:
    """Raise if any field is missing or has the wrong dtype/trailing shape."""
    specs = field_specs(cfg)
    missing = set(specs) - set(batch)
    if missing:
        raise ValueError(f"transition batch missing fields: {sorted(missing)}")
    for name, (dtype, shape) in specs.items():
        t = batch[name]
        if t.dtype != dtype:
            raise TypeError(f"{name}: expected dtype {dtype}, got {t.dtype}")
        trailing = tuple(t.shape[1:]) if batch_dim else tuple(t.shape)
        if trailing != shape:
            raise ValueError(
                f"{name}: expected trailing shape {shape}, got {trailing}"
            )


def move_batch(batch: Batch, device: str | torch.device) -> Batch:
    return {k: v.to(device) for k, v in batch.items()}
