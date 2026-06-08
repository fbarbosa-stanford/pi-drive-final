"""Replay buffer for the RLT learner core.

This is the single highest-correctness-risk file: it is where the chunked
C-step return machinery is assembled and where the "timeout vs. true terminal"
bootstrap decision is made *once* and frozen into each stored row (see
``transition.py``). The TD loss consumes ``bootstrap`` / ``gamma_pow`` verbatim
and never re-derives them, so this is the only place that bug can live.

Insertion is two-phase because the (sparse, terminal) reward is unknown until
the episode ends:

  1. ``add_step(...)`` stages one raw per-timestep record per ``episode_id``.
     Each record carries the *current* state ``x_t = (z_rl, s_p, a_ref, a)``,
     the *one-step-ahead* next state ``x_{t+1}`` (so a sub-chunk that runs to
     the episode boundary still has a real bootstrap state -- crucial for
     TIMEOUT), the immediate reward (almost always 0), and bookkeeping.
  2. ``finalize_episode(episode_id, done_type, terminal_reward)`` backfills the
     terminal reward, rewrites intervention rows so ``a_ref == a``, and emits
     stride-subsampled C-step sub-chunks into the ring storage.

Storage is a flat ring of preallocated tensors (``transition.make_storage``);
sampling returns a plain ``dict[str, Tensor]`` batch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import torch
from torch import Tensor

from .config import RLTConfig
from .transition import (
    Batch,
    DoneType,
    SourceType,
    field_specs,
    make_storage,
)


class ReplayBuffer:
    def __init__(self, cfg: RLTConfig, device: str | torch.device | None = None):
        self.cfg = cfg
        self.device = torch.device(device or cfg.device)
        self.capacity = cfg.capacity
        self.C = cfg.chunk_len
        self.d = cfg.action_dim
        self.stride = cfg.stride
        self.gamma = cfg.gamma

        self._storage = make_storage(self.capacity, cfg, self.device)
        self._specs = field_specs(cfg)
        self._pos = 0
        self._size = 0

        # episode_id -> list[staged step dict]; + last-touch wall-clock for the
        # staging watchdog.
        self._staging: dict[int, list[dict]] = {}
        self._staged_at: dict[int, float] = {}

    # ------------------------------------------------------------------ #
    # phase 1: staging
    # ------------------------------------------------------------------ #
    def add_step(
        self,
        episode_id: int,
        z_rl: Tensor,
        s_p: Tensor,
        a_ref: Tensor,
        a: Tensor,
        *,
        z_rl_next: Tensor,
        s_p_next: Tensor,
        a_ref_next: Tensor,
        reward: float = 0.0,
        source: SourceType | int = SourceType.ONLINE,
        is_intervention: bool = False,
        now: float | None = None,
    ) -> None:
        """Stage one decision step. ``*_next`` is the state one control step
        ahead (what the env returned after executing this step's chunk head)."""
        step = {
            "z_rl": self._as(z_rl, "z_rl"),
            "s_p": self._as(s_p, "s_p"),
            "a_ref": self._as(a_ref, "a_ref"),
            "a": self._as(a, "a"),
            "z_rl_next": self._as(z_rl_next, "z_rl_next"),
            "s_p_next": self._as(s_p_next, "s_p_next"),
            "a_ref_next": self._as(a_ref_next, "a_ref_next"),
            "reward": float(reward),
            "source": int(source),
            "is_intervention": bool(is_intervention),
        }
        self._staging.setdefault(episode_id, []).append(step)
        self._staged_at[episode_id] = time.monotonic() if now is None else now

    def _as(self, t: Tensor, name: str) -> Tensor:
        dtype, shape = self._specs[name]
        out = torch.as_tensor(t, dtype=dtype, device=self.device)
        if tuple(out.shape) != shape:
            raise ValueError(f"{name}: expected shape {shape}, got {tuple(out.shape)}")
        return out

    # ------------------------------------------------------------------ #
    # phase 2: finalize -> emit sub-chunks
    # ------------------------------------------------------------------ #
    def finalize_episode(
        self,
        episode_id: int,
        done_type: DoneType | int,
        terminal_reward: float = 1.0,
    ) -> int:
        """Backfill reward, fix intervention refs, emit stride-subsampled
        C-step sub-chunks. Returns the number of rows written."""
        steps = self._staging.pop(episode_id, None)
        self._staged_at.pop(episode_id, None)
        if not steps:
            return 0

        done_type = DoneType(int(done_type))
        T = len(steps)

        # (1) terminal-reward backfill: only SUCCESS lands a nonzero reward, on
        # the last step. FAILURE/TIMEOUT/NONE leave the (sparse) rewards as-is.
        if done_type == DoneType.SUCCESS:
            steps[-1]["reward"] += float(terminal_reward)

        # (2) intervention rows: the human action is *both* executed and the
        # reference the BC anchor pulls toward, so a_ref := a.
        for st in steps:
            if st["is_intervention"]:
                st["a_ref"] = st["a"].clone()

        # (3) emit stride-subsampled sub-chunks.
        n_written = 0
        for t0 in range(0, T, self.stride):
            self._emit_subchunk(steps, t0, T, done_type)
            n_written += 1
        return n_written

    def _emit_subchunk(
        self, steps: list[dict], t0: int, T: int, done_type: DoneType
    ) -> None:
        C = self.C
        n = min(C, T - t0)                 # effective steps (<= C); >= 1 always
        reaches_end = (t0 + n == T)        # window runs into the episode end
        is_true_terminal = done_type in (DoneType.SUCCESS, DoneType.FAILURE)

        # bootstrap iff the window does NOT land on a true terminal. Interior
        # windows and TIMEOUT both bootstrap (time-limit != end of MDP).
        bootstrap = 0.0 if (reaches_end and is_true_terminal) else 1.0
        # done_type is the episode's only at the boundary window; interior
        # windows are mid-episode (NONE).
        row_done = done_type if reaches_end else DoneType.NONE

        rewards = torch.zeros(C, dtype=torch.float32, device=self.device)
        discounts = torch.zeros(C, dtype=torch.float32, device=self.device)
        for k in range(n):
            rewards[k] = steps[t0 + k]["reward"]
            discounts[k] = 1.0

        last = steps[t0 + n - 1]           # state after the last executed step
        head = steps[t0]

        row = {
            "z_rl": head["z_rl"],
            "s_p": head["s_p"],
            "a_ref": head["a_ref"],
            "a": head["a"],
            "rewards": rewards,
            "discounts": discounts,
            "n_steps": torch.tensor(n, dtype=torch.int64, device=self.device),
            "z_rl_next": last["z_rl_next"],
            "s_p_next": last["s_p_next"],
            "a_ref_next": last["a_ref_next"],
            "done_type": torch.tensor(int(row_done), dtype=torch.int64, device=self.device),
            "bootstrap": torch.tensor(bootstrap, dtype=torch.float32, device=self.device),
            "gamma_pow": torch.tensor(self.gamma ** n, dtype=torch.float32, device=self.device),
            "source": torch.tensor(head["source"], dtype=torch.int64, device=self.device),
            "is_intervention": torch.tensor(head["is_intervention"], dtype=torch.bool, device=self.device),
            "episode_id": torch.tensor(0, dtype=torch.int64, device=self.device),
        }
        self._write(row)

    def _write(self, row: dict[str, Tensor]) -> None:
        i = self._pos
        for name in self._specs:
            self._storage[name][i] = row[name]
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------ #
    # staging watchdog
    # ------------------------------------------------------------------ #
    def flush_stale(self, now: float | None = None) -> int:
        """Finalize episodes that were never closed (stale > staging_ttl_s) as
        TIMEOUT, so a dropped rollout can't leak staged steps forever."""
        now = time.monotonic() if now is None else now
        ttl = self.cfg.staging_ttl_s
        stale = [eid for eid, t in self._staged_at.items() if now - t > ttl]
        total = 0
        for eid in stale:
            total += self.finalize_episode(eid, DoneType.TIMEOUT)
        return total

    # ------------------------------------------------------------------ #
    # sampling
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return self._size

    def ready(self) -> bool:
        return self._size >= self.cfg.learning_starts

    def sample(self, batch_size: int | None = None) -> Batch:
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        bs = batch_size or self.cfg.batch_size
        idx = self._sample_indices(bs)
        return {name: self._storage[name][idx] for name in self._specs}

    def _sample_indices(self, bs: int) -> Tensor:
        weights = self.cfg.source_weights
        if not weights:
            return torch.randint(0, self._size, (bs,), device=self.device)

        # per-source upweighting: map each present row's source -> its weight.
        src = self._storage["source"][: self._size]
        w = torch.zeros(self._size, dtype=torch.float32, device=self.device)
        for name, wt in weights.items():
            sid = int(SourceType[name.upper()])
            w[src == sid] = float(wt)
        if float(w.sum()) <= 0.0:  # no present row matched a weighted source
            return torch.randint(0, self._size, (bs,), device=self.device)
        return torch.multinomial(w, bs, replacement=True)
