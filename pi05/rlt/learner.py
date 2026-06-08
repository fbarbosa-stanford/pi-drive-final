"""Async off-policy actor-critic learner for RLT.

Owns the online + Polyak-target actor/critics, their optimisers, and drives the
update schedule fixed by the spec:

  - **UTD** critic updates per learner step (update-to-data ratio).
  - **critic_per_actor : 1** critic:actor cadence (2:1 by default) via a running
    counter, so the actor moves on a slower clock than the value function.
  - **Polyak** target update after *every* online update (critics after each
    critic step, actor target after each actor step).
  - ``learning_starts`` gate before any update runs.
  - periodic **actor-only** export (``export_every``) through the atomic-swap
    seam so the runtime hot-reloads weights without cross-thread tensor races
    (the rollout reads the exported artifact, never the live module).

``step()`` is one synchronous learner iteration (used by the validation CLI and
tests). ``start()/stop()`` run the same loop on a background thread; a lock
serialises buffer access so a rollout thread can ``add_step`` concurrently.
"""

from __future__ import annotations

import copy
import threading

import torch

from .buffer import ReplayBuffer
from .config import RLTConfig
from .export import export_actor
from .losses import critic_loss, policy_loss
from .nets import GaussianActor, QEnsemble, polyak_update


class AsyncLearner:
    def __init__(
        self,
        cfg: RLTConfig,
        buffer: ReplayBuffer,
        device: str | torch.device | None = None,
    ):
        self.cfg = cfg
        self.buffer = buffer
        self.device = torch.device(device or cfg.device)

        self.actor = GaussianActor(
            state_dim=cfg.state_dim,
            chunk_len=cfg.chunk_len,
            action_dim=cfg.action_dim,
            hidden=cfg.actor_hidden,
            action_low=(cfg.accel_min, cfg.curvature_min),
            action_high=(cfg.accel_max, cfg.curvature_max),
            std=(cfg.actor_std * (cfg.accel_max - cfg.accel_min),
                 cfg.actor_std * (cfg.curvature_max - cfg.curvature_min)),
        ).to(self.device)
        self.q = QEnsemble(
            state_dim=cfg.state_dim,
            action_flat_dim=cfg.action_flat_dim,
            hidden=cfg.critic_hidden,
            n_critics=cfg.n_critics,
        ).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).eval()
        self.target_q = copy.deepcopy(self.q).eval()
        for p in self.target_actor.parameters():
            p.requires_grad_(False)
        for p in self.target_q.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.critic_lr)

        self.step_count = 0
        self._critic_since_actor = 0
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # single updates
    # ------------------------------------------------------------------ #
    def _critic_update(self) -> float:
        batch = self.buffer.sample()
        loss = critic_loss(batch, self.q, self.target_actor, self.target_q, self.cfg)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.critic_opt.step()
        polyak_update(self.q, self.target_q, self.cfg.tau)
        return float(loss.detach())

    def _actor_update(self) -> float:
        batch = self.buffer.sample()
        loss = policy_loss(batch, self.actor, self.q, self.cfg)
        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip)
        self.actor_opt.step()
        polyak_update(self.actor, self.target_actor, self.cfg.tau)
        return float(loss.detach())

    # ------------------------------------------------------------------ #
    # one learner iteration
    # ------------------------------------------------------------------ #
    def step(self) -> dict[str, float]:
        """UTD critic updates + proportional actor updates. No-op (returns {})
        until the buffer passes the learning_starts gate."""
        if not self.buffer.ready():
            return {}

        critic_losses: list[float] = []
        actor_losses: list[float] = []
        with self.lock:
            for _ in range(self.cfg.utd):
                critic_losses.append(self._critic_update())
                self._critic_since_actor += 1
                if self._critic_since_actor >= self.cfg.critic_per_actor:
                    actor_losses.append(self._actor_update())
                    self._critic_since_actor = 0

        self.step_count += 1
        self._maybe_export()

        out = {"critic_loss": sum(critic_losses) / len(critic_losses)}
        if actor_losses:
            out["actor_loss"] = sum(actor_losses) / len(actor_losses)
        return out

    def _maybe_export(self) -> None:
        if self.cfg.export_every > 0 and self.step_count % self.cfg.export_every == 0:
            self.export()

    def export(self) -> str:
        with self.lock:
            return export_actor(self.actor, cfg=self.cfg)

    # ------------------------------------------------------------------ #
    # background thread
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self.buffer.ready():
                self._stop.wait(0.05)
                continue
            self.step()
