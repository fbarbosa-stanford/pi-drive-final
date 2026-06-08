"""Kinematic unicycle test env + synthetic VLA/egomotion seam stand-ins.

A goal-reach task that exercises every edge case the learner core must handle
end-to-end *without* π0.5 or hardware:

  - **SUCCESS** (terminal, r=1): inside the goal radius at low speed.
  - **FAILURE** (terminal, r=0): out of bounds.
  - **TIMEOUT** (r=0, **must bootstrap**): reached ``max_steps`` -- this is the
    case that catches the classic "timeout treated as terminal" bug.

The per-step kinematics are the *same* integration as
``pi05/inference/infer.py:actions_to_trajectory`` (v clamped ≥ 0; heading +=
v·κ·dt; x,y += v·{cos,sin}(heading)·dt) so the env and the real trajectory
decoder never diverge.

The synthetic seams let the core be tested against the real interfaces:
  - ``SyntheticVLA`` (``VLAWrapper``): a scripted imperfect pure-pursuit
    controller emits the reference chunk ã; ``z_rl`` is a fixed seeded linear
    projection of task features + noise -- high-dim and opaque (so the actor
    can't trivially invert it) yet carries enough signal for Q to be learnable.
  - ``SyntheticEgomotion`` (``EgomotionStream``): noisy (v, θ̇) padded to s_p.

``force_outcome`` + the seed make tests able to craft exact boundary episodes.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from ..config import RLTConfig
from ..interfaces import EgomotionStream, VLAWrapper
from ..transition import DoneType


def _wrap(angle: float) -> float:
    """Wrap to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class UnicycleEnv:
    def __init__(
        self,
        cfg: RLTConfig,
        seed: int = 0,
        bounds: float = 30.0,
        goal_radius: float = 1.5,
        v_success: float = 1.0,
        max_steps: int = 80,
        goal: tuple[float, float] = (15.0, 0.0),
    ):
        self.cfg = cfg
        self.dt = cfg.dt
        self.bounds = bounds
        self.goal_radius = goal_radius
        self.v_success = v_success
        self.max_steps = max_steps
        self.goal = goal
        self.rng = np.random.default_rng(seed)
        self._forced: DoneType | None = None
        self.reset()

    def reset(self) -> dict:
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.theta_dot = 0.0
        self.t = 0
        self._forced = None
        return self._obs()

    def force_outcome(self, done_type: DoneType | int) -> None:
        """Coerce the *next* ``step`` to terminate with this done_type. Lets a
        test materialise an exact SUCCESS / FAILURE / TIMEOUT boundary row."""
        self._forced = DoneType(int(done_type))

    def _obs(self) -> dict:
        gx, gy = self.goal
        dist = math.hypot(gx - self.x, gy - self.y)
        bearing = math.atan2(gy - self.y, gx - self.x)
        heading_err = _wrap(bearing - self.theta)
        return {
            "x": self.x, "y": self.y, "theta": self.theta, "v": self.v,
            "theta_dot": self.theta_dot, "t": self.t,
            "dist": dist, "heading_err": heading_err,
            "time_left": (self.max_steps - self.t) / self.max_steps,
        }

    def step(self, action_head) -> tuple[dict, float, DoneType]:
        """Execute one control step (the chunk head, receding-horizon).
        ``action_head`` is (accel, curvature). Returns (obs, reward, done)."""
        accel, kappa = float(action_head[0]), float(action_head[1])
        v_prev = self.v
        self.v = max(self.v + accel * self.dt, 0.0)
        self.theta_dot = self.v * kappa
        self.theta = _wrap(self.theta + self.v * kappa * self.dt)
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt
        self.t += 1

        obs = self._obs()
        done, reward = self._terminate(obs)
        return obs, reward, done

    def _terminate(self, obs: dict) -> tuple[DoneType, float]:
        if self._forced is not None:
            forced, self._forced = self._forced, None
            return forced, (1.0 if forced == DoneType.SUCCESS else 0.0)
        if obs["dist"] < self.goal_radius and self.v < self.v_success:
            return DoneType.SUCCESS, 1.0
        if abs(self.x) > self.bounds or abs(self.y) > self.bounds:
            return DoneType.FAILURE, 0.0
        if self.t >= self.max_steps:
            return DoneType.TIMEOUT, 0.0
        return DoneType.NONE, 0.0


class SyntheticVLA(VLAWrapper):
    """Scripted pure-pursuit reference chunk + seeded high-dim ``z_rl``."""

    _N_FEATURES = 4  # [dist, heading_err, v, time_left]

    def __init__(
        self,
        cfg: RLTConfig,
        seed: int = 0,
        z_noise: float = 0.05,
        ref_noise: float = 0.1,
        steer_bias: float = 0.02,
    ):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.z_noise = z_noise
        self.ref_noise = ref_noise
        self.steer_bias = steer_bias
        gen = torch.Generator().manual_seed(seed)
        # fixed projection features -> z_rl; opaque but information-preserving.
        self._W = torch.randn(cfg.token_dim, self._N_FEATURES, generator=gen)

    @property
    def token_dim(self) -> int:
        return self.cfg.token_dim

    @property
    def chunk_len(self) -> int:
        return self.cfg.chunk_len

    def encode(self, obs: dict) -> tuple[Tensor, Tensor]:
        feats = torch.tensor(
            [obs["dist"] / 30.0, obs["heading_err"] / math.pi,
             obs["v"] / 5.0, obs["time_left"]],
            dtype=torch.float32,
        )
        noise = torch.from_numpy(
            self.rng.normal(0.0, self.z_noise, self.cfg.token_dim).astype("float32")
        )
        z_rl = self._W @ feats + noise
        a_ref = self._pure_pursuit(obs)
        return z_rl, a_ref

    def _pure_pursuit(self, obs: dict) -> Tensor:
        cfg = self.cfg
        dist, herr, v = obs["dist"], obs["heading_err"], obs["v"]
        lookahead = max(dist, 1.0)
        kappa = 2.0 * math.sin(herr) / lookahead + self.steer_bias
        target_v = min(cfg.speed_cap_mps, 0.5 * dist + 0.5)
        accel = 1.5 * (target_v - v)
        accel = float(np.clip(accel, cfg.accel_min, cfg.accel_max))
        kappa = float(np.clip(kappa, cfg.curvature_min, cfg.curvature_max))
        base = torch.tensor([accel, kappa], dtype=torch.float32)
        chunk = base.unsqueeze(0).repeat(cfg.chunk_len, 1)
        chunk += torch.from_numpy(
            self.rng.normal(0.0, self.ref_noise, chunk.shape).astype("float32")
        )
        chunk[:, 0].clamp_(cfg.accel_min, cfg.accel_max)
        chunk[:, 1].clamp_(cfg.curvature_min, cfg.curvature_max)
        return chunk


class SyntheticEgomotion(EgomotionStream):
    """Noisy (v, θ̇) padded to ``dim_proprio`` -- stand-in for the streamed
    egomotion readout. Hold latest obs via ``set_obs`` so ``read()`` matches the
    arg-less interface the real stream will expose."""

    def __init__(self, cfg: RLTConfig, seed: int = 0, noise: float = 0.02):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self._obs: dict | None = None

    @property
    def dim(self) -> int:
        return self.cfg.dim_proprio

    def set_obs(self, obs: dict) -> None:
        self._obs = obs

    def read(self) -> Tensor:
        obs = self._obs or {"v": 0.0, "theta_dot": 0.0}
        s = torch.zeros(self.cfg.dim_proprio, dtype=torch.float32)
        s[0] = obs["v"] + float(self.rng.normal(0.0, self.noise))
        if self.cfg.dim_proprio > 1:
            s[1] = obs["theta_dot"] + float(self.rng.normal(0.0, self.noise))
        return s
