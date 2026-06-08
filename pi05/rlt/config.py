"""RLT learner-core configuration.

Every knob for the RL-Token (RLT) actor-critic lives here. Defaults follow the
RLT paper / the Cart FSD spec; ``__post_init__`` validates invariants so a bad
config fails loudly at construction instead of silently mistraining.

The two knobs that matter most (per the spec):
  - ``gamma`` is the *speed* knob (sparse terminal reward + discount => finishing
    sooner yields higher return). Lower => more speed pressure, harder credit
    assignment.
  - ``beta`` is the BC-anchor / stability knob (how hard the actor is pulled
    toward the VLA reference chunk). Start high, tune down while watching for
    aggression. It also acts as the implicit comfort/smoothness prior, so there
    is deliberately no separate smoothness reward term.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RLTConfig:
    # --- action / chunk geometry ---
    action_dim: int = 2          # (acceleration, curvature) -- pi0.5 native
    chunk_len: int = 5           # C: RL action-chunk length (0.5s @ 10Hz)
    stride: int = 2              # sub-chunk subsampling stride into the buffer
    control_hz: float = 10.0

    # --- state dims ---
    token_dim: int = 2048        # z_rl (RL token) width
    dim_proprio: int = 16        # s_p width (streamed egomotion readout)

    # --- RL core ---
    gamma: float = 0.97          # discount == the speed knob
    beta: float = 1.0            # BC anchor == stability knob (start high)
    actor_std: float = 0.05      # fixed Gaussian std (fraction of action range)
    ref_dropout_p: float = 0.5   # prob of zeroing a_ref to the actor (policy only)
    tau: float = 0.005           # Polyak target-net update rate

    # --- networks ---
    actor_hidden: tuple[int, ...] = (256, 256)   # (512, 512, 512) for hard segments
    critic_hidden: tuple[int, ...] = (256, 256)
    n_critics: int = 2           # ensemble size; min-of-N for targets

    # --- optimisation ---
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    grad_clip: float = 1.0
    batch_size: int = 256
    utd: int = 5                 # update-to-data ratio (per learner step)
    critic_per_actor: int = 2    # critic updates per actor update (2:1)
    learning_starts: int = 1000  # min buffer transitions before updates begin

    # --- buffer ---
    capacity: int = 500_000
    staging_ttl_s: float = 120.0  # flush never-finalized episodes as TIMEOUT
    # opt-in per-source sampling weights; None => uniform over present sources.
    # keys are SourceType names: "demo", "warmup", "online", "intervention".
    source_weights: dict[str, float] | None = None

    # --- rollout / warmup ---
    n_warmup_episodes: int = 30

    # --- safety (native action units) ---
    accel_min: float = -9.8      # m/s^2
    accel_max: float = 9.8
    curvature_min: float = -0.2  # rad/m
    curvature_max: float = 0.2
    max_accel_rate: float = 5.0   # |a_{t+1}-a_t| per step cap (jerk), m/s^2/step
    max_curvature_rate: float = 0.1  # curvature-rate cap, rad/m/step
    speed_cap_mps: float = 4.0    # training speed cap (single-digit m/s)

    # --- export ---
    export_every: int = 500       # learner steps between actor exports
    export_path: str = "actor_latest.pt"
    export_fmt: str = "torchscript"  # "torchscript" | "state_dict"

    # --- misc ---
    device: str = "cpu"          # "cpu" | "mps" | "cuda"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.action_dim <= 0:
            raise ValueError("action_dim must be > 0")
        if self.chunk_len <= 0:
            raise ValueError("chunk_len (C) must be > 0")
        if self.stride <= 0:
            raise ValueError("stride must be > 0")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("gamma must be in (0, 1)")
        if self.beta < 0.0:
            raise ValueError("beta must be >= 0")
        if self.actor_std <= 0.0:
            raise ValueError("actor_std must be > 0")
        if not (0.0 <= self.ref_dropout_p <= 1.0):
            raise ValueError("ref_dropout_p must be in [0, 1]")
        if not (0.0 < self.tau <= 1.0):
            raise ValueError("tau must be in (0, 1]")
        if self.n_critics < 2:
            raise ValueError("n_critics must be >= 2 (min-of-N needs >= 2)")
        if self.utd < 1:
            raise ValueError("utd must be >= 1")
        if self.critic_per_actor < 1:
            raise ValueError("critic_per_actor must be >= 1")
        if self.token_dim <= 0 or self.dim_proprio < 0:
            raise ValueError("token_dim must be > 0 and dim_proprio >= 0")
        if self.accel_min >= self.accel_max:
            raise ValueError("accel_min must be < accel_max")
        if self.curvature_min >= self.curvature_max:
            raise ValueError("curvature_min must be < curvature_max")
        if self.source_weights is not None:
            for k, v in self.source_weights.items():
                if v < 0:
                    raise ValueError(f"source_weights[{k}] must be >= 0")

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def state_dim(self) -> int:
        """Flat input width consumed by critic / actor trunk: z_rl + s_p."""
        return self.token_dim + self.dim_proprio

    @property
    def action_flat_dim(self) -> int:
        """Flattened action-chunk width: C * d."""
        return self.chunk_len * self.action_dim
