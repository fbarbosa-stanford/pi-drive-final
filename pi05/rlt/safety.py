"""Native-units safety backstop for the RL action chunk.

Clamps the proposed ``(accel, curvature)`` chunk in π0.5's *native* trained
units -- this is the only safety the learner core knows about. Three layers,
applied in order:

  1. **Absolute bounds** -- accel ∈ [accel_min, accel_max] m/s²,
     curvature ∈ [curvature_min, curvature_max] rad/m (π0.5's trained range).
  2. **Per-step rate caps** -- |Δaccel| ≤ max_accel_rate (jerk) and
     |Δcurvature| ≤ max_curvature_rate per 10 Hz step, chained from the
     previously-executed action when available (smoothness / comfort backstop).
  3. **Training speed cap** -- projects speed forward at dt and limits accel so
     the cart neither exceeds speed_cap_mps nor reverses during bring-up.

NOTE: ``limits.py`` (effective_gas_cap / STEERING_*) is the single source of
truth for the *actuation* clamp, but it operates in normalized pot / steering-
degree units and therefore applies at the deferred tracker seam (accel→gas-pot,
curvature→steering-angle), NOT here. The two clamps compose without double-
converting: native units here, command units there.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import RLTConfig
from .interfaces import SafetyLayer


class NativeUnitsSafetyLayer(SafetyLayer):
    def __init__(self, cfg: RLTConfig):
        self.cfg = cfg

    def clamp(self, a: Tensor, state: dict | None = None) -> tuple[Tensor, bool]:
        """``a`` is a (C, d) chunk of (accel, curvature). ``state`` may carry
        ``prev_action`` (d,) for rate chaining and ``speed`` (float) for the
        speed cap. Returns (safe_chunk, was_modified)."""
        cfg = self.cfg
        state = state or {}
        orig = a
        a = a.clone()

        # (1) absolute bounds, per dim.
        a[:, 0].clamp_(cfg.accel_min, cfg.accel_max)
        a[:, 1].clamp_(cfg.curvature_min, cfg.curvature_max)

        # (2) per-step rate caps, chained from the previous executed action.
        rate = torch.tensor(
            [cfg.max_accel_rate, cfg.max_curvature_rate],
            dtype=a.dtype, device=a.device,
        )
        prev = state.get("prev_action")
        prev = a[0].clone() if prev is None else torch.as_tensor(
            prev, dtype=a.dtype, device=a.device
        )
        for k in range(a.shape[0]):
            lo, hi = prev - rate, prev + rate
            a[k] = torch.maximum(torch.minimum(a[k], hi), lo)
            prev = a[k].clone()

        # (3) training speed cap: keep projected speed in [0, speed_cap_mps].
        speed = state.get("speed")
        if speed is not None:
            dt = cfg.dt
            v = float(speed)
            for k in range(a.shape[0]):
                accel = float(a[k, 0])
                v_next = v + accel * dt
                if v_next > cfg.speed_cap_mps:
                    accel = (cfg.speed_cap_mps - v) / dt
                elif v_next < 0.0:
                    accel = -v / dt
                a[k, 0] = accel
                v = max(0.0, min(cfg.speed_cap_mps, v + accel * dt))

        was_modified = not torch.allclose(a, orig)
        return a, was_modified
