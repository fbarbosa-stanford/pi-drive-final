"""Native-units safety backstop: bounds, rate caps, speed cap, was_modified."""

from __future__ import annotations

import torch

from pi05.rlt.safety import NativeUnitsSafetyLayer

from .helpers import small_cfg


def test_absolute_bounds_clamped():
    cfg = small_cfg()
    safety = NativeUnitsSafetyLayer(cfg)
    a = torch.tensor([[100.0, 100.0], [-100.0, -100.0], [0.0, 0.0]])
    safe, modified = safety.clamp(a)
    assert modified
    assert torch.all(safe[:, 0] >= cfg.accel_min) and torch.all(safe[:, 0] <= cfg.accel_max)
    assert torch.all(safe[:, 1] >= cfg.curvature_min) and torch.all(safe[:, 1] <= cfg.curvature_max)


def test_rate_caps_chained_from_prev_action():
    cfg = small_cfg()
    safety = NativeUnitsSafetyLayer(cfg)
    # within bounds, but a big jump in one step.
    a = torch.tensor([[9.0, 0.0], [9.0, 0.0], [9.0, 0.0]])
    prev = torch.tensor([0.0, 0.0])
    safe, modified = safety.clamp(a, {"prev_action": prev})
    assert modified
    # first step can't exceed prev + max_accel_rate.
    assert float(safe[0, 0]) <= cfg.max_accel_rate + 1e-5
    # consecutive accel deltas respect the jerk cap.
    deltas = (safe[1:, 0] - safe[:-1, 0]).abs()
    assert torch.all(deltas <= cfg.max_accel_rate + 1e-5)


def test_speed_cap_limits_projected_speed():
    cfg = small_cfg()
    safety = NativeUnitsSafetyLayer(cfg)
    a = torch.zeros(5, 2)
    a[:, 0] = 9.0  # full accel every step
    safe, _ = safety.clamp(a, {"speed": 0.0})
    # integrate projected speed; must never exceed the training cap.
    v = 0.0
    for k in range(5):
        v = max(0.0, v + float(safe[k, 0]) * cfg.dt)
        assert v <= cfg.speed_cap_mps + 1e-5


def test_unmodified_action_reports_false():
    cfg = small_cfg()
    safety = NativeUnitsSafetyLayer(cfg)
    a = torch.zeros(cfg.chunk_len, cfg.action_dim)
    safe, modified = safety.clamp(a, {"prev_action": torch.zeros(2), "speed": 0.0})
    assert not modified
    assert torch.allclose(safe, a)
