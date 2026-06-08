"""Actor/critic shapes, fixed σ, min-of-N, ref-dropout, export round-trip."""

from __future__ import annotations

import torch

from pi05.rlt.export import export_actor
from pi05.rlt.losses import _ref_dropout

from .helpers import make_actor, make_q, small_cfg


def _inputs(cfg, B=4):
    z = torch.randn(B, cfg.token_dim)
    s = torch.randn(B, cfg.dim_proprio)
    a_ref = torch.randn(B, cfg.chunk_len, cfg.action_dim)
    return z, s, a_ref


def test_actor_output_shape_and_bounds():
    cfg = small_cfg()
    actor = make_actor(cfg)
    z, s, a_ref = _inputs(cfg)
    mu = actor(z, s, a_ref)
    assert mu.shape == (4, cfg.chunk_len, cfg.action_dim)
    # tanh-squashed into the native bounds, per dim.
    assert torch.all(mu[..., 0] >= cfg.accel_min - 1e-4)
    assert torch.all(mu[..., 0] <= cfg.accel_max + 1e-4)
    assert torch.all(mu[..., 1] >= cfg.curvature_min - 1e-4)
    assert torch.all(mu[..., 1] <= cfg.curvature_max + 1e-4)


def test_fixed_sigma_buffer():
    cfg = small_cfg()
    actor = make_actor(cfg)
    expected = torch.tensor([
        cfg.actor_std * (cfg.accel_max - cfg.accel_min),
        cfg.actor_std * (cfg.curvature_max - cfg.curvature_min),
    ])
    assert torch.allclose(actor.a_std, expected)


def test_qensemble_shapes_and_min():
    cfg = small_cfg()
    q = make_q(cfg)
    z, s, a_ref = _inputs(cfg)
    a = torch.randn(4, cfg.chunk_len, cfg.action_dim)
    allq = q(z, s, a)
    assert allq.shape == (cfg.n_critics, 4)
    assert torch.allclose(q.min_q(z, s, a), allq.min(dim=0).values)


def test_ref_dropout_asymmetry():
    a_ref = torch.randn(8, 3, 2)
    assert torch.all(_ref_dropout(a_ref, 1.0, training=True) == 0.0)  # fully dropped
    assert torch.allclose(_ref_dropout(a_ref, 1.0, training=False), a_ref)  # eval keeps
    assert torch.allclose(_ref_dropout(a_ref, 0.0, training=True), a_ref)  # p=0 keeps


def test_export_torchscript_round_trip(tmp_path):
    cfg = small_cfg()
    actor = make_actor(cfg).eval()
    z, s, a_ref = _inputs(cfg)
    with torch.no_grad():
        ref = actor(z, s, a_ref)
    path = export_actor(actor, path=str(tmp_path / "actor.pt"), fmt="torchscript")
    loaded = torch.jit.load(path)
    with torch.no_grad():
        got = loaded(z, s, a_ref)
    assert torch.allclose(got, ref, atol=1e-5)
