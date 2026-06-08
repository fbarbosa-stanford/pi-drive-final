"""Chunked C-step TD target arithmetic, isolated from real nets via stubs so
every boundary case is checked against a hand-computed value."""

from __future__ import annotations

import math

import torch

from pi05.rlt.losses import _discounted_reward_sum, critic_td_target

from .helpers import small_cfg


class StubActor:
    def __init__(self, C, d):
        self.C, self.d = C, d

    def rsample(self, z, s, a_ref):
        return torch.zeros(z.shape[0], self.C, self.d)


class StubQ:
    def __init__(self, val):
        self.val = val

    def min_q(self, z, s, a):
        return torch.full((z.shape[0],), float(self.val))


def _batch(cfg, *, rewards, discounts, bootstrap, n):
    return {
        "rewards": torch.tensor([rewards], dtype=torch.float32),
        "discounts": torch.tensor([discounts], dtype=torch.float32),
        "bootstrap": torch.tensor([bootstrap], dtype=torch.float32),
        "gamma_pow": torch.tensor([cfg.gamma ** n], dtype=torch.float32),
        "z_rl_next": torch.zeros(1, cfg.token_dim),
        "s_p_next": torch.zeros(1, cfg.dim_proprio),
        "a_ref_next": torch.zeros(1, cfg.chunk_len, cfg.action_dim),
    }


def test_discounted_reward_sum_exponents():
    g = 0.9
    r = torch.tensor([[1.0, 1.0, 1.0]])
    d = torch.ones(1, 3)
    out = float(_discounted_reward_sum(r, d, g))
    assert math.isclose(out, 1 + g + g ** 2, rel_tol=1e-6)


def test_interior_chunk_bootstraps_with_gamma_pow_C():
    cfg = small_cfg(gamma=0.9)  # C=3
    actor, q = StubActor(cfg.chunk_len, cfg.action_dim), StubQ(2.0)
    b = _batch(cfg, rewards=[0, 0, 0], discounts=[1, 1, 1], bootstrap=1.0, n=3)
    y = float(critic_td_target(b, actor, q, cfg))
    assert math.isclose(y, cfg.gamma ** 3 * 2.0, rel_tol=1e-6)


def test_true_terminal_does_not_bootstrap():
    cfg = small_cfg(gamma=0.9)
    actor, q = StubActor(cfg.chunk_len, cfg.action_dim), StubQ(5.0)
    # terminal reward at last step, no bootstrap.
    b = _batch(cfg, rewards=[0, 0, 1], discounts=[1, 1, 1], bootstrap=0.0, n=3)
    y = float(critic_td_target(b, actor, q, cfg))
    assert math.isclose(y, cfg.gamma ** 2 * 1.0, rel_tol=1e-6)


def test_timeout_vs_terminal_differ_by_exactly_gamma_pow_minQ():
    cfg = small_cfg(gamma=0.9)
    actor, q = StubActor(cfg.chunk_len, cfg.action_dim), StubQ(3.0)
    common = dict(rewards=[0, 0, 1], discounts=[1, 1, 1], n=2)
    y_term = float(critic_td_target(_batch(cfg, bootstrap=0.0, **common), actor, q, cfg))
    y_to = float(critic_td_target(_batch(cfg, bootstrap=1.0, **common), actor, q, cfg))
    assert math.isclose(y_to - y_term, cfg.gamma ** 2 * 3.0, rel_tol=1e-6)


def test_post_terminal_steps_are_masked():
    cfg = small_cfg(gamma=0.9)
    actor, q = StubActor(cfg.chunk_len, cfg.action_dim), StubQ(0.0)
    # a bogus reward sits past the effective length but discounts mask it out.
    b = _batch(cfg, rewards=[0, 1, 99], discounts=[1, 1, 0], bootstrap=0.0, n=2)
    y = float(critic_td_target(b, actor, q, cfg))
    assert math.isclose(y, cfg.gamma ** 1 * 1.0, rel_tol=1e-6)
