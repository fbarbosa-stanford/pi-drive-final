"""Replay-buffer correctness: stride subsampling, boundary tails, terminal
backfill, intervention refs, multi-source sampling, staging watchdog."""

from __future__ import annotations

import math

import torch

from pi05.rlt.buffer import ReplayBuffer
from pi05.rlt.transition import DoneType, SourceType

from .helpers import add_episode, small_cfg


def _rows(buf):
    n = len(buf)
    return {k: buf._storage[k][:n] for k in buf._specs}


def test_stride_start_indices():
    cfg = small_cfg()  # C=3, stride=2
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=5)
    n = buf.finalize_episode(0, DoneType.TIMEOUT)
    assert n == math.ceil(5 / cfg.stride) == 3
    heads = [float(buf._storage["z_rl"][i, 0]) for i in range(len(buf))]
    assert heads == [0.0, 2.0, 4.0]


def test_interior_full_chunk_and_boundary_tail():
    cfg = small_cfg()
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=5)
    buf.finalize_episode(0, DoneType.TIMEOUT)
    rows = _rows(buf)
    # t0=0 interior: n == C and never indexes past T-1.
    assert int(rows["n_steps"][0]) == cfg.chunk_len
    # t0=4 boundary tail kept with n==1; next-state is the step-4 next (==5).
    assert int(rows["n_steps"][2]) == 1
    assert float(rows["z_rl_next"][2, 0]) == 5.0


def test_success_backfill_and_terminal_bootstrap():
    cfg = small_cfg()
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=5)
    buf.finalize_episode(0, DoneType.SUCCESS, terminal_reward=1.0)
    rows = _rows(buf)
    heads = [float(rows["z_rl"][i, 0]) for i in range(3)]
    assert heads == [0.0, 2.0, 4.0]

    # Interior chunk (head 0) does not reach the terminal: bootstraps, no reward.
    assert float(rows["bootstrap"][0]) == 1.0
    assert float(rows["rewards"][0].sum()) == 0.0
    assert math.isclose(float(rows["gamma_pow"][0]), cfg.gamma ** 3, rel_tol=1e-6)

    # Chunk head=2 covers steps 2,3,4: terminal reward lands at index 2, no boot.
    assert float(rows["bootstrap"][1]) == 0.0
    assert float(rows["rewards"][1][2]) == 1.0
    assert float(rows["rewards"][1][:2].sum()) == 0.0

    # Chunk head=4 (n=1) holds the terminal reward at index 0, no bootstrap.
    assert float(rows["bootstrap"][2]) == 0.0
    assert float(rows["rewards"][2][0]) == 1.0


def test_timeout_bootstraps_everywhere_no_reward():
    cfg = small_cfg()
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=5)
    buf.finalize_episode(0, DoneType.TIMEOUT)
    rows = _rows(buf)
    # Timeout != terminal: every row bootstraps, and nothing is backfilled.
    assert torch.all(rows["bootstrap"] == 1.0)
    assert float(rows["rewards"].sum()) == 0.0


def test_intervention_sets_aref_equal_to_a():
    cfg = small_cfg()
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=4, intervention=True)
    buf.finalize_episode(0, DoneType.TIMEOUT)
    rows = _rows(buf)
    # a was t+0.5, a_ref was t; intervention rewrites a_ref := a.
    assert torch.allclose(rows["a_ref"], rows["a"])


def test_multi_source_weighting_excludes_zero_weight():
    cfg = small_cfg(source_weights={"demo": 1.0, "online": 0.0})
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 0, T=4, source=SourceType.DEMO)
    add_episode(buf, cfg, 1, T=4, source=SourceType.ONLINE)
    buf.finalize_episode(0, DoneType.TIMEOUT)
    buf.finalize_episode(1, DoneType.TIMEOUT)
    batch = buf.sample(256)
    assert torch.all(batch["source"] == int(SourceType.DEMO))


def test_staging_watchdog_flushes_as_timeout():
    cfg = small_cfg()
    buf = ReplayBuffer(cfg)
    add_episode(buf, cfg, 7, T=3, now=0.0)  # never explicitly finalized
    assert len(buf) == 0
    n = buf.flush_stale(now=cfg.staging_ttl_s + 1.0)
    assert n >= 1 and len(buf) >= 1
    rows = _rows(buf)
    # Flushed as TIMEOUT -> boundary rows bootstrap.
    assert torch.all(rows["bootstrap"] == 1.0)
    # Episode is no longer staged.
    assert buf.flush_stale(now=cfg.staging_ttl_s + 100.0) == 0
