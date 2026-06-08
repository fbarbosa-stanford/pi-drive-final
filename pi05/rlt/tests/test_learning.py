"""End-to-end smoke: the whole core wired through the real seam interfaces runs
and produces finite losses / Q-values. The *improvement* gate lives in
``validate.py``; here we only guard against shape/NaN regressions cheaply."""

from __future__ import annotations

import math
import random

import torch

from pi05.rlt.buffer import ReplayBuffer
from pi05.rlt.learner import AsyncLearner
from pi05.rlt.safety import NativeUnitsSafetyLayer
from pi05.rlt.transition import DoneType, SourceType
from pi05.rlt.envs.unicycle import SyntheticEgomotion, SyntheticVLA, UnicycleEnv

from .helpers import small_cfg


def _run_episode(env, vla, ego, buf, safety, eid, actor, explore, rng):
    obs = env.reset()
    prev = None
    while True:
        z_rl, a_ref = vla.encode(obs)
        ego.set_obs(obs)
        s_p = ego.read()
        if actor is None:
            a = a_ref.clone()
            src = SourceType.WARMUP
        else:
            with torch.no_grad():
                a = (actor.rsample if explore else actor.forward)(
                    z_rl[None], s_p[None], a_ref[None]
                ).squeeze(0)
            src = SourceType.ONLINE
        a, _ = safety.clamp(a, {"prev_action": prev, "speed": obs["v"]})
        prev = a[0].clone()
        nobs, reward, done = env.step(a[0])
        z2, ar2 = vla.encode(nobs)
        ego.set_obs(nobs)
        s2 = ego.read()
        buf.add_step(eid, z_rl, s_p, a_ref, a, z_rl_next=z2, s_p_next=s2,
                     a_ref_next=ar2, reward=0.0, source=src)
        obs = nobs
        if done != DoneType.NONE:
            buf.finalize_episode(eid, done, terminal_reward=reward)
            return done


def test_pipeline_runs_with_finite_losses():
    cfg = small_cfg(
        token_dim=16, dim_proprio=4, actor_hidden=(32, 32), critic_hidden=(32, 32),
        batch_size=32, utd=2, learning_starts=50, capacity=5000, export_every=0,
    )
    torch.manual_seed(0)
    rng = random.Random(0)
    env = UnicycleEnv(cfg, seed=0, max_steps=30)
    vla = SyntheticVLA(cfg, seed=0)
    ego = SyntheticEgomotion(cfg, seed=0)
    safety = NativeUnitsSafetyLayer(cfg)
    buf = ReplayBuffer(cfg)
    learner = AsyncLearner(cfg, buf)

    eid = 0
    for _ in range(10):  # warmup with the scripted reference
        _run_episode(env, vla, ego, buf, safety, eid, None, False, rng)
        eid += 1

    assert buf.ready()
    saw_update = False
    for _ in range(20):
        _run_episode(env, vla, ego, buf, safety, eid, learner.actor, True, rng)
        eid += 1
        for _ in range(4):
            m = learner.step()
            if m:
                saw_update = True
                assert all(math.isfinite(v) for v in m.values())

    assert saw_update
    probe = buf.sample(32)
    with torch.no_grad():
        q = learner.q.min_q(probe["z_rl"], probe["s_p"], probe["a"])
    assert torch.all(torch.isfinite(q))
