"""End-to-end training-curve validation (the "learning curve climbs" gate).

Wires the whole learner core against the unicycle env through the real seam
interfaces (``SyntheticVLA`` / ``SyntheticEgomotion``):

  seed warmup (scripted reference) -> online loop (actor + exploration +
  scripted human interventions) -> assert the probe-Q trends up and losses stay
  finite. Exits non-zero on a flatline / NaN, so CI can use it as a smoke gate.

Run: ``uv run python -m pi05.rlt.validate``  (use ``--help`` for knobs).
"""

from __future__ import annotations

import argparse
import math
import sys

import torch

from .buffer import ReplayBuffer
from .config import RLTConfig
from .interfaces import VLAWrapper
from .learner import AsyncLearner
from .safety import NativeUnitsSafetyLayer
from .transition import DoneType, SourceType
from .envs.unicycle import SyntheticEgomotion, SyntheticVLA, UnicycleEnv


def _smoke_cfg(seed: int) -> RLTConfig:
    """Smaller-but-faithful config so the gate runs in seconds, not minutes."""
    return RLTConfig(
        token_dim=128,
        dim_proprio=8,
        actor_hidden=(128, 128),
        critic_hidden=(128, 128),
        batch_size=128,
        utd=4,
        learning_starts=400,
        capacity=50_000,
        export_every=0,         # no disk export during the gate
        ref_dropout_p=0.5,
        seed=seed,
        device="cpu",
    )


def _act(actor, z_rl, s_p, a_ref, *, explore: bool) -> torch.Tensor:
    z, s, r = z_rl.unsqueeze(0), s_p.unsqueeze(0), a_ref.unsqueeze(0)
    with torch.no_grad():
        a = actor.rsample(z, s, r) if explore else actor.forward(z, s, r)
    return a.squeeze(0)


def _run_episode(
    env: UnicycleEnv,
    vla: VLAWrapper,
    ego: SyntheticEgomotion,
    buffer: ReplayBuffer,
    safety: NativeUnitsSafetyLayer,
    episode_id: int,
    *,
    actor=None,
    explore: bool = False,
    intervention_prob: float = 0.0,
    rng=None,
) -> DoneType:
    obs = env.reset()
    prev_head = None
    done = DoneType.NONE
    while True:
        z_rl, a_ref = vla.encode(obs)
        ego.set_obs(obs)
        s_p = ego.read()

        is_interv = bool(rng and rng.random() < intervention_prob)
        if actor is None or is_interv:
            a = a_ref.clone()                  # scripted reference / "human"
            source = SourceType.INTERVENTION if is_interv else SourceType.WARMUP
        else:
            a = _act(actor, z_rl, s_p, a_ref, explore=explore)
            source = SourceType.ONLINE

        state = {"prev_action": prev_head, "speed": obs["v"]}
        a, _ = safety.clamp(a, state)
        prev_head = a[0].clone()

        next_obs, reward, done = env.step(a[0])
        z_rl_next, a_ref_next = vla.encode(next_obs)
        ego.set_obs(next_obs)
        s_p_next = ego.read()

        buffer.add_step(
            episode_id, z_rl, s_p, a_ref, a,
            z_rl_next=z_rl_next, s_p_next=s_p_next, a_ref_next=a_ref_next,
            reward=0.0, source=source, is_intervention=is_interv,
        )
        obs = next_obs
        if done != DoneType.NONE:
            buffer.finalize_episode(episode_id, done, terminal_reward=reward)
            return done


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="RLT learner-core training-curve gate")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--online-episodes", type=int, default=300)
    p.add_argument("--intervention-prob", type=float, default=0.1)
    p.add_argument("--min-q-gain", type=float, default=0.05,
                   help="required probe-Q increase (last vs first window)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    cfg = _smoke_cfg(args.seed)
    rng = __import__("random").Random(args.seed)

    env = UnicycleEnv(cfg, seed=args.seed)
    vla = SyntheticVLA(cfg, seed=args.seed)
    ego = SyntheticEgomotion(cfg, seed=args.seed)
    safety = NativeUnitsSafetyLayer(cfg)
    buffer = ReplayBuffer(cfg)
    learner = AsyncLearner(cfg, buffer)

    eid = 0
    # --- warmup: scripted reference fills the buffer past learning_starts ---
    for _ in range(cfg.n_warmup_episodes):
        _run_episode(env, vla, ego, buffer, safety, eid, actor=None, rng=rng)
        eid += 1

    # --- online: actor + exploration + scripted interventions, interleaved
    #     with learner updates. Track probe-Q + success over the run. ---
    probe = None
    probe_q_hist: list[float] = []
    success_hist: list[int] = []
    for _ in range(args.online_episodes):
        done = _run_episode(
            env, vla, ego, buffer, safety, eid,
            actor=learner.actor, explore=True,
            intervention_prob=args.intervention_prob, rng=rng,
        )
        eid += 1
        success_hist.append(int(done == DoneType.SUCCESS))

        for _ in range(8):  # ~one learner step per env-step in this env
            metrics = learner.step()
            if metrics and not all(math.isfinite(v) for v in metrics.values()):
                print(f"FAIL: non-finite loss {metrics}", file=sys.stderr)
                return 1

        if buffer.ready():
            if probe is None:
                probe = buffer.sample(256)
            with torch.no_grad():
                q = learner.q.min_q(probe["z_rl"], probe["s_p"], probe["a"])
            probe_q_hist.append(float(q.mean()))

    if len(probe_q_hist) < 6:
        print("FAIL: learning never started (buffer under learning_starts)",
              file=sys.stderr)
        return 1

    w = len(probe_q_hist) // 3
    first = sum(probe_q_hist[:w]) / w
    last = sum(probe_q_hist[-w:]) / w
    first_sr = sum(success_hist[: len(success_hist) // 3]) / max(1, len(success_hist) // 3)
    last_sr = sum(success_hist[-len(success_hist) // 3:]) / max(1, len(success_hist) // 3)

    print(f"probe-Q: first={first:.3f} last={last:.3f} gain={last - first:+.3f}")
    print(f"success: first={first_sr:.2%} last={last_sr:.2%}")

    if last - first < args.min_q_gain:
        print(f"FAIL: probe-Q flatlined (gain {last - first:+.3f} "
              f"< {args.min_q_gain})", file=sys.stderr)
        return 1
    print("PASS: probe-Q trends up, losses finite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
