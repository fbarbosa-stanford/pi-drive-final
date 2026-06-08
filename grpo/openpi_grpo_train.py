#!/usr/bin/env python3
"""Real Flow-GRPO post-training loop (gradient + optax + orbax) on Mark's BC ckpt.

Unlike ``openpi_grpo_runner.py`` (ranking + numpy surrogate loss only, policy never
changes), this loop actually updates the weights:

    per clip index:
      1. build model-space observation, tile to group size G
      2. sample G candidate action chunks from the *current* policy
      3. un-normalize -> physical -> composite reward -> group advantages
      4. jitted gradient step (advantage-weighted flow-matching loss)
    periodically + at end: save an orbax checkpoint compare_fifty can load

Run under the openpi venv (see ``grpo/modal_posttrain.py::train_grpo_real``)::

    /opt/openpi/.venv/bin/python grpo/openpi_grpo_train.py \
        --ckpt-dir /cache/checkpoints/bc_hf/... \
        --dataset-root /cache/hf/lerobot/markmusic--pi05-physical-av-bc \
        --num-steps 500 --group-size 8 --save-interval 100 \
        --out-dir /cache/checkpoints/pi05_driving/grpo-posttrain
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from grpo.constants import LEROBOT_TOLERANCE_S, OPENPI_CONFIG_NAME
from grpo.eval_indices import pick_eval_indices
from grpo.flow_grpo import FlowGRPOConfig, FlowGRPOTrainer
from grpo.openpi_grpo_runner import _context_from_sample, _ensure_checkpoint_layout
from rewards.flat_actions import FLAT_ACTION_DIM, unflatten_actions


def build_datasets(config, *, dataset_root, repo_id=None):
    """Return ``(frames, data_config)`` — a parquet-backed ``FrameSource``.

    We bypass ``LeRobotDataset`` (non-contiguous episode_index + broken hf_xet download
    crash its frame loader) and read parquet rows directly, applying openpi's transforms
    to get model-space items. ``frames.row(i)`` gives the physical GT action / prompt for
    reward context; ``frames.model_item(i)`` gives the normalized/tokenized model input.
    """
    from grpo.parquet_frames import FrameSource

    data_config = config.data.create(config.assets_dirs, config.model)
    frames = FrameSource(dataset_root, data_config)
    print(f"[data] {frames.describe()}", flush=True)
    return frames, data_config


def _leaf_to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def prepare_group_observation(transformed_item: dict, group_size: int):
    """Convert a transformed item to a batched (B==G) ``Observation`` of jax arrays."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    def tile(arr):
        arr = _leaf_to_np(arr)
        return np.broadcast_to(arr[None], (group_size, *arr.shape)).copy()

    batched = {}
    for key, val in transformed_item.items():
        if isinstance(val, dict):
            batched[key] = {k: tile(v) for k, v in val.items()}
        else:
            batched[key] = tile(val)

    obs_dict = {k: v for k, v in batched.items() if k != "actions"}
    obs = _model.Observation.from_dict(obs_dict)
    return jax.tree.map(jnp.asarray, obs)


def unnormalize_actions(norm_chunk_G, data_config):
    """(G, action_horizon, action_dim) normalized -> (G, FLAT_ACTION_DIM) physical.

    Norm stats are per action_dim (=32); they broadcast over the action_horizon (=4)
    axis. We unnormalize on the last axis, then flatten the chunk to 128 for reward
    scoring. Self-contained (no strict-mode Unnormalize); inverts ``Normalize`` exactly.
    """
    chunk = np.asarray(norm_chunk_G, dtype=np.float32)  # (G, ah, ad)
    stats_map = data_config.norm_stats or {}
    stats = stats_map.get("actions") or stats_map.get("action")
    if stats is None:
        return chunk.reshape(chunk.shape[0], -1)[:, :FLAT_ACTION_DIM]
    ad = chunk.shape[-1]
    if data_config.use_quantile_norm and getattr(stats, "q01", None) is not None:
        q01 = np.asarray(stats.q01, np.float32)[:ad]
        q99 = np.asarray(stats.q99, np.float32)[:ad]
        phys = (chunk + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    else:
        mean = np.asarray(stats.mean, np.float32)[:ad]
        std = np.asarray(stats.std, np.float32)[:ad]
        phys = chunk * (std + 1e-6) + mean
    return phys.reshape(chunk.shape[0], -1)[:, :FLAT_ACTION_DIM]


def train(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    dataset_root: str,
    out_dir: str,
    num_steps: int,
    group_size: int,
    save_interval: int,
    seed: int,
    repo_id: str | None = None,
    kl_coef: float = 1.0,
    peak_lr: float = 1e-6,
    resume_from: str | None = None,
    exp_name: str = "grpo-posttrain",
) -> dict:
    import jax
    import openpi.training.config as _config

    from grpo.trainable_state import (
        build_train_state,
        make_grpo_train_step,
        make_ref_flow,
        make_sampler,
        save_checkpoint,
    )

    _ensure_checkpoint_layout(ckpt_dir, openpi_dir)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", str(Path(ckpt_dir) / "params"))
    config = _config.get_config(OPENPI_CONFIG_NAME)

    print(f"[grpo] building train state from {ckpt_dir} (peak_lr={peak_lr}, resume_from={resume_from})", flush=True)
    state, _tx = build_train_state(config, seed=seed, peak_lr=peak_lr, resume_from=resume_from)
    # KL anchor stays the BC reference. On resume, load BC fresh (state is the resumed policy).
    if resume_from is not None:
        ref_state, _ = build_train_state(config, seed=seed, peak_lr=peak_lr)
    else:
        ref_state = state
    start_step = int(state.step)
    frames, data_config = build_datasets(config, dataset_root=dataset_root, repo_id=repo_id)
    dataset_len = len(frames)
    print(f"[grpo] dataset len={dataset_len}, group_size={group_size}, steps={num_steps}, kl_coef={kl_coef}", flush=True)

    sampler = make_sampler(config)
    ref_flow_fn = make_ref_flow(config)
    train_step = make_grpo_train_step(config, kl_coef=kl_coef)
    ranker = FlowGRPOTrainer(FlowGRPOConfig(group_size=group_size, kl_coef=kl_coef))

    # Fresh clips + randomness per segment (so resumed segments don't repeat the same clips).
    indices = pick_eval_indices(dataset_len, max(num_steps, 1), seed=seed + start_step)
    if not indices:
        indices = [0]

    history = []
    rng = jax.random.key(seed + start_step)
    for step in range(num_steps):
        idx = indices[step % len(indices)]
        item = frames.model_item(idx)
        raw_sample = frames.row(idx)

        obs = prepare_group_observation(item, group_size)
        ad = config.model.action_dim
        ah = config.model.action_horizon
        rng, sample_rng, step_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(sample_rng, (group_size, ah, ad))
        norm_actions = sampler(state, sample_rng, obs, noise)  # (G, ah, ad) normalized
        phys_flat = unnormalize_actions(np.asarray(norm_actions), data_config)  # (G, 128)

        gt_flat = np.asarray(raw_sample["action"], dtype=np.float32).reshape(-1)[:FLAT_ACTION_DIM]
        context = _context_from_sample(raw_sample, gt_flat)
        candidates = [unflatten_actions(c) for c in phys_flat]
        ranking = ranker.run_composite_ranking_step(candidates, context)

        advantages = jax.numpy.asarray(ranking.advantages)
        # Reference flow under frozen BC, SAME rng as the policy step -> paired noise for the KL term.
        flow_ref = ref_flow_fn(ref_state, step_rng, obs, norm_actions)
        state, metrics = train_step(state, step_rng, obs, norm_actions, advantages, flow_ref)
        del obs, norm_actions, noise, flow_ref, advantages  # release per-step device buffers
        if (step + 1) % 20 == 0:
            import gc as _gc
            _gc.collect()

        row = {
            "step": step,
            "index": int(idx),
            "reward_mean": float(ranking.summary["reward_mean"]),
            "reward_std": float(ranking.summary["reward_std"]),
            **{k: float(v) for k, v in metrics.items()},
        }
        history.append(row)
        if step % max(1, save_interval // 5) == 0 or step == num_steps - 1:
            print(json.dumps(row), flush=True)

        # Use the GLOBAL step (state.step continues across resumes) for save naming.
        if save_interval > 0 and int(state.step) % save_interval == 0:
            import gc as _gc
            _gc.collect()  # free GPU headroom before the save spike
            saved = save_checkpoint(state, config, out_dir, int(state.step))
            print(f"[grpo] saved checkpoint -> {saved}", flush=True)

    import gc as _gc
    _gc.collect()
    final_step = int(state.step)  # global (start_step + num_steps)
    final = save_checkpoint(state, config, out_dir, final_step)
    print(f"[grpo] final checkpoint -> {final}", flush=True)

    report = {
        "exp_name": exp_name,
        "ckpt_dir": ckpt_dir,
        "out_dir": out_dir,
        "final_checkpoint": final,
        "num_steps": num_steps,
        "group_size": group_size,
        "kl_coef": kl_coef,
        "indices": [int(i) for i in indices],
        "history": history,
    }
    report_dir = Path("/cache/grpo_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{exp_name}_train.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[grpo] wrote {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-GRPO post-training (real gradient step)")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default=f"/cache/checkpoints/{OPENPI_CONFIG_NAME}/grpo-posttrain")
    parser.add_argument("--num-steps", type=int, default=500)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kl-coef", type=float, default=1.0)
    parser.add_argument("--peak-lr", type=float, default=1e-6)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--exp-name", default="grpo-posttrain")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint step dir (e.g. .../grpo-5k-v2/2500) to resume params+opt_state+step from",
    )
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(openpi_dir)
    train(
        openpi_dir=openpi_dir,
        ckpt_dir=args.ckpt_dir,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        num_steps=args.num_steps,
        group_size=args.group_size,
        save_interval=args.save_interval,
        seed=args.seed,
        repo_id=args.repo_id,
        kl_coef=args.kl_coef,
        peak_lr=args.peak_lr,
        exp_name=args.exp_name,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    import jax  # noqa: F401  (ensure venv import works before train())

    main()
