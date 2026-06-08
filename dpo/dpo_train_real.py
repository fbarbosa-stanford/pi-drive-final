#!/usr/bin/env python3
"""Real DPO post-training loop (gradient + optax + orbax) on Mark's BC ckpt.

Mirrors ``grpo/openpi_grpo_train.py`` but the per-clip signal is a *pairwise*
Alpamayo/AR1 preference instead of group advantages:

    per clip index:
      1. build model-space observation, tile to group size G
      2. sample G candidate action chunks from the current policy
      3. un-normalize -> physical -> Alpamayo picks chosen (min ADE) vs rejected (max ADE)
      4. DPO gradient step on the chosen/rejected pair (paired noise + frozen BC reference)
    periodically + at end: save an orbax checkpoint compare_fifty / eval_fifty can load

Run under the openpi venv (see ``dpo/modal_posttrain.py::train_dpo_real``).
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

from dpo.alpamayo_preference import AlpamayoPreferenceConfig, pick_preference_pair
from dpo.constants import OPENPI_CONFIG_NAME
from dpo.context_builders import make_smoke_context
from grpo.eval_indices import pick_eval_indices
from grpo.openpi_grpo_runner import _context_from_sample, _ensure_checkpoint_layout
from grpo.openpi_grpo_train import build_datasets, prepare_group_observation, unnormalize_actions
from rewards.flat_actions import FLAT_ACTION_DIM, unflatten_actions


def _load_label_for_index(index: int, labels_path: str | None):
    if not labels_path or not Path(labels_path).exists():
        return None
    from grpo.label_cache import LabelCache

    records = LabelCache(labels_path).load()
    return records[index] if index < len(records) else None


def train(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    dataset_root: str,
    out_dir: str,
    num_clips: int,
    group_size: int,
    save_interval: int,
    seed: int,
    beta: float,
    peak_lr: float = 1e-6,
    repo_id: str | None = None,
    labels_path: str | None = None,
    exp_name: str = "dpo-posttrain",
    resume_from: str | None = None,
) -> dict:
    import gc as _gc

    import jax
    import jax.numpy as jnp
    import openpi.training.config as _config

    from dpo.trainable_dpo import make_dpo_train_step, make_ref_logp
    from grpo.trainable_state import build_train_state, make_sampler, save_checkpoint

    _ensure_checkpoint_layout(ckpt_dir, openpi_dir)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", str(Path(ckpt_dir) / "params"))
    config = _config.get_config(OPENPI_CONFIG_NAME)

    print(f"[dpo] building train state from {ckpt_dir} (peak_lr={peak_lr})", flush=True)
    state, _tx = build_train_state(config, seed=seed, peak_lr=peak_lr, resume_from=resume_from)
    if resume_from is not None:
        # On resume, `state` is the partially-trained policy; the DPO reference must still
        # anchor to the original BC weights, so load a fresh BC state for it.
        print(f"[dpo] resumed from {resume_from}; loading fresh BC for reference", flush=True)
        ref_state, _ = build_train_state(config, seed=seed, peak_lr=peak_lr)
    else:
        ref_state = state  # frozen BC reference (initial params; immutable pytree)
    start_step = int(state.step)
    frames, data_config = build_datasets(config, dataset_root=dataset_root, repo_id=repo_id)
    dataset_len = len(frames)
    print(f"[dpo] dataset len={dataset_len}, group_size={group_size}, clips={num_clips}", flush=True)

    sampler = make_sampler(config)
    dpo_step = make_dpo_train_step(config, beta=beta)
    ref_logp = make_ref_logp(config)
    pref_cfg = AlpamayoPreferenceConfig()

    # Fresh clips + randomness per segment (so resumed segments don't repeat the same clips).
    indices = pick_eval_indices(dataset_len, max(num_clips, 1), seed=seed + start_step)
    if not indices:
        indices = [0]

    history = []
    rng = jax.random.key(seed + start_step)
    ad = config.model.action_dim
    ah = config.model.action_horizon
    for step in range(num_clips):
        idx = indices[step % len(indices)]
        item = frames.model_item(idx)
        raw_sample = frames.row(idx)

        obs = prepare_group_observation(item, group_size)
        rng, sample_rng, loss_rng = jax.random.split(rng, 3)
        noise = jax.random.normal(sample_rng, (group_size, ah, ad))
        norm_actions = sampler(state, sample_rng, obs, noise)  # (G, ah, ad)
        phys_flat = unnormalize_actions(np.asarray(norm_actions), data_config)  # (G, 128)

        gt_flat = np.asarray(raw_sample["action"], dtype=np.float32).reshape(-1)[:FLAT_ACTION_DIM]
        label = _load_label_for_index(idx, labels_path)
        expert_xyz = None
        if label is not None:
            from dpo.context_builders import context_from_label_record

            context = context_from_label_record(label)
            if getattr(label, "expert_xyz", None) is not None:
                expert_xyz = np.asarray(label.expert_xyz, dtype=np.float32)
        else:
            context = _context_from_sample(raw_sample, gt_flat)
            if context.ar1_trajs is None:
                context = make_smoke_context(unflatten_actions(gt_flat))

        candidates = [unflatten_actions(c) for c in phys_flat]
        pair = pick_preference_pair(candidates, context=context, expert_xyz=expert_xyz, config=pref_cfg)
        if pair is None:
            history.append({"step": step, "index": int(idx), "status": "no_preference_pair"})
            continue

        # (1, ah, ad) chosen / rejected; obs1 = single-sample slice of the tiled obs.
        chosen = np.asarray(norm_actions)[pair.chosen_idx][None]
        rejected = np.asarray(norm_actions)[pair.rejected_idx][None]
        obs1 = jax.tree.map(lambda x: x[:1], obs)
        chosen = jnp.asarray(chosen)
        rejected = jnp.asarray(rejected)
        rw, rl = ref_logp(ref_state, loss_rng, obs1, chosen, rejected)
        state, metrics = dpo_step(state, loss_rng, obs1, chosen, rejected, rw, rl)
        # Release per-step device buffers promptly (DPO leaked them, OOMing ~step 1100).
        del obs, obs1, norm_actions, noise, chosen, rejected, rw, rl
        if (step + 1) % 20 == 0:
            _gc.collect()

        row = {
            "step": step,
            "index": int(idx),
            "status": "ok",
            "chosen_idx": pair.chosen_idx,
            "rejected_idx": pair.rejected_idx,
            "margin_m": float(pair.margin_m),
            **{k: float(v) for k, v in metrics.items()},
        }
        history.append(row)
        if step % max(1, save_interval // 5) == 0 or step == num_clips - 1:
            print(json.dumps(row), flush=True)

        # Use the GLOBAL step (state.step continues across resumes) for save naming.
        if save_interval > 0 and int(state.step) % save_interval == 0:
            import gc as _gc
            _gc.collect()  # free GPU headroom before the save spike
            print(f"[dpo] saved checkpoint -> {save_checkpoint(state, config, out_dir, int(state.step))}", flush=True)

    import gc as _gc
    _gc.collect()
    final = save_checkpoint(state, config, out_dir, int(state.step))  # global (start_step + num_clips)
    print(f"[dpo] final checkpoint -> {final}", flush=True)

    n_ok = sum(1 for r in history if r.get("status") == "ok")
    report = {
        "exp_name": exp_name,
        "ckpt_dir": ckpt_dir,
        "out_dir": out_dir,
        "final_checkpoint": final,
        "num_clips": num_clips,
        "n_ok": n_ok,
        "group_size": group_size,
        "beta": beta,
        "indices": [int(i) for i in indices],
        "history": history,
    }
    report_dir = Path("/cache/dpo_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{exp_name}_train.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"[dpo] wrote {report_path} (n_ok={n_ok}/{num_clips})", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO post-training (real gradient step)")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out-dir", default=f"/cache/checkpoints/{OPENPI_CONFIG_NAME}/dpo-posttrain")
    parser.add_argument("--num-clips", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beta", type=float, default=100.0)  # Diffusion-DPO effective coef (β·T)
    parser.add_argument(
        "--peak-lr",
        type=float,
        default=1e-6,
        help="Post-train LR (BC config peak 3e-5 is too hot and diverges)",
    )
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--labels-path", default=os.environ.get("AR1_LABELS_PATH"))
    parser.add_argument("--exp-name", default="dpo-posttrain")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Path to a checkpoint step dir (e.g. .../dpo-5k-v2/1000) to resume params+opt_state+step from",
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
        num_clips=args.num_clips,
        group_size=args.group_size,
        save_interval=args.save_interval,
        seed=args.seed,
        beta=args.beta,
        peak_lr=args.peak_lr,
        repo_id=args.repo_id,
        labels_path=args.labels_path,
        exp_name=args.exp_name,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    import jax  # noqa: F401

    main()
