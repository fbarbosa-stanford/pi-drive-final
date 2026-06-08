#!/usr/bin/env python3
"""Stage C: DPO post-training on cached Cosmos-3-judged preference pairs (offline DPO).

Unlike ``dpo_train_real.py`` (which samples + ranks online by ADE-to-GT each step), this
loads pre-computed (chosen, rejected) pairs produced by:
  1. ``grpo/modal_posttrain.py::gen_judge_candidates``  (sample G=2 from BC, render)
  2. ``dpo/modal_cosmos_judge.py::judge_candidates``     (Cosmos picks BETTER=1/2)

Each step: load a cached chosen/rejected NORMALIZED action pair for a clip, rebuild the clip
observation, and take a Diffusion-DPO gradient step (paired noise + frozen BC reference). The
candidates are frozen (sampled from BC at labeling time) -> standard offline DPO.
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

from dpo.constants import OPENPI_CONFIG_NAME
from grpo.openpi_grpo_runner import _ensure_checkpoint_layout
from grpo.openpi_grpo_train import build_datasets, prepare_group_observation


def _load_cosmos_pairs(labels_path: str, stage_dir: str):
    """Return [(index, chosen_norm(ah,ad), rejected_norm(ah,ad))] from the Cosmos labels + npz."""
    cands = Path(stage_dir) / "cands"
    pairs = []
    with open(labels_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            npz = cands / f"{r['stem']}.npz"
            if not npz.exists():
                continue
            norm = np.load(npz, allow_pickle=True)["norm"]  # (G, ah, ad)
            ci, ri = int(r["chosen_idx"]), int(r["rejected_idx"])
            if max(ci, ri) >= norm.shape[0]:
                continue
            pairs.append((int(r["index"]), norm[ci].astype(np.float32), norm[ri].astype(np.float32)))
    return pairs


def train(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    dataset_root: str,
    out_dir: str,
    labels_path: str,
    stage_dir: str,
    num_steps: int,
    save_interval: int,
    seed: int,
    beta: float,
    peak_lr: float = 1e-6,
    repo_id: str | None = None,
    exp_name: str = "dpo-cosmos",
    anchor_lambda: float = 0.0,
) -> dict:
    import gc as _gc

    import jax
    import jax.numpy as jnp
    import openpi.training.config as _config

    from dpo.trainable_dpo import make_dpo_train_step, make_ref_logp
    from grpo.trainable_state import build_train_state, save_checkpoint

    _ensure_checkpoint_layout(ckpt_dir, openpi_dir)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", str(Path(ckpt_dir) / "params"))
    config = _config.get_config(OPENPI_CONFIG_NAME)

    print(f"[cosmos-dpo] building BC train state from {ckpt_dir} (peak_lr={peak_lr})", flush=True)
    state, _tx = build_train_state(config, seed=seed, peak_lr=peak_lr)
    ref_state = state  # frozen BC reference
    frames, _data_config = build_datasets(config, dataset_root=dataset_root, repo_id=repo_id)

    pairs = _load_cosmos_pairs(labels_path, stage_dir)
    if not pairs:
        raise RuntimeError(f"no usable Cosmos pairs from {labels_path} (+ {stage_dir}/cands)")
    print(f"[cosmos-dpo] loaded {len(pairs)} chosen/rejected pairs; steps={num_steps} "
          f"beta={beta} anchor_lambda={anchor_lambda}", flush=True)

    dpo_step = make_dpo_train_step(config, beta=beta, anchor_lambda=anchor_lambda)
    ref_logp = make_ref_logp(config)

    history = []
    rng = jax.random.key(seed)
    for step in range(num_steps):
        index, chosen_a, rejected_a = pairs[step % len(pairs)]
        item = frames.model_item(index)
        obs1 = prepare_group_observation(item, 1)  # single-sample obs for this clip
        chosen = jnp.asarray(chosen_a[None])  # (1, ah, ad)
        rejected = jnp.asarray(rejected_a[None])
        rng, loss_rng = jax.random.split(rng)
        rw, rl = ref_logp(ref_state, loss_rng, obs1, chosen, rejected)
        state, metrics = dpo_step(state, loss_rng, obs1, chosen, rejected, rw, rl)
        del obs1, chosen, rejected, rw, rl
        if (step + 1) % 20 == 0:
            _gc.collect()

        row = {"step": step, "index": int(index), **{k: float(v) for k, v in metrics.items()}}
        history.append(row)
        if step % max(1, save_interval // 5) == 0 or step == num_steps - 1:
            print(json.dumps(row), flush=True)

        if save_interval > 0 and (step + 1) % save_interval == 0:
            _gc.collect()
            print(f"[cosmos-dpo] saved -> {save_checkpoint(state, config, out_dir, step + 1)}", flush=True)

    _gc.collect()
    final = save_checkpoint(state, config, out_dir, int(num_steps))
    print(f"[cosmos-dpo] final checkpoint -> {final}", flush=True)

    report = {
        "exp_name": exp_name, "ckpt_dir": ckpt_dir, "out_dir": out_dir,
        "final_checkpoint": final, "num_steps": num_steps, "n_pairs": len(pairs),
        "beta": beta, "labels_path": labels_path, "history": history,
    }
    rep_dir = Path("/cache/dpo_reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{exp_name}_train.json").write_text(json.dumps(report, indent=2))
    print(f"[cosmos-dpo] wrote report ({len(pairs)} pairs, {num_steps} steps)", flush=True)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="DPO on Cosmos-judged preference pairs")
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--labels-path", required=True)
    p.add_argument("--stage-dir", required=True, help="cosmos_stage/<tag> dir holding cands/*.npz")
    p.add_argument("--out-dir", default=f"/cache/checkpoints/{OPENPI_CONFIG_NAME}/dpo-cosmos")
    p.add_argument("--num-steps", type=int, default=5000)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beta", type=float, default=100.0)
    p.add_argument("--peak-lr", type=float, default=1e-6)
    p.add_argument("--anchor-lambda", type=float, default=0.0,
                   help="DPO+NLL anchor weight on flow_chosen (prevents flow-field collapse)")
    p.add_argument("--repo-id", default=None)
    p.add_argument("--exp-name", default="dpo-cosmos")
    args = p.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(openpi_dir)
    train(
        openpi_dir=openpi_dir, ckpt_dir=args.ckpt_dir, dataset_root=args.dataset_root,
        out_dir=args.out_dir, labels_path=args.labels_path, stage_dir=args.stage_dir,
        num_steps=args.num_steps, save_interval=args.save_interval, seed=args.seed,
        beta=args.beta, peak_lr=args.peak_lr, repo_id=args.repo_id, exp_name=args.exp_name,
        anchor_lambda=args.anchor_lambda,
    )


if __name__ == "__main__":
    import jax  # noqa: F401

    main()
