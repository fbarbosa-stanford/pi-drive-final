#!/usr/bin/env python3
"""One DPO step: sample G actions, AR1 picks best/worst, compute DPO loss."""

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

from dpo.constants import HF_BC_DATASET_REPO, OPENPI_CONFIG_NAME
from grpo.constants import LEROBOT_TOLERANCE_S
from dpo.context_builders import make_smoke_context
from dpo.trainer import DPOConfig, DPOTrainer
from grpo.openpi_grpo_runner import (
    _context_from_sample,
    _ensure_checkpoint_layout,
    _lerobot_obs_from_sample,
)
from rewards.flat_actions import flatten_actions, unflatten_actions


def _load_label_for_index(index: int, labels_path: str | None) -> object | None:
    if not labels_path or not Path(labels_path).exists():
        return None
    from grpo.label_cache import LabelCache

    records = LabelCache(labels_path).load()
    if index < len(records):
        return records[index]
    return None


def load_policy_and_dataset(
    openpi_dir: str,
    ckpt_dir: str,
    *,
    dataset_root: str | Path,
    repo_id: str | None = None,
):
    """Load openpi policy + LeRobot dataset from local volume (no HF eval repo)."""
    from grpo.openpi_grpo_runner import load_policy_and_dataset as _load

    return _load(openpi_dir, ckpt_dir, dataset_root=dataset_root, repo_id=repo_id)


def run_step(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    group_size: int,
    dataset_index: int,
    labels_path: str | None = None,
    noise_seed: int = 42,
    policy=None,
    dataset=None,
) -> dict:
    if policy is None or dataset is None:
        policy, dataset, _ = load_policy_and_dataset(openpi_dir, ckpt_dir)

    sample = dataset[dataset_index]
    obs = _lerobot_obs_from_sample(sample)
    gt_flat = np.asarray(sample.get("action", sample.get("actions")), dtype=np.float32).reshape(-1)

    rng = np.random.default_rng(noise_seed + dataset_index)
    candidates_flat = []
    flow_losses = []
    for _ in range(group_size):
        noise = rng.standard_normal((1, gt_flat.size), dtype=np.float32)
        out = policy.infer(obs, noise=noise)
        flat = np.asarray(out["actions"], dtype=np.float32).reshape(-1)
        candidates_flat.append(flat)
        flow_losses.append(float(np.mean((flat - gt_flat) ** 2)))

    candidates = [unflatten_actions(c) for c in candidates_flat]
    label_record = _load_label_for_index(dataset_index, labels_path)
    if label_record is not None:
        from dpo.context_builders import context_from_label_record

        context = context_from_label_record(label_record)
        expert_xyz = (
            np.asarray(label_record.expert_xyz, dtype=np.float32)
            if label_record.expert_xyz is not None
            else None
        )
    else:
        context = _context_from_sample(sample, gt_flat)
        expert_xyz = None
        if context.ar1_trajs is None:
            context = make_smoke_context(unflatten_actions(gt_flat))

    trainer = DPOTrainer(DPOConfig(group_size=group_size))
    result = trainer.run_preference_step(
        candidates,
        context,
        flow_losses,
        label_record=label_record,
        expert_xyz=expert_xyz,
    )
    if result is None:
        return {
            "dataset_index": dataset_index,
            "group_size": group_size,
            "status": "no_preference_pair",
            "flow_losses": flow_losses,
        }

    ranking, obj_out = result
    pair = ranking.pair
    from grpo.eval_metrics import eval_candidate_vs_references

    chosen_flat = candidates_flat[pair.chosen_idx]
    rejected_flat = candidates_flat[pair.rejected_idx]
    chosen_gt = eval_candidate_vs_references(chosen_flat, gt_flat=gt_flat)
    rejected_gt = eval_candidate_vs_references(rejected_flat, gt_flat=gt_flat)

    return {
        "dataset_index": dataset_index,
        "group_size": group_size,
        "status": "ok",
        "preference_source": pair.source,
        "chosen_idx": pair.chosen_idx,
        "rejected_idx": pair.rejected_idx,
        "chosen_ar1_ade_m": pair.chosen_score,
        "rejected_ar1_ade_m": pair.rejected_score,
        "margin_m": pair.margin_m,
        "all_ar1_scores_m": pair.all_scores,
        "chosen_gt_ade_m": chosen_gt.get("gt_ade_m"),
        "rejected_gt_ade_m": rejected_gt.get("gt_ade_m"),
        "chosen_gt_action_mse": chosen_gt.get("gt_action_mse"),
        "flow_losses": flow_losses,
        "dpo_loss": obj_out.loss,
        "objective_metrics": obj_out.metrics,
        "used_label_cache": label_record is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=0)
    parser.add_argument("--exp-name", default="dpo-smoke")
    parser.add_argument(
        "--labels-path",
        default=os.environ.get("AR1_LABELS_PATH", "/cache/labels/ar1_labels.jsonl"),
    )
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    reports = []
    steps = max(1, args.num_steps) if args.num_steps > 0 else 1
    for step in range(steps):
        report = run_step(
            openpi_dir=openpi_dir,
            ckpt_dir=args.ckpt_dir,
            group_size=args.group_size,
            dataset_index=(args.dataset_index + step) % 100,
            labels_path=args.labels_path,
        )
        report["step"] = step
        reports.append(report)
        print(json.dumps(report, indent=2))

    out_dir = Path("/cache/dpo_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.exp_name}_last.json"
    with out_path.open("w") as f:
        json.dump(reports[-1] if len(reports) == 1 else reports, f, indent=2)

    if reports[-1].get("status") == "no_preference_pair":
        sys.exit(1)


if __name__ == "__main__":
    main()
