#!/usr/bin/env python3
"""Flow-GRPO step with Mark's openpi BC checkpoint (run on Modal GPU)."""

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

from grpo.constants import HF_BC_DATASET_REPO, LEROBOT_TOLERANCE_S, OPENPI_CONFIG_NAME
from grpo.context_builders import make_straight_route
from grpo.eval_metrics import eval_candidate_vs_references
from grpo.flow_grpo import FlowGRPOConfig, FlowGRPOTrainer
from grpo.objective import compute_flow_grpo_loss
from rewards.flat_actions import unflatten_actions
from rewards.reward_context import RewardContext


def _ensure_checkpoint_layout(ckpt_dir: str, openpi_dir: str) -> Path:
    """HF orbax upload may be flat; openpi expects checkpoint_dir/params and .../assets."""
    root = Path(ckpt_dir)
    params_dir = root / "params"
    if not params_dir.is_dir():
        if (root / "manifest.ocdbt").exists() or (root / "_METADATA").exists():
            params_dir.mkdir(parents=True, exist_ok=True)
            for item in root.iterdir():
                if item.name in ("params", "assets"):
                    continue
                dest = params_dir / item.name
                if not dest.exists():
                    dest.symlink_to(item.resolve())
        else:
            params_dir = root

    asset_id = HF_BC_DATASET_REPO
    assets_dst = root / "assets" / asset_id
    if not assets_dst.is_dir():
        assets_src = Path(openpi_dir) / "assets" / OPENPI_CONFIG_NAME / asset_id
        if assets_src.is_dir():
            assets_dst.parent.mkdir(parents=True, exist_ok=True)
            assets_dst.symlink_to(assets_src.resolve())
        else:
            print(f"WARNING: norm stats missing at {assets_src}")

    return root


def _decode_lerobot_image(img) -> np.ndarray:
    """HF parquet may store images as dict(bytes=...) before torch transform."""
    if isinstance(img, np.ndarray):
        return img
    if hasattr(img, "convert"):
        return np.asarray(img.convert("RGB"))
    if isinstance(img, dict):
        raw = img.get("bytes")
        if raw is not None:
            from io import BytesIO

            from PIL import Image

            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        path = img.get("path")
        if path:
            from PIL import Image

            return np.asarray(Image.open(path).convert("RGB"))
    return np.asarray(img)


def _lerobot_obs_from_sample(sample: dict) -> dict:
    """Map LeRobot frame → policy infer() input."""
    raw_img = sample.get("observation.images.front") or sample.get("observation/image")
    img = _decode_lerobot_image(raw_img)
    state = sample.get("observation.state")
    if hasattr(state, "detach"):
        state = state.detach().cpu().numpy()
    prompt = sample.get("prompt") or sample.get("task") or "continue straight"
    return {
        "observation/image": img,
        "observation/state": state,
        "prompt": prompt,
    }


def _context_from_sample(sample: dict, gt_flat: np.ndarray) -> RewardContext:
    prompt = str(sample.get("prompt", sample.get("task", "continue straight")))
    tags = ["continue_straight"]
    pl = prompt.lower()
    if "left" in pl:
        tags = ["turn_left"]
    elif "right" in pl:
        tags = ["turn_right"]
    gt_chunk = unflatten_actions(gt_flat)
    state = np.asarray(sample.get("observation.state", [8.0, 0.0]), dtype=np.float32)
    return RewardContext(
        coc_tags=tags,
        coc_text=prompt,
        ar1_trajs=gt_chunk[np.newaxis, ...],
        route_polyline=make_straight_route(),
        initial_speed=float(state[0]) if state.size else 8.0,
        lead_vehicle_xy=np.array([30.0, 0.0], dtype=np.float32),
    )


def load_policy_and_dataset(
    openpi_dir: str,
    ckpt_dir: str,
    *,
    dataset_root: str | Path,
    repo_id: str | None = None,
):
    """Load openpi policy + LeRobot dataset from local volume (``root`` required)."""
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.config as _config
    from openpi.policies import policy_config

    _ensure_checkpoint_layout(ckpt_dir, openpi_dir)
    train_config = _config.get_config(OPENPI_CONFIG_NAME)
    policy = policy_config.create_trained_policy(train_config, ckpt_dir)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    rid = repo_id or data_config.repo_id
    root_kw = {"root": str(dataset_root)}
    meta = lerobot_dataset.LeRobotDatasetMetadata(rid, **root_kw)
    delta_timestamps = {
        key: [t / meta.fps for t in range(train_config.model.action_horizon)]
        for key in data_config.action_sequence_keys
    }
    dataset = lerobot_dataset.LeRobotDataset(
        rid,
        delta_timestamps=delta_timestamps,
        tolerance_s=LEROBOT_TOLERANCE_S,
        **root_kw,
    )
    return policy, dataset, train_config


def run_step(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    group_size: int,
    dataset_index: int,
    dataset_root: str | Path | None = None,
    noise_seed: int = 42,
    policy=None,
    dataset=None,
) -> dict:
    from openpi_patches.patch_openpi import prepend_openpi_venv

    if policy is None or dataset is None:
        prepend_openpi_venv(openpi_dir or os.environ.get("OPENPI_DIR", "/opt/openpi"))
        policy, dataset, _ = load_policy_and_dataset(
            openpi_dir, ckpt_dir, dataset_root=dataset_root
        )

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
    context = _context_from_sample(sample, gt_flat)

    trainer = FlowGRPOTrainer(FlowGRPOConfig(group_size=group_size))
    ranking = trainer.run_composite_ranking_step(candidates, context)

    metrics_list = [
        eval_candidate_vs_references(
            c, gt_flat=gt_flat, ar1_chunks=[context.ar1_trajs[0]],
        )
        for c in candidates_flat
    ]

    inputs = trainer.build_objective_inputs(ranking, flow_losses=flow_losses)
    obj_out = compute_flow_grpo_loss(inputs)

    return {
        "group_size": group_size,
        "dataset_index": dataset_index,
        "reward_std": ranking.summary["reward_std"],
        "rewards": ranking.rewards,
        "advantages": ranking.advantages.tolist(),
        "flow_losses": flow_losses,
        "metrics": metrics_list,
        "flow_grpo_loss": obj_out.loss,
        "objective_metrics": obj_out.metrics,
        "best_index": int(ranking.summary["best_index"]),
        "worst_index": int(ranking.summary["worst_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--dataset-root", default=None, help="Local LeRobot dataset root on volume")
    parser.add_argument("--num-steps", type=int, default=0, help="0 = single smoke step only")
    parser.add_argument("--exp-name", default="grpo-smoke")
    parser.add_argument("--noise-seed", type=int, default=42)
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    dataset_root = args.dataset_root or os.environ.get("LEROBOT_DATASET_ROOT")

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(openpi_dir)
    policy, dataset, _ = load_policy_and_dataset(
        openpi_dir, args.ckpt_dir, dataset_root=dataset_root
    )
    dataset_len = len(dataset)

    reports = []
    steps = max(1, args.num_steps) if args.num_steps > 0 else 1
    for step in range(steps):
        report = run_step(
            openpi_dir=openpi_dir,
            ckpt_dir=args.ckpt_dir,
            group_size=args.group_size,
            dataset_index=(args.dataset_index + step) % dataset_len,
            dataset_root=dataset_root,
            noise_seed=args.noise_seed,
            policy=policy,
            dataset=dataset,
        )
        report["step"] = step
        reports.append(report)
        print(json.dumps(report, indent=2))

    out_dir = Path("/cache/grpo_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.exp_name}_last.json"
    with out_path.open("w") as f:
        json.dump(reports[-1] if len(reports) == 1 else reports, f, indent=2)

    # Smoke-only gate: training runs may have low spread on some clips
    if args.num_steps == 0 and reports[-1]["reward_std"] < 1e-6:
        sys.exit(1)


if __name__ == "__main__":
    main()
