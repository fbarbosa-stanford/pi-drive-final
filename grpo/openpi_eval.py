#!/usr/bin/env python3
"""Open-loop eval: compare BC vs post-trained checkpoints on N eval samples."""

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

from grpo.constants import (
    FLAT_ACTION_DIM,
    HF_BC_DATASET_REPO,
    HF_BC_EVAL_DATASET_REPO,
    LEROBOT_TOLERANCE_S,
    OPENPI_CONFIG_NAME,
)
from grpo.context_builders import make_straight_route
from grpo.eval_metrics import eval_candidate_vs_references
from grpo.openpi_grpo_runner import (
    _context_from_sample,
    _ensure_checkpoint_layout,
    _lerobot_obs_from_sample,
)
from rewards.composite_reward import compute_reward
from rewards.flat_actions import unflatten_actions

# Re-export for modal
__all__ = [
    "eval_checkpoint",
    "compare_checkpoints",
    "pick_eval_indices",
    "summarize_eval_rows",
]


from grpo.eval_indices import pick_eval_holdout_indices, pick_eval_indices  # noqa: F401

def _fixed_noise(index: int, seed: int) -> np.ndarray:
    """Shape (1, 128) for openpi action_horizon=1, action_dim=128."""
    rng = np.random.default_rng(seed + index)
    return rng.standard_normal((1, FLAT_ACTION_DIM), dtype=np.float32)


def load_dataset(
    openpi_dir: str,
    repo_id: str = HF_BC_DATASET_REPO,
    dataset_root: str | Path | None = None,
):
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(openpi_dir or os.environ.get("OPENPI_DIR", "/opt/openpi"))
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    import openpi.training.config as _config

    # The dataset is fully cached on the volume. lerobot's HF "refs" version check
    # (get_safe_version -> list_repo_refs) hits the HF API and fails when we're throttled or
    # offline (429 / connection-reset). Neutralize it so eval uses the local cache only.
    try:
        import lerobot.common.datasets.utils as _lu

        _skip_version = lambda repo_id, version, *a, **k: version  # noqa: E731
        _lu.get_safe_version = _skip_version
        if hasattr(lerobot_dataset, "get_safe_version"):
            lerobot_dataset.get_safe_version = _skip_version
    except Exception:
        pass

    train_config = _config.get_config(OPENPI_CONFIG_NAME)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    root_kw = {"root": str(dataset_root)} if dataset_root else {}
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, **root_kw)
    delta_timestamps = {
        key: [t / meta.fps for t in range(train_config.model.action_horizon)]
        for key in data_config.action_sequence_keys
    }
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        delta_timestamps=delta_timestamps,
        tolerance_s=LEROBOT_TOLERANCE_S,
        **root_kw,
    )
    return train_config, dataset, meta


def load_policy(openpi_dir: str, ckpt_dir: str, train_config):
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(openpi_dir or os.environ.get("OPENPI_DIR", "/opt/openpi"))
    import openpi.policies.policy_config as policy_config

    _ensure_checkpoint_layout(ckpt_dir, openpi_dir)
    return policy_config.create_trained_policy(train_config, ckpt_dir)


def eval_one_sample(
    policy,
    dataset,
    index: int,
    *,
    noise_seed: int = 42,
    compute_composite: bool = True,
) -> dict:
    sample = dataset[index]
    obs = _lerobot_obs_from_sample(sample)
    noise = _fixed_noise(index, noise_seed)
    out = policy.infer(obs, noise=noise)
    pred_flat = np.asarray(out["actions"], dtype=np.float32).reshape(-1)
    gt_flat = np.asarray(sample.get("action", sample.get("actions")), dtype=np.float32).reshape(-1)

    state = np.asarray(sample.get("observation.state", [8.0, 0.0]), dtype=np.float32)
    metrics = eval_candidate_vs_references(
        pred_flat,
        gt_flat=gt_flat,
        initial_speed=float(state[0]) if state.size else 8.0,
    )

    row = {
        "index": int(index),
        "prompt": str(sample.get("prompt", sample.get("task", "")))[:120],
        "gt_action_mse": metrics.get("gt_action_mse", float("nan")),
        "gt_ade_m": metrics.get("gt_ade_m", float("nan")),
        "gt_fde_m": metrics.get("gt_fde_m", float("nan")),
    }

    if compute_composite:
        context = _context_from_sample(sample, gt_flat)
        breakdown = compute_reward(unflatten_actions(pred_flat), context)
        row["composite_reward"] = breakdown.total
        row["r_driving"] = breakdown.r_driving
        row["r_consistency"] = breakdown.r_consistency

    return row


def eval_checkpoint(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    label: str,
    indices: list[int],
    repo_id: str = HF_BC_DATASET_REPO,
    dataset_root: str | Path | None = None,
    noise_seed: int = 42,
) -> dict:
    train_config, dataset, _meta = load_dataset(openpi_dir, repo_id, dataset_root=dataset_root)
    policy = load_policy(openpi_dir, ckpt_dir, train_config)

    rows = []
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"[{label}] {i + 1}/{len(indices)} index={idx}", flush=True)
        rows.append(eval_one_sample(policy, dataset, idx, noise_seed=noise_seed))

    summary = summarize_eval_rows(rows)
    return {"label": label, "ckpt_dir": ckpt_dir, "n": len(rows), "summary": summary, "rows": rows}


def summarize_eval_rows(rows: list[dict]) -> dict:
    def _agg(key: str) -> dict:
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
        if not vals:
            return {"mean": None, "median": None, "std": None}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(np.mean(a)),
            "median": float(np.median(a)),
            "std": float(np.std(a)),
        }

    return {
        "gt_ade_m": _agg("gt_ade_m"),
        "gt_fde_m": _agg("gt_fde_m"),
        "gt_action_mse": _agg("gt_action_mse"),
        "composite_reward": _agg("composite_reward"),
    }


def find_latest_grpo_checkpoint(
    cache_dir: str,
    exp_name: str = "grpo-posttrain",
) -> str | None:
    """Latest orbax step under /cache/checkpoints/pi05_driving/{exp_name}/."""
    base = Path(cache_dir) / "checkpoints" / OPENPI_CONFIG_NAME / exp_name
    if not base.is_dir():
        return None
    steps = sorted(
        [int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()],
        reverse=True,
    )
    if not steps:
        return None
    # Return the step dir itself: create_trained_policy appends /params and /assets.
    return str(base / str(steps[0]))


def compare_checkpoints(
    *,
    openpi_dir: str,
    bc_ckpt_dir: str,
    grpo_ckpt_dir: str | None,
    num_samples: int = 50,
    seed: int = 42,
    repo_id: str = HF_BC_DATASET_REPO,
    dataset_root: str | Path | None = None,
    output_path: str | Path | None = None,
    save_per_sample_rows: bool = True,
    use_holdout_split: bool = True,
) -> dict:
    """Run open-loop eval on both checkpoints; same indices + same inference noise."""
    _train_config, dataset, _ = load_dataset(openpi_dir, repo_id, dataset_root=dataset_root)
    if use_holdout_split:
        indices = pick_eval_holdout_indices(len(dataset), num_samples, seed=seed)
    else:
        indices = pick_eval_indices(len(dataset), num_samples, seed=seed)
    print(f"Evaluating {len(indices)} samples from {repo_id} (seed={seed})")

    bc = eval_checkpoint(
        openpi_dir=openpi_dir,
        ckpt_dir=bc_ckpt_dir,
        label="bc_pretrained",
        indices=indices,
        repo_id=repo_id,
        dataset_root=dataset_root,
        noise_seed=seed,
    )
    if grpo_ckpt_dir is None:
        report = {
            "bc": bc,
            "grpo": None,
            "comparison": None,
            "verdict_hint": "grpo_checkpoint_missing",
            "note": "Only BC baseline ran. Train GRPO then re-run with grpo_ckpt_dir.",
        }
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                json.dump(
                    {
                        "bc_summary": bc["summary"],
                        "verdict_hint": report["verdict_hint"],
                        "note": report["note"],
                    },
                    f,
                    indent=2,
                )
        return report

    grpo = eval_checkpoint(
        openpi_dir=openpi_dir,
        ckpt_dir=grpo_ckpt_dir,
        label="grpo_posttrained",
        indices=indices,
        repo_id=repo_id,
        dataset_root=dataset_root,
        noise_seed=seed,
    )

    # Per-sample paired comparison
    wins_ade = 0
    wins_mse = 0
    wins_reward = 0
    paired = []
    for b, g in zip(bc["rows"], grpo["rows"]):
        assert b["index"] == g["index"]
        d_ade = g["gt_ade_m"] - b["gt_ade_m"]
        d_mse = g["gt_action_mse"] - b["gt_action_mse"]
        d_rw = (g.get("composite_reward", 0) or 0) - (b.get("composite_reward", 0) or 0)
        if np.isfinite(d_ade) and d_ade < 0:
            wins_ade += 1
        if np.isfinite(d_mse) and d_mse < 0:
            wins_mse += 1
        if d_rw > 0:
            wins_reward += 1
        paired.append({
            "index": b["index"],
            "delta_ade_m": d_ade,
            "delta_action_mse": d_mse,
            "delta_composite_reward": d_rw,
        })

    n = len(paired)
    comparison = {
        "num_samples": n,
        "indices": indices,
        "bc_pretrained": bc["summary"],
        "grpo_posttrained": grpo["summary"],
        "grpo_wins": {
            "ade_lower_is_better": wins_ade,
            "action_mse_lower_is_better": wins_mse,
            "composite_reward_higher_is_better": wins_reward,
            "ade_win_rate": wins_ade / n if n else 0.0,
            "mse_win_rate": wins_mse / n if n else 0.0,
            "reward_win_rate": wins_reward / n if n else 0.0,
        },
        "mean_delta": {
            "ade_m": float(np.nanmean([p["delta_ade_m"] for p in paired])),
            "action_mse": float(np.nanmean([p["delta_action_mse"] for p in paired])),
            "composite_reward": float(np.nanmean([p["delta_composite_reward"] for p in paired])),
        },
        "paired_samples": paired,
    }

    report = {
        "bc": bc,
        "grpo": grpo,
        "comparison": comparison,
        "verdict_hint": _verdict(comparison),
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        to_save = report
        if not save_per_sample_rows:
            to_save = {
                "comparison": comparison,
                "verdict_hint": report["verdict_hint"],
                "bc_summary": bc["summary"],
                "grpo_summary": grpo["summary"],
            }
        with path.open("w") as f:
            json.dump(to_save, f, indent=2)
        print(f"Wrote {path}")

    return report


def _verdict(comp: dict) -> str:
    md = comp["mean_delta"]
    wr = comp["grpo_wins"]
    if md["ade_m"] < -0.05 and wr["ade_win_rate"] > 0.55:
        return "grpo_better_ade"
    if md["ade_m"] > 0.05 and wr["ade_win_rate"] < 0.45:
        return "bc_better_ade"
    if md.get("composite_reward", 0) > 0.02 and wr["reward_win_rate"] > 0.55:
        return "grpo_better_reward"
    return "inconclusive_or_mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BC vs GRPO checkpoints on eval set")
    parser.add_argument("--bc-ckpt", required=True)
    parser.add_argument("--grpo-ckpt", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repo-id", default=HF_BC_DATASET_REPO)
    parser.add_argument("--output", type=Path, default=Path("eval_compare_report.json"))
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    report = compare_checkpoints(
        openpi_dir=openpi_dir,
        bc_ckpt_dir=args.bc_ckpt,
        grpo_ckpt_dir=args.grpo_ckpt,
        num_samples=args.num_samples,
        seed=args.seed,
        repo_id=args.repo_id,
        output_path=args.output,
    )
    print(json.dumps(report["comparison"], indent=2))
    print("verdict:", report["verdict_hint"])


if __name__ == "__main__":
    main()
