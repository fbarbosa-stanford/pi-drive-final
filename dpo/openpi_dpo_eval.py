#!/usr/bin/env python3
"""Batch DPO preference eval on N holdout samples (AR1 picks best in each group)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dpo.openpi_dpo_runner import load_policy_and_dataset, run_step
from grpo.eval_indices import pick_eval_holdout_indices, pick_eval_indices


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows if key in r and r.get("status") == "ok"]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "median": None, "std": None}
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
    }


def summarize_rows(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    return {
        "n_ok": len(ok),
        "n_failed": len(rows) - len(ok),
        "chosen_ar1_ade_m": _agg(ok, "chosen_ar1_ade_m"),
        "rejected_ar1_ade_m": _agg(ok, "rejected_ar1_ade_m"),
        "margin_m": _agg(ok, "margin_m"),
        "chosen_gt_ade_m": _agg(ok, "chosen_gt_ade_m"),
        "rejected_gt_ade_m": _agg(ok, "rejected_gt_ade_m"),
        "chosen_gt_action_mse": _agg(ok, "chosen_gt_action_mse"),
        "dpo_loss": _agg(ok, "dpo_loss"),
        "mean_dpo_accuracy": float(
            np.mean([r["objective_metrics"]["accuracy"] for r in ok if "objective_metrics" in r])
        )
        if ok
        else None,
    }


def eval_fifty(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    num_samples: int = 50,
    seed: int = 42,
    group_size: int = 8,
    dataset_root: str | Path | None = None,
    labels_path: str | None = None,
    use_holdout_split: bool = True,
    output_path: str | Path | None = None,
) -> dict:
    """Run DPO preference step on ``num_samples`` dataset indices (policy loaded once)."""
    from grpo.dataset_paths import resolve_bc_dataset

    if dataset_root is None:
        dataset_root, repo_id, use_holdout_split = resolve_bc_dataset(
            os.environ.get("CACHE_DIR", "/cache")
        )
    else:
        from grpo.constants import HF_BC_DATASET_REPO

        repo_id = HF_BC_DATASET_REPO
    policy, dataset, _ = load_policy_and_dataset(
        openpi_dir, ckpt_dir, dataset_root=dataset_root,
    )

    if use_holdout_split:
        indices = pick_eval_holdout_indices(len(dataset), num_samples, seed=seed)
    else:
        indices = pick_eval_indices(len(dataset), num_samples, seed=seed)

    print(f"DPO eval on {len(indices)} samples from {repo_id} (seed={seed})", flush=True)
    rows = []
    for i, idx in enumerate(indices):
        if i % 5 == 0:
            print(f"  [{i + 1}/{len(indices)}] index={idx}", flush=True)
        row = run_step(
            openpi_dir=openpi_dir,
            ckpt_dir=ckpt_dir,
            group_size=group_size,
            dataset_index=idx,
            labels_path=labels_path,
            noise_seed=seed,
            policy=policy,
            dataset=dataset,
        )
        rows.append(row)

    summary = summarize_rows(rows)
    report = {
        "num_samples": len(indices),
        "indices": indices,
        "seed": seed,
        "group_size": group_size,
        "repo_id": repo_id,
        "use_holdout_split": use_holdout_split,
        "ckpt_dir": ckpt_dir,
        "summary": summary,
        "rows": rows,
    }

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {path}")

    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("/cache/dpo_eval/eval_50_seed42.json"))
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    report = eval_fifty(
        openpi_dir=openpi_dir,
        ckpt_dir=args.ckpt_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        group_size=args.group_size,
        output_path=args.output,
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
