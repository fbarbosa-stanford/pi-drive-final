#!/usr/bin/env python3
"""DPO post-training loop over N dataset clips (AR1 preferences, one policy load)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from dpo.openpi_dpo_runner import load_policy_and_dataset, run_step
from grpo.eval_indices import pick_eval_holdout_indices, pick_eval_indices


def _summarize_steps(steps: list[dict]) -> dict:
    ok = [s for s in steps if s.get("status") == "ok"]
    losses = [s["dpo_loss"] for s in ok if "dpo_loss" in s]
    margins = [s["margin_m"] for s in ok if "margin_m" in s]
    return {
        "n_clips": len(steps),
        "n_ok": len(ok),
        "n_skipped": len(steps) - len(ok),
        "mean_dpo_loss": float(np.mean(losses)) if losses else None,
        "mean_margin_m": float(np.mean(margins)) if margins else None,
        "mean_chosen_gt_ade_m": _mean_key(ok, "chosen_gt_ade_m"),
        "mean_rejected_gt_ade_m": _mean_key(ok, "rejected_gt_ade_m"),
    }


def _mean_key(rows: list[dict], key: str) -> float | None:
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.mean(vals)) if vals else None


def train_on_clips(
    *,
    openpi_dir: str,
    ckpt_dir: str,
    dataset_root: str | Path,
    repo_id: str,
    num_clips: int = 50,
    seed: int = 42,
    group_size: int = 8,
    labels_path: str | None = None,
    use_holdout_split: bool = True,
    exp_name: str = "dpo-posttrain",
    output_path: str | Path | None = None,
) -> dict:
    """Run one DPO preference step per clip index (ranking + loss; backward TBD)."""
    policy, dataset, _ = load_policy_and_dataset(
        openpi_dir, ckpt_dir, dataset_root=dataset_root, repo_id=repo_id,
    )

    if use_holdout_split:
        indices = pick_eval_holdout_indices(len(dataset), num_clips, seed=seed)
    else:
        indices = pick_eval_indices(len(dataset), num_clips, seed=seed)

    print(
        f"DPO post-train on {len(indices)} clips (group_size={group_size}, seed={seed})",
        flush=True,
    )

    steps: list[dict] = []
    for i, idx in enumerate(indices):
        if i % 5 == 0:
            print(f"  clip {i + 1}/{len(indices)} index={idx}", flush=True)
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
        row["clip"] = i
        steps.append(row)

    summary = _summarize_steps(steps)
    report = {
        "exp_name": exp_name,
        "num_clips": len(indices),
        "indices": indices,
        "seed": seed,
        "group_size": group_size,
        "use_holdout_split": use_holdout_split,
        "ckpt_dir": ckpt_dir,
        "summary": summary,
        "steps": steps,
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

    parser = argparse.ArgumentParser(description="DPO post-train on N clips")
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--num-clips", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--exp-name", default="dpo-posttrain")
    parser.add_argument(
        "--labels-path",
        default=os.environ.get("AR1_LABELS_PATH", "/cache/labels/ar1_labels.jsonl"),
    )
    args = parser.parse_args()

    openpi_dir = os.environ.get("OPENPI_DIR", "/opt/openpi")
    out = Path("/cache/dpo_reports") / f"{args.exp_name}_clips{args.num_clips}.json"
    report = train_on_clips(
        openpi_dir=openpi_dir,
        ckpt_dir=args.ckpt_dir,
        num_clips=args.num_clips,
        seed=args.seed,
        group_size=args.group_size,
        labels_path=args.labels_path,
        exp_name=args.exp_name,
        output_path=out,
    )
    print(json.dumps(report["summary"], indent=2))
    if report["summary"]["n_ok"] == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
