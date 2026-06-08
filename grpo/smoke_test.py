#!/usr/bin/env python3
"""Smoke test: composite reward ranking + Flow-GRPO advantages (no policy loss).

Usage:
  python -m grpo.smoke_test
  python -m grpo.smoke_test --horizon 64 --group-size 12
  python -m grpo.smoke_test --legacy   # old AlpamayoRanker path
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from grpo.context_builders import make_synthetic_smoke_context
from grpo.flow_grpo import FlowGRPOConfig, FlowGRPOTrainer
from grpo.objective import FlowGRPOInputs, compute_flow_grpo_loss


def _synthetic_reference(horizon: int = 64) -> np.ndarray:
    t = np.arange(horizon, dtype=np.float32)
    accel = 0.15 * np.ones_like(t)
    curvature = 0.003 * np.sin(t / 12.0)
    return np.stack([accel, curvature], axis=1).astype(np.float32)


def _bad_trajectory(reference: np.ndarray) -> np.ndarray:
    """Off-route / harsh candidate for ranking contrast."""
    bad = reference.copy()
    bad[:, 0] -= 1.5
    bad[:, 1] += 0.04
    return bad


def run_composite_smoke(
    *,
    group_size: int = 12,
    horizon: int = 64,
    coc_text: str = "Proceed straight and maintain lane while following traffic.",
) -> dict:
    ref = _synthetic_reference(horizon)
    context = make_synthetic_smoke_context(
        ref,
        coc_tags=["continue_straight"],
        coc_text=coc_text,
    )

    trainer = FlowGRPOTrainer(FlowGRPOConfig(group_size=group_size))
    result = trainer.dry_run_synthetic(ref, context)

    # Inject one clearly bad candidate so reward spread is guaranteed
    candidates = list(result.candidates)
    candidates[1] = _bad_trajectory(ref)
    result = trainer.run_composite_ranking_step(candidates, context)

    best_i = int(result.summary["best_index"])
    worst_i = int(result.summary["worst_index"])
    best_b = result.breakdowns[best_i]
    worst_b = result.breakdowns[worst_i]

    # Synthetic flow MSE: worse trajectories → higher loss (aligned with ranking)
    flow_losses = []
    ref = candidates[0]
    for cand in candidates:
        flow_losses.append(float(np.mean((cand - ref) ** 2)))

    obj_out = compute_flow_grpo_loss(
        FlowGRPOInputs(
            advantages=result.advantages,
            flow_losses=flow_losses,
            flow_log_prob_scale=0.05,
        )
    )
    objective_ready = True
    objective_error = None

    report = {
        "mode": "composite_flow_grpo",
        "horizon": horizon,
        "group_size": group_size,
        "summary": result.summary,
        "best": {
            "index": best_i,
            "reward": result.rewards[best_i],
            "advantage": float(result.advantages[best_i]),
            "r_driving": best_b.r_driving,
            "r_consistency": best_b.r_consistency,
            "r_ref": best_b.r_ref_clipped,
            "min_ade_ar1": best_b.min_ade_ar1,
            "meta_actions": sorted(best_b.meta_actions),
        },
        "worst": {
            "index": worst_i,
            "reward": result.rewards[worst_i],
            "advantage": float(result.advantages[worst_i]),
            "r_driving": worst_b.r_driving,
            "r_consistency": worst_b.r_consistency,
            "r_ref": worst_b.r_ref_clipped,
            "min_ade_ar1": worst_b.min_ade_ar1,
            "meta_actions": sorted(worst_b.meta_actions),
        },
        "objective_implemented": objective_ready,
        "objective": obj_out.metrics,
        "flow_grpo_loss": obj_out.loss,
        "ok": bool(
            result.summary["reward_std"] > 1e-6
            and result.rewards[best_i] > result.rewards[worst_i]
            and np.isfinite(obj_out.loss)
        ),
    }
    return report


def run_legacy_smoke(*, group_size: int = 12, horizon: int = 16) -> dict:
    from grpo.trainer import GRPOTrainer, GRPOTrainerConfig
    from rewards.action_space import actions_to_xyz
    from rewards.alpamayo_ranker import RankerConfig

    t = np.arange(horizon, dtype=np.float32)
    ref = np.stack([0.1 * np.ones_like(t), 0.002 * np.sin(t / 5.0)], axis=1).astype(np.float32)
    trainer = GRPOTrainer(
        GRPOTrainerConfig(
            group_size=group_size,
            ranker=RankerConfig(action_format="accel_curvature"),  # type: ignore[arg-type]
        )
    )
    gt_xyz = actions_to_xyz(ref, action_format="accel_curvature")
    result = trainer.dry_run_synthetic(ref, gt_xyz=gt_xyz, expert_xyz=gt_xyz)
    return {
        "mode": "legacy_alpamayo_ranker",
        "summary": result.summary,
        "ok": result.summary["reward_std"] > 1e-6,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Flow-GRPO composite reward smoke test")
    parser.add_argument("--legacy", action="store_true", help="Run old AlpamayoRanker smoke")
    parser.add_argument("--group-size", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=64)
    args = parser.parse_args()

    if args.legacy:
        report = run_legacy_smoke(group_size=args.group_size, horizon=min(args.horizon, 32))
    else:
        report = run_composite_smoke(group_size=args.group_size, horizon=args.horizon)

    print("Flow-GRPO smoke test")
    print("=" * 40)
    print(json.dumps({k: v for k, v in report.items() if k != "best" and k != "worst"}, indent=2))
    if "best" in report:
        print("best:", json.dumps(report["best"], indent=2))
        print("worst:", json.dumps(report["worst"], indent=2))
    print(f"PASS: {report['ok']}")

    if not report["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
