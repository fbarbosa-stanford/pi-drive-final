#!/usr/bin/env python3
"""Local DPO smoke: synthetic candidates + AR1 modes → preference pair + loss."""

from __future__ import annotations

import argparse
import json

import numpy as np

from dpo.alpamayo_preference import pick_preference_pair
from dpo.context_builders import make_smoke_context
from dpo.trainer import DPOConfig, DPOTrainer
from rewards.flat_actions import flatten_actions


def _synthetic_group(reference: np.ndarray, group_size: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    group = [reference.copy()]
    for i in range(1, group_size):
        noise = rng.normal(0, 1.0, size=reference.shape).astype(np.float32)
        scale = np.array([0.15, 0.02], dtype=np.float32) * (i + 1)
        group.append(reference + noise * scale)
    return group


def run_smoke(group_size: int = 6, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    ref = rng.normal(0, 0.02, size=(64, 2)).astype(np.float32)
    ref[0, 0] = 0.1
    candidates = _synthetic_group(ref, group_size, seed)
    context = make_smoke_context(ref)

    pair = pick_preference_pair(candidates, context=context)
    assert pair is not None

    flow_losses = [
        float(np.mean((flatten_actions(c) - flatten_actions(ref)) ** 2))
        for c in candidates
    ]

    trainer = DPOTrainer(DPOConfig(group_size=group_size))
    result = trainer.run_preference_step(candidates, context, flow_losses)
    assert result is not None
    _, obj_out = result

    return {
        "chosen_idx": pair.chosen_idx,
        "rejected_idx": pair.rejected_idx,
        "margin_m": pair.margin_m,
        "dpo_loss": obj_out.loss,
        "metrics": obj_out.metrics,
        "chosen_is_ref": pair.chosen_idx == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_smoke(group_size=args.group_size, seed=args.seed)
    print(json.dumps(report, indent=2))
    if not report["chosen_is_ref"]:
        raise SystemExit("expected reference trajectory to win AR1 ranking in smoke test")


if __name__ == "__main__":
    main()
