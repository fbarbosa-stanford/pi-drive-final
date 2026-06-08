# BC Cold-Start v2 Run Report

**Run name:** `bc-coldstart-v2`
**Date:** 2026-06-01
**W&B:** [markmusic/openpi/runs/ufmjtp6j](https://wandb.ai/markmusic/openpi/runs/ufmjtp6j)
**Checkpoint:** `markmusic/pi05-driving-bc-v2-checkpoint` (HuggingFace, private)

## Model

| Parameter | Value |
|---|---|
| Base model | pi0.5 (3.35B params) |
| VLM backbone | PaliGemma (SigLIP + Gemma-2B) |
| Action expert | Gemma-300M |
| VLM fine-tune | LoRA rank=32, alpha=64 on Gemma-2B |
| Action expert fine-tune | Full (all parameters) |
| Action dim | 128 (flat 64x2: acceleration + curvature at 10Hz, 6.4s horizon) |
| Action horizon | 1 (single chunk) |
| Image inputs | 1 front camera (224x224 via SigLIP), 2 dummy wrist cameras masked out |
| State inputs | [speed, heading_rate], padded to 128 |
| Text prompt | Fixed: "drive" (no navigation conditioning) |

## Data

| Split | Samples | Source |
|---|---|---|
| Train | 176,173 | PhysicalAI-AV ground truth egomotion |
| Eval (held-out) | 31,058 | PhysicalAI-AV ground truth egomotion |
| **Total** | **207,231** | **625 batch parquets** |

**Filters applied:** Daytime only, right-hand-traffic countries (US, Germany, France, etc.), front-wide camera required, has egomotion.

**Action labels:** Ground truth egomotion converted to (acceleration, curvature) via `traj_to_action()` from alpamayo's `UnicycleAccelCurvatureActionSpace`. No AR1 pseudo-labels.

**Navigation conditioning:** Disabled for this run. All samples receive the prompt "drive" regardless of trajectory geometry. Navigation labels exist in the dataset but are not used. Mode collapse will be addressed in Stage 2 (RL).

## Training Hyperparameters

| Parameter | Value |
|---|---|
| GPU | 8x H100 80GB (Modal) |
| Steps | 15,000 |
| Batch size | 96 (12/GPU) |
| Optimizer | AdamW (beta1=0.9, beta2=0.999) |
| Gradient clipping | 1.0 (max norm) |
| Peak LR | 3e-5 |
| Final LR | 3e-6 |
| LR schedule | Cosine decay |
| Warmup | 750 steps (5%) |
| EMA decay | 0.99 (openpi default) |
| Precision | bfloat16 |
| FSDP devices | 1 (pure data parallelism) |
| Checkpoint interval | 500 steps |
| Eval interval | 500 steps (on held-out eval set) |
| Loss function | Flow-matching MSE |

## Results

### Final Metrics

| Metric | Value |
|---|---|
| Final train loss | 0.080 |
| Final eval loss | 0.100 |
| Train/eval gap | 0.020 |
| Total wall time | ~3.1 hours |
| Training rate | ~1.3 steps/sec |

### Loss Trajectory

| Step | Train Loss | Eval Loss | LR |
|---|---|---|---|
| 0 | 1.209 | - | 4.0e-8 |
| 500 | 0.503 | 0.304 | 2.0e-5 |
| 1000 | 0.267 | 0.201 | 2.9e-5 |
| 2000 | 0.173 | 0.151 | 2.8e-5 |
| 3000 | 0.123 | 0.132 | 2.6e-5 |
| 5000 | 0.105 | 0.118 | 2.1e-5 |
| 7500 | 0.093 | 0.109 | 1.6e-5 |
| 10000 | 0.087 | 0.104 | 1.1e-5 |
| 12000 | 0.086 | 0.099 | 7.8e-6 |
| 14999 | 0.080 | 0.100 | 3.0e-6 |

### Analysis

**Convergence:** Loss dropped sharply in the first 2000 steps (1.21 -> 0.17), then continued a steady decline. The model learned the bulk of the driving behavior early and refined it over the remaining steps.

**Overfitting:** Minimal. The train/eval gap stayed below 0.02 throughout training. Eval loss plateaued around step 12000 at ~0.099 and stayed flat through the end, while train loss continued to decrease slightly. This suggests the model has extracted most generalizable signal from the data and further training yields diminishing returns.

**Gradient norm:** Started at ~0.4, peaked at ~1.9 around step 350 (during the steepest learning phase), then settled to ~0.4 by the end. No gradient explosions observed.

**Parameter norm:** Stable throughout at ~1804.0, increasing by only ~1.0 over the full run (1803.94 -> 1804.94). The LoRA + full action expert fine-tune did not cause parameter drift.

**Learning rate:** Cosine schedule worked as intended. Warmup from near-zero to 3e-5 peak at step 750, then smooth decay to 3e-6 at the end.

## Artifacts

- `loss.csv` - Training loss (every 50 steps)
- `eval_loss.csv` - Held-out eval loss (every 500 steps)
- `grad_norm.csv` - Gradient norm (every 50 steps)
- `learning_rate.csv` - Learning rate schedule (every 50 steps)
- `param_norm.csv` - Parameter norm (every 50 steps)

## Key Decisions

1. **No navigation labels** - All prompts set to "drive". The dataset contains derived nav labels (turn left/right, bear left/right, drive forward) but they are unused. Rationale: mode collapse from lack of navigation conditioning will be addressed in Stage 2 via RL, not BC.

2. **Full action expert fine-tune** - The 300M Gemma action expert is fully fine-tuned (not LoRA). Robot arm manipulation actions don't transfer to driving, so the action expert needs to learn from scratch.

3. **LoRA on VLM** - The 2B Gemma VLM retains its vision-language knowledge about roads, traffic, and driving scenes via LoRA (rank=32, alpha=64). Only attention and FFN layers are adapted.

4. **Flat action dim 128** - Actions are 64 timesteps x 2 (acceleration, curvature) flattened to 128, with action_horizon=1. This represents a 6.4-second trajectory at 10Hz using the unicycle kinematic model.

## Next Steps

- Open-loop evaluation: run inference on held-out clips, convert predicted (accel, curvature) back to XYZ via `action_to_traj()`, compute minADE against ground truth
- Stage 2: L4 counterfactual critique RL using AR1-10B to refine the BC policy
- Scale up data (more PhysicalAI-AV clips) if BC performance is insufficient
- Add navigation conditioning if mode collapse is observed during evaluation
