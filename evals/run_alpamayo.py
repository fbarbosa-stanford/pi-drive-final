"""Run Alpamayo-R1 (Alpamayo 1) inference on N random clips from the
NVIDIA PhysicalAI-Autonomous-Vehicles dataset, on a Modal GPU.

Usage:
    modal run run_alpamayo.py                 # 10 random clips, L40S
    modal run run_alpamayo.py --n 10 --seed 0

Auth:
  - HF token is read locally from ~/.cache/huggingface/token at launch and
    injected as a Modal secret (HF_TOKEN). Account `alexjk1m` must have gated
    access to both nvidia/Alpamayo-R1-10B and nvidia/PhysicalAI-Autonomous-Vehicles.
  - Runs on whichever Modal profile is active (currently georg / `golfmo`).
"""

import os
import modal

# ---- image: no flash-attn; use sdpa attention (documented fallback) ----
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "torchvision")
    .pip_install(
        "transformers==4.57.1",
        "accelerate",
        "einops>=0.8.1",
        "hydra-core>=1.3.2",
        "pillow>=12.0.0",
        "scipy",
        "colorlog>=6.0.0",
        "av>=14.4.0",
        "huggingface-hub>=0.36.0",
        "hf_transfer",
        "pandas",
        "pyarrow",
        "tqdm",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/NVlabs/physical_ai_av.git /opt/physical_ai_av",
        "git clone --depth 1 https://github.com/NVlabs/alpamayo.git /opt/alpamayo",
        "pip install --no-deps -e /opt/physical_ai_av",
        "pip install --no-deps -e /opt/alpamayo",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("alpamayo-r1-inference")

# Persist HF cache (22GB weights + streamed clips) across runs.
cache_vol = modal.Volume.from_name("alpamayo-hf-cache", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-alexjk1m")


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60,
    secrets=[hf_secret],
    volumes={"/cache": cache_vol},
)
def run(n: int = 10, seed: int = 0, t0_us: int = 5_100_000, num_traj_samples: int = 1):
    import random
    import numpy as np
    import torch

    import physical_ai_av
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    print("=" * 70)
    print(f"Alpamayo-R1 inference | n={n} seed={seed} t0_us={t0_us} "
          f"num_traj_samples={num_traj_samples}")
    print("=" * 70)

    # --- dataset interface + candidate clip pool ---
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    candidates = list(avdi.clip_index.index)
    random.Random(seed).shuffle(candidates)
    print(f"Clip index loaded: {len(candidates)} clips available.")

    # --- load model once (sdpa attn -> no flash-attn needed) ---
    print("Loading model nvidia/Alpamayo-R1-10B (downloads ~22GB on first run)...")
    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained(
        "nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16
    ).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.")

    results = []
    attempts = 0
    max_attempts = min(len(candidates), n * 5 + 10)

    for clip_id in candidates:
        if len(results) >= n or attempts >= max_attempts:
            break
        attempts += 1
        try:
            data = load_physical_aiavdataset(clip_id, t0_us=t0_us, avdi=avdi)
            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                continue_final_message=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = helper.to_device(
                {
                    "tokenized_data": inputs,
                    "ego_history_xyz": data["ego_history_xyz"],
                    "ego_history_rot": data["ego_history_rot"],
                },
                "cuda",
            )

            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs,
                    top_p=0.98,
                    temperature=0.6,
                    num_traj_samples=num_traj_samples,
                    max_generation_length=256,
                    return_extra=True,
                )

            cot = extra["cot"][0]
            gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
            pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
            diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
            min_ade = float(diff.min())

            results.append({"clip_id": clip_id, "cot": cot, "min_ade": min_ade})
            print("\n" + "-" * 70)
            print(f"[{len(results)}/{n}] clip_id={clip_id}  minADE={min_ade:.3f} m")
            print("Chain-of-Causation:")
            print(cot if isinstance(cot, str) else str(cot))
        except Exception as e:
            print(f"  ! skip {clip_id}: {type(e).__name__}: {e}")
            continue

    print("\n" + "=" * 70)
    print(f"DONE: {len(results)}/{n} clips ({attempts} attempted)")
    ades = [r["min_ade"] for r in results]
    if ades:
        print(f"minADE  mean={np.mean(ades):.3f}  median={np.median(ades):.3f}  "
              f"min={np.min(ades):.3f}  max={np.max(ades):.3f}")
    print("=" * 70)
    return results


@app.local_entrypoint()
def main(n: int = 10, seed: int = 0, num_traj_samples: int = 1):
    results = run.remote(n=n, seed=seed, num_traj_samples=num_traj_samples)
    print(f"\nReturned {len(results)} results:")
    for i, r in enumerate(results, 1):
        print(f"  {i:>2}. {r['clip_id']}  minADE={r['min_ade']:.3f} m")
