"""Apples-to-apples trajectory eval on 200 PhysicalAI-AV clips:
   Alpamayo-R1 (baseline)  vs  pi0.5 BC  vs  pi0.5 GRPO.

Methodology (locked with user):
  * SAME 200 (clip_id, t0) for all three models.
  * SINGLE shared ground truth = true logged ego future trajectory
    (`ego_future_xyz`, 64 pts / 6.4 s, ego frame at t0).
  * ADE = mean L2 over the 64 waypoints, in meters. Alpamayo = 1 sample.
  * pi0.5 outputs normalized (accel, curvature) chunks; we decode them with
    Alpamayo's *proper inverse* `action_to_traj` (denormalize + integrate) so
    they land in the same ego frame as the GT — the fair decode. We also report
    the pi-drive-style crude integrator for continuity with the authors' numbers,
    and the "encoding floor" (decode(encode(GT)) vs GT) to prove the decode is faithful.

Stages (orchestrated by the local entrypoint):
  A build_eval_set   (CPU, alpamayo env)  -> /eval/eval_set.parquet
  B run_alpamayo     (L40S, alpamayo env) -> /eval/alpamayo_results.parquet
  C run_pi05         (H100, openpi env)   -> /eval/pi_pred_{bc,grpo}.parquet
  D decode_score_pi  (CPU, alpamayo env)  -> /eval/pi_results.parquet
  report             (local)              -> printed table

Run:
  modal run eval_compare.py                # full pipeline, n=200
  modal run eval_compare.py --n 200 --seed 0
"""

import modal

# --------------------------------------------------------------------------
# Shared volumes / secret
# --------------------------------------------------------------------------
EVAL_VOL = modal.Volume.from_name("alpamayo-pi-eval", create_if_missing=True)
HF_CACHE = modal.Volume.from_name("alpamayo-hf-cache", create_if_missing=True)   # reuses cached 22GB Alpamayo weights
PI_CACHE = modal.Volume.from_name("pi05-eval-cache", create_if_missing=True)
HF_SECRET = modal.Secret.from_name("huggingface-alexjk1m")

OPENPI_DIR = "/opt/openpi"
BC_REPO = "markmusic/pi05-driving-bc-v2-checkpoint"
GRPO_REPO = "fbarbosa1/pi05-driving-grpo-5k-checkpoint"
DPO_REPO = "fbarbosa1/pi05-driving-dpo-cosmos"   # Cosmos-3-judged DPO (db-b10), weights now on HF
T0_US = 5_000_000

# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------
alpamayo_img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch==2.8.0", "torchvision")
    .pip_install(
        "transformers==4.57.1", "accelerate", "einops>=0.8.1", "hydra-core>=1.3.2",
        "pillow>=12.0.0", "scipy", "colorlog>=6.0.0", "av>=14.4.0",
        "huggingface-hub>=0.36.0", "hf_transfer", "pandas", "pyarrow", "tqdm",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/NVlabs/physical_ai_av.git /opt/physical_ai_av",
        "git clone --depth 1 https://github.com/NVlabs/alpamayo.git /opt/alpamayo",
        "pip install --no-deps -e /opt/physical_ai_av",
        "pip install --no-deps -e /opt/alpamayo",
    )
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

pi_img = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "git-lfs", "build-essential", "clang", "ffmpeg")
    .pip_install("uv")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
        f"cd {OPENPI_DIR} && uv pip install pyarrow pillow numpy",
    )
    .pip_install("huggingface_hub", "pyarrow", "pillow")
    .env({"HF_HOME": "/pcache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("alpamayo-pi-eval")


# ==========================================================================
# A. Build the common eval set (200 clips, shared GT) — CPU
# ==========================================================================
@app.function(image=alpamayo_img, timeout=60 * 60, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=32 * 1024)
def build_eval_set(n: int = 200, seed: int = 0):
    import io, numpy as np, pandas as pd, torch
    from PIL import Image
    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    from alpamayo_r1.geometry.rotation import so3_to_yaw_torch

    RHT = {"United States","Germany","France","Italy","Spain","Netherlands","China","Canada",
           "Mexico","Brazil","Poland","Sweden","Norway","Denmark","Finland","Austria",
           "Switzerland","Belgium","Czech Republic","Portugal"}

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    # attach data_collection for right-hand-traffic/daytime filtering (best effort)
    try:
        from huggingface_hub import hf_hub_download
        dc = hf_hub_download("nvidia/PhysicalAI-Autonomous-Vehicles",
                             "metadata/data_collection.parquet", repo_type="dataset")
        avdi.data_collection = pd.read_parquet(dc)
    except Exception as e:
        print(f"data_collection unavailable ({e}); relying on sensor + per-sample filters")

    # candidate pool: clips with front-wide cam + egomotion present, RHT+daytime if available
    fp = avdi.feature_presence
    cand = list(avdi.clip_index.index)
    try:
        dc = avdi.data_collection
        cc = next((c for c in dc.columns if c.lower() == "country"), None)
        tc = next((c for c in dc.columns if c.lower() in ("time_of_day","timeofday","tod")), None)
        mask = pd.Series(True, index=dc.index)
        if cc: mask &= dc[cc].isin(RHT)
        if tc: mask &= dc[tc] == "daytime"
        ids = set(dc.loc[mask, "clip_id"]) if "clip_id" in dc.columns else set(dc.index[mask])
        cand = [c for c in cand if c in ids]
        print(f"After RHT/daytime filter: {len(cand)} clips")
    except Exception as e:
        print(f"country/daytime filter skipped ({e})")

    rng = np.random.default_rng(seed)
    rng.shuffle(cand)

    aspace = UnicycleAccelCurvatureActionSpace()
    dt = 0.1
    rows, attempts, floor_ades = [], 0, []
    for clip_id in cand:
        if len(rows) >= n or attempts >= n * 6:
            break
        attempts += 1
        try:
            d = load_physical_aiavdataset(
                clip_id, t0_us=T0_US, avdi=avdi, maybe_stream=True,
                camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV], num_frames=1,
            )
            hist = d["ego_history_xyz"][0, 0].numpy()
            speed = float(np.linalg.norm(np.diff(hist, axis=0), axis=1)[-1] / dt)
            if speed < 1.0:
                continue
            fut = d["ego_future_xyz"][0, 0].numpy()
            if float(np.sum(np.linalg.norm(np.diff(fut, axis=0), axis=1))) < 10.0:
                continue

            yaws = so3_to_yaw_torch(d["ego_history_rot"][0, 0]).numpy()
            heading_rate = float((yaws[-1] - yaws[-2]) / dt) if len(yaws) > 1 else 0.0

            gt_action = aspace.traj_to_action(
                traj_history_xyz=d["ego_history_xyz"], traj_history_rot=d["ego_history_rot"],
                traj_future_xyz=d["ego_future_xyz"], traj_future_rot=d["ego_future_rot"],
            )[0, 0].numpy()  # (64,2) normalized

            # encoding floor: decode(encode(GT)) vs GT
            recon, _ = aspace.action_to_traj(
                torch.from_numpy(gt_action).reshape(1, 1, 64, 2).float(),
                d["ego_history_xyz"], d["ego_history_rot"])
            recon_xy = recon[0, 0, :, :2].numpy()
            floor_ades.append(float(np.mean(np.linalg.norm(recon_xy - fut[:, :2], axis=1))))

            img = d["image_frames"][0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(img).resize((640, 480), Image.LANCZOS).save(buf, "JPEG", quality=90)

            rows.append({
                "clip_id": clip_id, "t0_us": T0_US,
                "image_bytes": buf.getvalue(),
                "speed": speed, "heading_rate": heading_rate,
                "gt_action": gt_action.flatten().tolist(),            # 128
                "ego_future_xy": fut[:, :2].flatten().tolist(),       # 128  (shared GT)
                "ego_history_xyz": hist.flatten().tolist(),           # 48
                "ego_history_rot": d["ego_history_rot"][0, 0].numpy().flatten().tolist(),  # 144
            })
            if len(rows) % 25 == 0:
                print(f"  collected {len(rows)}/{n} ({attempts} attempts)")
        except Exception as e:
            continue

    df = pd.DataFrame(rows)
    df.to_parquet("/eval/eval_set.parquet", index=False)
    EVAL_VOL.commit()
    print(f"\nBuilt eval set: {len(df)} clips ({attempts} attempts).")
    if floor_ades:
        print(f"Unicycle encoding floor ADE: mean={np.mean(floor_ades):.4f}m  "
              f"max={np.max(floor_ades):.4f}m  (≈0 means the pi0.5 decode is faithful)")
    return {"n": len(df), "floor_mean": float(np.mean(floor_ades)) if floor_ades else None}


# ==========================================================================
# B. Alpamayo-R1 inference — L40S
# ==========================================================================
@app.function(image=alpamayo_img, gpu="L40S", timeout=60 * 60, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=32 * 1024)
def run_alpamayo(front_only: bool = False, num_frames: int = 4):
    import numpy as np, pandas as pd, torch
    import physical_ai_av
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    # Ablation: restrict Alpamayo to the SAME single front-wide camera pi0.5 uses.
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface() if front_only else None
    cam_feats = [avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV] if front_only else None
    tag = f"front-only(nf={num_frames})" if front_only else "full(4cam)"

    df = pd.read_parquet("/eval/eval_set.parquet")
    print(f"Alpamayo [{tag}]: scoring {len(df)} clips")

    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    print("Model loaded.")

    out = []
    for i, row in df.iterrows():
        clip_id, t0 = row["clip_id"], int(row["t0_us"])
        gt_xy = np.array(row["ego_future_xy"], dtype=np.float32).reshape(64, 2)
        try:
            if front_only:
                data = load_physical_aiavdataset(clip_id, t0_us=t0, avdi=avdi,
                                                 camera_features=cam_feats, num_frames=num_frames)
            else:
                data = load_physical_aiavdataset(clip_id, t0_us=t0)
            messages = helper.create_message(data["image_frames"].flatten(0, 1))
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                continue_final_message=True, return_dict=True, return_tensors="pt")
            mi = helper.to_device({"tokenized_data": inputs,
                                   "ego_history_xyz": data["ego_history_xyz"],
                                   "ego_history_rot": data["ego_history_rot"]}, "cuda")
            torch.cuda.manual_seed_all(42)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                    max_generation_length=256, return_extra=True)
            pred_xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]  # (64,2)
            ade = float(np.mean(np.linalg.norm(pred_xy - gt_xy, axis=1)))
            fde = float(np.linalg.norm(pred_xy[-1] - gt_xy[-1]))
            out.append({"clip_id": clip_id, "t0_us": t0, "ade": ade, "fde": fde,
                        "pred_xy": pred_xy.flatten().tolist()})
            if len(out) % 25 == 0:
                print(f"  {len(out)}/{len(df)}  running mean ADE={np.mean([o['ade'] for o in out]):.3f}")
        except Exception as e:
            print(f"  ! alpamayo skip {clip_id}: {type(e).__name__}: {e}")

    res = pd.DataFrame(out)
    out_path = "/eval/alpamayo_results_frontonly.parquet" if front_only else "/eval/alpamayo_results.parquet"
    res.to_parquet(out_path, index=False)
    EVAL_VOL.commit()
    print(f"Alpamayo [{tag}] done: {len(res)} scored, mean ADE={res['ade'].mean():.3f}m "
          f"median={res['ade'].median():.3f}m FDE={res['fde'].mean():.3f}m")
    return {"n": len(res), "mean_ade": float(res["ade"].mean()),
            "median_ade": float(res["ade"].median()), "mean_fde": float(res["fde"].mean())}


# ==========================================================================
# C. pi0.5 inference (BC + GRPO) — H100, runs in openpi venv subprocess
# ==========================================================================
_PI_INFER = r'''
import json, sys, io, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

OPENPI_DIR = "/opt/openpi"

def _patch_openpi():
    dst = f"{OPENPI_DIR}/src/openpi/policies/driving_policy.py"
    with open(dst, "w") as f:
        f.write("""
import dataclasses, einops, numpy as np
from openpi import transforms
from openpi.models import model as _model
def _parse_image(image):
    if isinstance(image, dict) and 'bytes' in image:
        import io; from PIL import Image as _I
        image = np.array(_I.open(io.BytesIO(image['bytes'])))
    else:
        image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255*image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
@dataclasses.dataclass(frozen=True)
class DrivingInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05
    def __call__(self, data):
        base = _parse_image(data["observation/image"])
        out = {"state": np.asarray(data["observation/state"], dtype=np.float32),
               "image": {"base_0_rgb": base, "left_wrist_0_rgb": np.zeros_like(base),
                         "right_wrist_0_rgb": np.zeros_like(base)},
               "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_,
                              "right_wrist_0_rgb": np.False_}}
        if "actions" in data: out["actions"] = data["actions"]
        out["prompt"] = "drive"
        return out
@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        a = np.asarray(data["actions"], dtype=np.float32)
        if a.ndim == 1: a = a[np.newaxis, :]
        return {"actions": a}
""")
    gp = f"{OPENPI_DIR}/src/openpi/models/gemma.py"
    c = open(gp).read()
    if "gemma_2b_lora_driving" not in c:
        c = c.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]')
        c = c.replace('    if variant == "gemma_300m_lora":',
            '    if variant == "gemma_2b_lora_driving":\n'
            '        return Config(\n'
            '            width=2048, depth=18, mlp_dim=16_384,\n'
            '            num_heads=8, num_kv_heads=1, head_dim=256,\n'
            '            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},\n'
            '        )\n'
            '    if variant == "gemma_300m_lora":')
        open(gp, "w").write(c)
    cp = f"{OPENPI_DIR}/src/openpi/training/config.py"
    c = open(cp).read()
    if "pi05_driving" not in c:
        c = c.replace("import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy")
        ddc = (
            "\n@dataclasses.dataclass(frozen=True)\n"
            "class LeRobotDrivingDataConfig(DataConfigFactory):\n"
            "    @override\n"
            "    def create(self, assets_dirs, model_config):\n"
            "        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform({\n"
            '            "observation/image": "observation.images.front",\n'
            '            "observation/state": "observation.state",\n'
            '            "actions": "action", "prompt": "prompt"})])\n'
            "        data_transforms = _transforms.Group(\n"
            "            inputs=[driving_policy.DrivingInputs(model_type=model_config.model_type)],\n"
            "            outputs=[driving_policy.DrivingOutputs()])\n"
            "        model_transforms = ModelTransformFactory()(model_config)\n"
            "        return dataclasses.replace(self.create_base_config(assets_dirs, model_config),\n"
            "            repack_transforms=repack_transform, data_transforms=data_transforms,\n"
            '            model_transforms=model_transforms, action_sequence_keys=("action",))\n\n')
        c = c.replace("@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
                      ddc + "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:")
        dtc = (
            "\n    TrainConfig(\n"
            '        name="pi05_driving",\n'
            "        model=pi0_config.Pi0Config(pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving", action_expert_variant="gemma_300m"),\n'
            "        data=LeRobotDrivingDataConfig(repo_id=\"markmusic/pi05-physical-av-bc\",\n"
            "            base_config=DataConfig(prompt_from_task=True)),\n"
            "        weight_loader=weight_loaders.CheckpointWeightLoader(\"gs://openpi-assets/checkpoints/pi05_base/params\"),\n"
            "        freeze_filter=pi0_config.Pi0Config(pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving", action_expert_variant="gemma_300m").get_freeze_filter(),\n'
            "        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=750, peak_lr=3e-5, decay_steps=15_000, decay_lr=3e-6),\n"
            "        optimizer=_optimizer.AdamW(b1=0.9, b2=0.999, clip_gradient_norm=1.0),\n"
            "        num_train_steps=15_000, batch_size=96, fsdp_devices=1,\n"
            "        save_interval=500, log_interval=50,\n"
            '        checkpoint_base_dir="/pcache/checkpoints"),\n')
        c = c.replace("    *polaris_config.get_polaris_configs(),\n]",
                      "    *polaris_config.get_polaris_configs()," + dtc + "]")
        open(cp, "w").write(c)
    print("openpi patched")

def main():
    args = json.loads(sys.argv[1])
    ckpt_local = args["ckpt_local"]; out_path = args["out_path"]; model = args["model"]
    _patch_openpi()
    from openpi.training import config as _config
    from openpi.policies.policy_config import create_trained_policy
    import openpi.transforms as transforms
    cfg = _config.get_config("pi05_driving")
    repack = transforms.Group(inputs=[transforms.RepackTransform({
        "observation/image": "observation.images.front", "observation/state": "observation.state",
        "actions": "action", "prompt": "prompt"})])
    print(f"Loading {model} policy from {ckpt_local} ...")
    policy = create_trained_policy(cfg, ckpt_local, repack_transforms=repack, default_prompt="drive")
    print("Policy loaded.")

    t = pq.read_table("/eval/eval_set.parquet")
    n = t.num_rows
    clip_ids = t.column("clip_id").to_pylist()
    t0s = t.column("t0_us").to_pylist()
    imgs = t.column("image_bytes").to_pylist()
    speeds = t.column("speed").to_pylist()
    hrates = t.column("heading_rate").to_pylist()
    gtacts = t.column("gt_action").to_pylist()

    preds = []
    for i in range(n):
        image = np.array(Image.open(io.BytesIO(imgs[i])).convert("RGB"))
        obs = {"observation.images.front": image,
               "observation.state": np.array([speeds[i], hrates[i]], dtype=np.float32),
               "action": np.array(gtacts[i], dtype=np.float32), "prompt": "drive"}
        a = np.asarray(policy.infer(obs)["actions"], dtype=np.float32).reshape(-1)[:128]
        preds.append({"clip_id": clip_ids[i], "t0_us": t0s[i], "pred_action": a.tolist()})
        if (i+1) % 25 == 0:
            print(f"  {model}: {i+1}/{n}")
    import pandas as pd
    pd.DataFrame(preds).to_parquet(out_path, index=False)
    print(f"{model}: wrote {len(preds)} preds to {out_path}")

if __name__ == "__main__":
    main()
'''


@app.function(image=pi_img, gpu="H100", timeout=60 * 60, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/pcache": PI_CACHE}, memory=64 * 1024)
def run_pi05(model: str):
    """model in {'bc','grpo','dpo'}."""
    import os, json, shutil, subprocess
    from huggingface_hub import snapshot_download

    repo = {"bc": BC_REPO, "grpo": GRPO_REPO, "dpo": DPO_REPO}[model]
    ckpt_local = f"/pcache/eval_ckpt_{model}"
    params_dir = f"{ckpt_local}/params"
    assets_dir = f"{ckpt_local}/assets/markmusic/pi05-physical-av-bc"
    os.makedirs(assets_dir, exist_ok=True)
    norm_dst = f"{assets_dir}/norm_stats.json"

    # params: BC repo is raw orbax at root -> params/; GRPO & DPO have params/+assets/ subdirs
    if not os.path.exists(f"{params_dir}/_METADATA"):
        if model == "bc":
            snapshot_download(repo, local_dir=params_dir, repo_type="model")
        else:
            snapshot_download(repo, local_dir=ckpt_local, repo_type="model",
                              allow_patterns=["params/**", "assets/**"])
        PI_CACHE.commit()
    if not os.path.exists(norm_dst):  # BC has none -> reuse GRPO's (same base dataset)
        ns = snapshot_download(GRPO_REPO, repo_type="model",
                               allow_patterns=["assets/markmusic/pi05-physical-av-bc/norm_stats.json"])
        shutil.copy2(f"{ns}/assets/markmusic/pi05-physical-av-bc/norm_stats.json", norm_dst)
    print(f"params: {os.listdir(params_dir)[:6]} | norm_stats present: {os.path.exists(norm_dst)}")

    script = "/tmp/pi_infer.py"
    open(script, "w").write(_PI_INFER)
    out_path = f"/eval/pi_pred_{model}.parquet"
    args = json.dumps({"ckpt_local": ckpt_local, "out_path": out_path, "model": model})
    r = subprocess.run([f"{OPENPI_DIR}/.venv/bin/python", "-u", script, args], cwd=OPENPI_DIR, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pi05 {model} infer failed ({r.returncode})")
    EVAL_VOL.commit()
    return {"model": model, "out": out_path}


# ==========================================================================
# D. Decode pi0.5 predictions -> XY (proper inverse) + score vs shared GT — CPU
# ==========================================================================
@app.function(image=alpamayo_img, timeout=60 * 30, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def decode_score_pi():
    import numpy as np, pandas as pd, torch
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

    aspace = UnicycleAccelCurvatureActionSpace()
    eval_df = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")

    def crude_xy(actions, speed, dt=0.1):  # pi-drive-style integrator (for continuity)
        v, h, x, y, pts = speed, 0.0, 0.0, 0.0, []
        for a, k in actions:
            v = max(v + float(a) * dt, 0.0); h += v * float(k) * dt
            x += v * np.cos(h) * dt; y += v * np.sin(h) * dt; pts.append([x, y])
        return np.array(pts)

    import os
    rows = []
    for model in ["bc", "grpo", "dpo"]:
        if not os.path.exists(f"/eval/pi_pred_{model}.parquet"):
            continue
        pred_df = pd.read_parquet(f"/eval/pi_pred_{model}.parquet")
        ades_fair, ades_crude, fdes = [], [], []
        for _, pr in pred_df.iterrows():
            er = eval_df.loc[pr["clip_id"]]
            gt = np.array(er["ego_future_xy"], dtype=np.float32).reshape(64, 2)
            act = np.array(pr["pred_action"], dtype=np.float32).reshape(64, 2)
            hist = torch.from_numpy(np.array(er["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
            hrot = torch.from_numpy(np.array(er["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
            fut, _ = aspace.action_to_traj(torch.from_numpy(act).reshape(1, 1, 64, 2), hist, hrot)
            pred_xy = fut[0, 0, :, :2].numpy()
            ades_fair.append(float(np.mean(np.linalg.norm(pred_xy - gt, axis=1))))
            fdes.append(float(np.linalg.norm(pred_xy[-1] - gt[-1])))
            cxy = crude_xy(act, float(er["speed"]))
            ades_crude.append(float(np.mean(np.linalg.norm(cxy - gt, axis=1))))
        rows.append({"model": f"pi05_{model}", "n": len(ades_fair),
                     "mean_ade": float(np.mean(ades_fair)), "median_ade": float(np.median(ades_fair)),
                     "mean_fde": float(np.mean(fdes)), "median_fde": float(np.median(fdes)),
                     "mean_ade_crude": float(np.mean(ades_crude))})
        print(f"{model}: fair ADE={np.mean(ades_fair):.3f}  crude ADE={np.mean(ades_crude):.3f}")
    out = pd.DataFrame(rows)
    out.to_parquet("/eval/pi_results.parquet", index=False)
    EVAL_VOL.commit()
    return out.to_dict("records")


# ==========================================================================
# Orchestration
# ==========================================================================
@app.local_entrypoint()
def main(n: int = 200, seed: int = 0):
    print("=== A. Build common eval set ===")
    a = build_eval_set.remote(n=n, seed=seed)
    print(a)

    print("\n=== B/C. Alpamayo (L40S) + pi0.5 BC & GRPO (H100) in parallel ===")
    al = run_alpamayo.spawn()
    pbc = run_pi05.spawn("bc")
    pgr = run_pi05.spawn("grpo")
    al_res = al.get()
    pbc.get(); pgr.get()
    print(al_res)

    print("\n=== D. Decode + score pi0.5 vs shared GT ===")
    pi_res = decode_score_pi.remote()

    # ---- report ----
    import pandas as pd
    alp = pd.read_parquet  # not available locally; pull via function-less summary below
    print("\n" + "=" * 64)
    print(f"RESULTS — {n} PhysicalAI-AV clips, shared true-ego GT, ADE in meters")
    print("=" * 64)
    print(f"{'model':<14}{'n':>5}{'mean ADE':>11}{'median':>9}{'mean FDE':>10}")
    print("-" * 64)
    print(f"{'alpamayo_r1':<14}{a['n']:>5}{al_res['mean_ade']:>11.3f}{'':>9}{'':>10}  (baseline)")
    for r in pi_res:
        print(f"{r['model']:<14}{r['n']:>5}{r['mean_ade']:>11.3f}{r['median_ade']:>9.3f}{r['mean_fde']:>10.3f}")
    print("-" * 64)
    print(f"unicycle encoding floor (pi0.5 decode faithfulness): "
          f"{a.get('floor_mean'):.4f} m" if a.get('floor_mean') is not None else "")
    print("note: pi0.5 'fair' ADE uses Alpamayo's proper action_to_traj decode (denorm+integrate).")
    print("      'mean_ade_crude' (pi-drive-style integrator) per model:")
    for r in pi_res:
        print(f"        {r['model']}: {r['mean_ade_crude']:.3f} m")
    print("=" * 64)


# ==========================================================================
# Consolidated 4-model table (mean/median ADE+FDE) + BEV overlay for 1 clip
# ==========================================================================
viz_img = alpamayo_img.pip_install("matplotlib")


@app.function(image=viz_img, timeout=60 * 20, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def analyze_and_plot(clip_id: str = ""):
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    bc = pd.read_parquet("/eval/pi_pred_bc.parquet").set_index("clip_id")
    gr = pd.read_parquet("/eval/pi_pred_grpo.parquet").set_index("clip_id")
    alp = pd.read_parquet("/eval/alpamayo_results.parquet")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet")

    def decode(pred_action, row):
        act = torch.from_numpy(np.array(pred_action, np.float32).reshape(1, 1, 64, 2))
        hist = torch.from_numpy(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hrot = torch.from_numpy(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hist, hrot)
        return fut[0, 0, :, :2].numpy()

    recs = []
    for cid in ev.index:
        if cid not in bc.index or cid not in gr.index:
            continue
        gt = np.array(ev.loc[cid, "ego_future_xy"], np.float32).reshape(64, 2)
        b = decode(bc.loc[cid, "pred_action"], ev.loc[cid])
        g = decode(gr.loc[cid, "pred_action"], ev.loc[cid])
        recs.append({"clip_id": cid,
                     "bc_ade": float(np.mean(np.linalg.norm(b - gt, axis=1))),
                     "bc_fde": float(np.linalg.norm(b[-1] - gt[-1])),
                     "gr_ade": float(np.mean(np.linalg.norm(g - gt, axis=1))),
                     "gr_fde": float(np.linalg.norm(g[-1] - gt[-1])),
                     "lat": abs(float(gt[-1, 1]))})
    R = pd.DataFrame(recs)

    table = {
        "alpamayo_full":       [alp.ade.mean(), alp.ade.median(), alp.fde.mean(), alp.fde.median()],
        "alpamayo_front_only": [alpf.ade.mean(), alpf.ade.median(), alpf.fde.mean(), alpf.fde.median()],
        "pi05_bc":             [R.bc_ade.mean(), R.bc_ade.median(), R.bc_fde.mean(), R.bc_fde.median()],
        "pi05_grpo":           [R.gr_ade.mean(), R.gr_ade.median(), R.gr_fde.mean(), R.gr_fde.median()],
    }
    print(f"{'model':<20}{'mean_ADE':>10}{'median_ADE':>12}{'mean_FDE':>10}{'median_FDE':>12}")
    for k, v in table.items():
        print(f"{k:<20}{v[0]:>10.3f}{v[1]:>12.3f}{v[2]:>10.3f}{v[3]:>12.3f}")
    table = {k: [round(float(x), 3) for x in v] for k, v in table.items()}

    # pick a turning clip where GRPO clearly beats BC
    if not clip_id:
        c = R[(R.lat > 5) & (R.gr_ade < R.bc_ade)].copy()
        c["gain"] = c.bc_ade - c.gr_ade
        c = c.sort_values("gain", ascending=False)
        clip_id = c.iloc[0]["clip_id"]
    row = ev.loc[clip_id]
    gt = np.array(row["ego_future_xy"], np.float32).reshape(64, 2)
    b = decode(bc.loc[clip_id, "pred_action"], row)
    g = decode(gr.loc[clip_id, "pred_action"], row)
    speed = float(row["speed"])
    fwd, latd = float(gt[-1, 0]), float(gt[-1, 1])
    nav = ("stop" if fwd < 2 else "drive forward" if abs(latd) < 1 else
           "turn left" if latd > 3 else "turn right" if latd < -3 else
           "bear left" if latd > 0 else "bear right")

    # optional Alpamayo trajectories (from alpamayo_traj for this clip)
    import os, json
    alp_full = alp_front = None
    if os.path.exists("/eval/alpamayo_traj.json"):
        aj = json.load(open("/eval/alpamayo_traj.json"))
        if aj.get("clip_id") == str(clip_id):
            alp_full = np.array(aj["full_xy"], np.float32)
            alp_front = np.array(aj["front_xy"], np.float32)

    o = np.zeros((1, 2))
    gtp, bp, gp = np.vstack([o, gt]), np.vstack([o, b]), np.vstack([o, g])
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(Image.open(io.BytesIO(row["image_bytes"])).convert("RGB"))
    ax[0].axis("off")
    ax[0].set_title(f"clip {str(clip_id)[:8]} — '{nav}'  v={speed:.1f}", fontsize=15)
    ax[1].plot(gtp[:, 0], gtp[:, 1], "k-", lw=3, label="GT")
    if alp_front is not None:
        ax[1].plot(*np.vstack([o, alp_front]).T, "m--", lw=2, label="Alpamayo (1-cam)")
    ax[1].plot(bp[:, 0], bp[:, 1], "b--", lw=2.5, label="BC (pretrained)")
    ax[1].plot(gp[:, 0], gp[:, 1], "r--", lw=2.5, label="GRPO")
    ax[1].set_xlabel("x (forward)", fontsize=12)
    ax[1].set_ylabel("y (left)", fontsize=12)
    ax[1].set_title("trajectory, top-down (m)", fontsize=15)
    ax[1].legend(loc="upper right", fontsize=12); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    sel = R[R.clip_id == clip_id].iloc[0]
    return {"png": buf.getvalue(), "clip_id": str(clip_id), "nav": nav, "speed": speed,
            "table": table, "bc_ade": float(sel.bc_ade), "gr_ade": float(sel.gr_ade)}


@app.local_entrypoint()
def report_and_viz(clip_id: str = ""):
    import pathlib
    r = analyze_and_plot.remote(clip_id=clip_id)
    p = pathlib.Path("pi05_bev.png"); p.write_bytes(r["png"])
    print(f"\nSaved BEV figure -> {p}")
    print(f"clip {r['clip_id']}  '{r['nav']}'  v={r['speed']:.1f}  "
          f"BC ADE={r['bc_ade']:.2f}  GRPO ADE={r['gr_ade']:.2f}")
    print("\nmodel                mean_ADE  median_ADE  mean_FDE  median_FDE")
    for k, v in r["table"].items():
        print(f"{k:<20}{v[0]:>10}{v[1]:>12}{v[2]:>10}{v[3]:>12}")


@app.function(image=alpamayo_img, gpu="L40S", timeout=60 * 30, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=32 * 1024)
def alpamayo_traj(clip_id: str):
    """Run Alpamayo on ONE clip in both configs (4-cam full + 1-cam front-only),
    capture predicted XY trajectories, save to /eval/alpamayo_traj.json."""
    import json, numpy as np, torch
    import physical_ai_av
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper

    import pandas as pd
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    t0 = int(ev.loc[clip_id, "t0_us"])
    gt = np.array(ev.loc[clip_id, "ego_future_xy"], np.float32).reshape(64, 2)

    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def run(front_only):
        if front_only:
            data = load_physical_aiavdataset(clip_id, t0_us=t0, avdi=avdi,
                                             camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV],
                                             num_frames=1)
        else:
            data = load_physical_aiavdataset(clip_id, t0_us=t0, avdi=avdi)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                               continue_final_message=True, return_dict=True, return_tensors="pt")
        mi = helper.to_device({"tokenized_data": inputs,
                               "ego_history_xyz": data["ego_history_xyz"],
                               "ego_history_rot": data["ego_history_rot"]}, "cuda")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1,
                max_generation_length=256, return_extra=True)
        xy = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
        ade = float(np.mean(np.linalg.norm(xy - gt, axis=1)))
        return xy, ade

    full_xy, full_ade = run(False)
    front_xy, front_ade = run(True)
    print(f"clip {clip_id}: Alpamayo 4-cam ADE={full_ade:.2f}  1-cam ADE={front_ade:.2f}")
    json.dump({"clip_id": clip_id, "full_xy": full_xy.tolist(), "front_xy": front_xy.tolist(),
               "full_ade": full_ade, "front_ade": front_ade}, open("/eval/alpamayo_traj.json", "w"))
    EVAL_VOL.commit()
    return {"clip_id": clip_id, "full_ade": full_ade, "front_ade": front_ade}


@app.function(image=viz_img, gpu="L40S", timeout=60 * 30, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=32 * 1024)
def make_bev(nav_filter: str):
    """Pick the clip in `nav_filter` category where GRPO most beats BC, run Alpamayo
    1-cam on it, and render GT / Alpamayo(1-cam) / BC / GRPO overlay."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import physical_ai_av
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    bc = pd.read_parquet("/eval/pi_pred_bc.parquet").set_index("clip_id")
    gr = pd.read_parquet("/eval/pi_pred_grpo.parquet").set_index("clip_id")

    def decode(pred, row):
        act = torch.from_numpy(np.array(pred, np.float32).reshape(1, 1, 64, 2))
        hist = torch.from_numpy(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hrot = torch.from_numpy(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hist, hrot)
        return fut[0, 0, :, :2].numpy()

    def navcat(gt):
        fwd, latd = float(gt[-1, 0]), float(gt[-1, 1])
        if fwd < 2: return "stop"
        if abs(latd) < 1: return "drive forward"
        if latd > 3: return "turn left"
        if latd < -3: return "turn right"
        return "bear left" if latd > 0 else "bear right"

    best = None
    for cid in ev.index:
        if cid not in bc.index or cid not in gr.index:
            continue
        gt = np.array(ev.loc[cid, "ego_future_xy"], np.float32).reshape(64, 2)
        if navcat(gt) != nav_filter:
            continue
        b = decode(bc.loc[cid, "pred_action"], ev.loc[cid])
        g = decode(gr.loc[cid, "pred_action"], ev.loc[cid])
        bade = float(np.mean(np.linalg.norm(b - gt, axis=1)))
        gade = float(np.mean(np.linalg.norm(g - gt, axis=1)))
        if gade < bade and (best is None or (bade - gade) > best["gain"]):
            best = {"cid": cid, "gain": bade - gade, "gt": gt, "b": b, "g": g, "bade": bade, "gade": gade}
    if best is None:
        return {"error": f"no clip with GRPO<BC in category '{nav_filter}'"}
    cid = best["cid"]

    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    t0 = int(ev.loc[cid, "t0_us"])
    data = load_physical_aiavdataset(cid, t0_us=t0, avdi=avdi,
                                     camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV], num_frames=1)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                           continue_final_message=True, return_dict=True, return_tensors="pt")
    mi = helper.to_device({"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"],
                           "ego_history_rot": data["ego_history_rot"]}, "cuda")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=256, return_extra=True)
    alp = pred_xyz.cpu().numpy()[0, 0, 0, :, :2]
    alp_ade = float(np.mean(np.linalg.norm(alp - best["gt"], axis=1)))

    speed = float(ev.loc[cid, "speed"])
    o = np.zeros((1, 2))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(Image.open(io.BytesIO(ev.loc[cid, "image_bytes"])).convert("RGB")); ax[0].axis("off")
    ax[0].set_title(f"clip {str(cid)[:8]} — '{nav_filter}'  v={speed:.1f}", fontsize=15)
    ax[1].plot(*np.vstack([o, best["gt"]]).T, "k-", lw=3, label="GT")
    ax[1].plot(*np.vstack([o, alp]).T, "m--", lw=2, label="Alpamayo (1-cam)")
    ax[1].plot(*np.vstack([o, best["b"]]).T, "b--", lw=2.5, label="BC (pretrained)")
    ax[1].plot(*np.vstack([o, best["g"]]).T, "r--", lw=2.5, label="GRPO")
    ax[1].set_xlabel("x (forward)", fontsize=12); ax[1].set_ylabel("y (left)", fontsize=12)
    ax[1].set_title("trajectory, top-down (m)", fontsize=15)
    ax[1].legend(loc="best", fontsize=12); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"{nav_filter}: clip {cid} v={speed:.1f} BC_ADE={best['bade']:.2f} "
          f"GRPO_ADE={best['gade']:.2f} Alp1cam_ADE={alp_ade:.2f}")
    return {"png": buf.getvalue(), "clip_id": str(cid), "speed": speed,
            "bc_ade": best["bade"], "gr_ade": best["gade"], "alp_ade": alp_ade}


@app.local_entrypoint()
def bev_for(nav: str = "turn right", out: str = "bev.png"):
    import pathlib
    r = make_bev.remote(nav_filter=nav)
    if "error" in r:
        print(r["error"]); return
    pathlib.Path(out).write_bytes(r["png"])
    print(f"saved {out}  clip {r['clip_id']} v={r['speed']:.1f}  "
          f"BC={r['bc_ade']:.2f}  GRPO={r['gr_ade']:.2f}  Alpamayo1cam={r['alp_ade']:.2f}")


@app.function(image=viz_img, gpu="L40S", timeout=60 * 40, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=32 * 1024)
def make_combined(left_clip: str):
    """Combined 3-row BEV (turn left / turn right / drive forward). Left clip fixed;
    right + straight auto-picked to best show GRPO beating Alpamayo(1-cam) cleanly."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    import physical_ai_av
    from alpamayo_r1.config import AlpamayoR1Config
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1 import helper
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    bc = pd.read_parquet("/eval/pi_pred_bc.parquet").set_index("clip_id")
    gr = pd.read_parquet("/eval/pi_pred_grpo.parquet").set_index("clip_id")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet").set_index("clip_id")

    def decode(pred, row):
        act = torch.from_numpy(np.array(pred, np.float32).reshape(1, 1, 64, 2))
        hist = torch.from_numpy(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hrot = torch.from_numpy(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hist, hrot)
        return fut[0, 0, :, :2].numpy()

    def navcat(gt):
        fwd, latd = float(gt[-1, 0]), float(gt[-1, 1])
        if fwd < 2: return "stop"
        if abs(latd) < 1: return "drive forward"
        if latd > 3: return "turn left"
        if latd < -3: return "turn right"
        return "bear left" if latd > 0 else "bear right"

    def pick(nav):
        # GRPO clean (low ADE) AND clearly beats Alpamayo-1cam, Alpamayo wrong but not absurd.
        # For straights, prefer a long/fast clip (crisp line); for turns, prefer biggest margin.
        for gr_cap, alp_ceil in [(2.5, 10.0), (3.5, 14.0), (5.0, 20.0)]:
            cands = []
            for cid in ev.index:
                if cid not in bc.index or cid not in gr.index or cid not in alpf.index:
                    continue
                gt = np.array(ev.loc[cid, "ego_future_xy"], np.float32).reshape(64, 2)
                if navcat(gt) != nav:
                    continue
                g = decode(gr.loc[cid, "pred_action"], ev.loc[cid])
                gade = float(np.mean(np.linalg.norm(g - gt, axis=1)))
                alp1 = float(alpf.loc[cid, "ade"])
                if gade < gr_cap and gade < alp1 and alp1 < alp_ceil:
                    cands.append((cid, alp1 - gade, float(gt[-1, 0])))
            if cands:
                key = (lambda c: c[2]) if nav == "drive forward" else (lambda c: c[1])
                return max(cands, key=key)[0]
        return None

    clips = {"turn left": left_clip, "turn right": pick("turn right"), "drive forward": pick("drive forward")}
    print("selected clips:", clips)

    cfg = AlpamayoR1Config.from_pretrained("nvidia/Alpamayo-R1-10B")
    cfg.attn_implementation = "sdpa"
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", config=cfg, dtype=torch.bfloat16).to("cuda")
    model.eval()
    processor = helper.get_processor(model.tokenizer)
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    def alp_traj(cid):
        t0 = int(ev.loc[cid, "t0_us"])
        data = load_physical_aiavdataset(cid, t0_us=t0, avdi=avdi,
                                         camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV], num_frames=1)
        messages = helper.create_message(data["image_frames"].flatten(0, 1))
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                               continue_final_message=True, return_dict=True, return_tensors="pt")
        mi = helper.to_device({"tokenized_data": inputs, "ego_history_xyz": data["ego_history_xyz"],
                               "ego_history_rot": data["ego_history_rot"]}, "cuda")
        torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, _, _ = model.sample_trajectories_from_data_with_vlm_rollout(
                data=mi, top_p=0.98, temperature=0.6, num_traj_samples=1, max_generation_length=256, return_extra=True)
        return pred_xyz.cpu().numpy()[0, 0, 0, :, :2]

    o = np.zeros((1, 2))
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    for r_i, (nav, cid) in enumerate(clips.items()):
        gt = np.array(ev.loc[cid, "ego_future_xy"], np.float32).reshape(64, 2)
        b = decode(bc.loc[cid, "pred_action"], ev.loc[cid])
        g = decode(gr.loc[cid, "pred_action"], ev.loc[cid])
        alp = alp_traj(cid)
        gade = float(np.mean(np.linalg.norm(g - gt, axis=1)))
        bade = float(np.mean(np.linalg.norm(b - gt, axis=1)))
        aade = float(np.mean(np.linalg.norm(alp - gt, axis=1)))
        speed = float(ev.loc[cid, "speed"])
        print(f"{nav}: clip {cid} v={speed:.1f} GRPO={gade:.2f} Alpamayo1cam={aade:.2f} BC={bade:.2f}")

        ax0, ax1 = axes[r_i, 0], axes[r_i, 1]
        ax0.imshow(Image.open(io.BytesIO(ev.loc[cid, "image_bytes"])).convert("RGB")); ax0.axis("off")
        ax0.set_title(f"{nav}  —  clip {str(cid)[:8]}  v={speed:.1f} m/s", fontsize=14)
        ax1.plot(*np.vstack([o, gt]).T, "k-", lw=3, label="GT")
        ax1.plot(*np.vstack([o, alp]).T, "m--", lw=2, label=f"Alpamayo 1-cam  (ADE {aade:.1f})")
        ax1.plot(*np.vstack([o, b]).T, "b--", lw=2, label=f"BC  (ADE {bade:.1f})")
        ax1.plot(*np.vstack([o, g]).T, "r-", lw=2.5, label=f"GRPO  (ADE {gade:.1f})")
        ax1.set_xlabel("x (forward, m)"); ax1.set_ylabel("y (left, m)")
        ax1.set_title("trajectory, top-down", fontsize=14)
        ax1.legend(loc="best", fontsize=11); ax1.grid(alpha=0.3)
    fig.suptitle("pi0.5 GRPO vs Alpamayo (single front camera) vs BC", fontsize=17, y=0.995)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return {"png": buf.getvalue(), "clips": {k: str(v) for k, v in clips.items()}}


@app.local_entrypoint()
def combined_bev(left_clip: str = "1bc6dae4-a3a9-4afe-a882-a84d8ca9e0d1", out: str = "bev_combined.png"):
    import pathlib
    r = make_combined.remote(left_clip=left_clip)
    pathlib.Path(out).write_bytes(r["png"])
    print(f"saved {out}  clips={r['clips']}")


# ==========================================================================
# Metric 1: Longitudinal jerk (CPU; uses existing predictions + GT actions)
# ==========================================================================
@app.function(image=alpamayo_img, timeout=60 * 20, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def jerk_eval():
    import json, numpy as np, pandas as pd
    DT = 0.1
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")

    def chunk_jerk(accel):           # accel: (64,) physical m/s^2
        j = np.diff(accel) / DT      # (63,) m/s^3
        return float(np.sqrt(np.mean(j ** 2))), float(np.max(np.abs(j)))

    # GT jerk per clip (shared baseline)
    gt_acc = {cid: np.array(ev.loc[cid, "gt_action"], np.float32).reshape(64, 2)[:, 0] for cid in ev.index}
    gt_rms, gt_peak = {}, {}
    for cid, a in gt_acc.items():
        gt_rms[cid], gt_peak[cid] = chunk_jerk(a)
    a0 = next(iter(gt_acc.values()))
    print(f"sanity: GT accel sample range = [{a0.min():.2f}, {a0.max():.2f}] m/s^2 "
          f"(physical if ~±a few)")

    out = {}
    for model, fn in [("bc", "/eval/pi_pred_bc.parquet"), ("grpo", "/eval/pi_pred_grpo.parquet")]:
        pred = pd.read_parquet(fn).set_index("clip_id")
        rms_p, peak_p, rms_r, peak_r = [], [], [], []
        for cid in pred.index:
            if cid not in gt_acc:
                continue
            a = np.array(pred.loc[cid, "pred_action"], np.float32).reshape(64, 2)[:, 0]
            rp, pp = chunk_jerk(a)
            rms_p.append(rp); peak_p.append(pp)
            if gt_rms[cid] > 1e-6:
                rms_r.append(rp / gt_rms[cid])
            if gt_peak[cid] > 1e-6:
                peak_r.append(pp / gt_peak[cid])
        out[model] = {
            "n": len(rms_p),
            "rms_mean": float(np.mean(rms_p)), "rms_median": float(np.median(rms_p)),
            "peak_mean": float(np.mean(peak_p)), "peak_median": float(np.median(peak_p)),
            "rms_ratio_vs_gt_mean": float(np.mean(rms_r)), "rms_ratio_vs_gt_median": float(np.median(rms_r)),
            "peak_ratio_vs_gt_mean": float(np.mean(peak_r)), "peak_ratio_vs_gt_median": float(np.median(peak_r)),
        }
    out["gt"] = {
        "rms_mean": float(np.mean(list(gt_rms.values()))), "rms_median": float(np.median(list(gt_rms.values()))),
        "peak_mean": float(np.mean(list(gt_peak.values()))), "peak_median": float(np.median(list(gt_peak.values()))),
    }
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def run_jerk():
    import json
    print(json.dumps(jerk_eval.remote(), indent=2))


# ---- XY-derived jerk: consistent basis for ALL models incl. Alpamayo ----
@app.function(image=alpamayo_img, timeout=60 * 20, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def jerk_xy_eval():
    """Longitudinal jerk derived identically from each model's predicted XY path
    (speed->accel->jerk), so Alpamayo (XY output, no accel channel) is comparable
    to pi0.5. GT baseline derived the same way."""
    import json, numpy as np, pandas as pd, torch
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    DT = 0.1
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")

    def xy_jerk(xy):                     # xy: (64,2) ego frame, t0+0.1..t0+6.4
        p = np.vstack([np.zeros((1, 2)), xy])         # prepend ego origin at t0 -> 65 pts
        v = np.linalg.norm(np.diff(p, axis=0), axis=1) / DT   # speed (64,)
        a = np.diff(v) / DT                                   # long. accel (63,)
        j = np.diff(a) / DT                                   # jerk (62,)
        return float(np.sqrt(np.mean(j ** 2))), float(np.max(np.abs(j)))

    def decode(pred_action, row):
        act = torch.from_numpy(np.array(pred_action, np.float32).reshape(1, 1, 64, 2))
        hx = torch.from_numpy(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hr = torch.from_numpy(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hx, hr)
        return fut[0, 0, :, :2].numpy()

    # GT jerk per clip (XY-derived)
    gt_rms, gt_peak = {}, {}
    for cid in ev.index:
        r, p = xy_jerk(np.array(ev.loc[cid, "ego_future_xy"], np.float32).reshape(64, 2))
        gt_rms[cid], gt_peak[cid] = r, p

    def summarize(per_clip_rms, per_clip_peak, cids):
        rr = [per_clip_rms[c] / gt_rms[c] for c in cids if gt_rms[c] > 1e-6]
        pr = [per_clip_peak[c] / gt_peak[c] for c in cids if gt_peak[c] > 1e-6]
        return {"n": len(cids),
                "rms_mean": float(np.mean(list(per_clip_rms.values()))),
                "rms_median": float(np.median(list(per_clip_rms.values()))),
                "peak_mean": float(np.mean(list(per_clip_peak.values()))),
                "rms_ratio_vs_gt_mean": float(np.mean(rr)), "rms_ratio_vs_gt_median": float(np.median(rr)),
                "peak_ratio_vs_gt_mean": float(np.mean(pr)), "peak_ratio_vs_gt_median": float(np.median(pr))}

    out = {"gt": {"rms_mean": float(np.mean(list(gt_rms.values()))),
                  "rms_median": float(np.median(list(gt_rms.values()))),
                  "peak_mean": float(np.mean(list(gt_peak.values())))}}

    # Alpamayo (full 4-cam) and front-only (1-cam) from saved trajectories
    import os
    for label, path in [("alpamayo_full", "/eval/alpamayo_results.parquet"),
                        ("alpamayo_front_only", "/eval/alpamayo_results_frontonly.parquet")]:
        if not os.path.exists(path):
            continue
        alp = pd.read_parquet(path)
        if "pred_xy" not in alp.columns:
            continue
        rms, peak = {}, {}
        for _, r in alp.iterrows():
            rms[r["clip_id"]], peak[r["clip_id"]] = xy_jerk(np.array(r["pred_xy"], np.float32).reshape(64, 2))
        out[label] = summarize(rms, peak, list(rms.keys()))

    # pi0.5 BC / GRPO / DPO from decoded action chunks
    for model in ["bc", "grpo", "dpo"]:
        fn = f"/eval/pi_pred_{model}.parquet"
        if not os.path.exists(fn):
            continue
        pred = pd.read_parquet(fn).set_index("clip_id")
        rms, peak = {}, {}
        for cid in pred.index:
            if cid not in ev.index:
                continue
            xy = decode(pred.loc[cid, "pred_action"], ev.loc[cid])
            rms[cid], peak[cid] = xy_jerk(xy)
        out[f"pi05_{model}"] = summarize(rms, peak, list(rms.keys()))

    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def run_jerk_xy():
    import json
    print(json.dumps(jerk_xy_eval.remote(), indent=2))


@app.function(image=viz_img, timeout=60 * 15, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def bev_dpo(clip_left: str, clip_right: str):
    """Turn-left + turn-right BEV overlaying GT / Alpamayo-1cam / BC / GRPO / DPO."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet").set_index("clip_id")
    preds = {m: pd.read_parquet(f"/eval/pi_pred_{m}.parquet").set_index("clip_id") for m in ["bc", "grpo", "dpo"]}

    def decode(pa, row):
        act = torch.tensor(np.array(pa, np.float32).reshape(1, 1, 64, 2))
        hx = torch.tensor(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hr = torch.tensor(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hx, hr)
        return fut[0, 0, :, :2].numpy()

    o = np.zeros((1, 2))
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    for i, (label, cid) in enumerate([("turn left", clip_left), ("turn right", clip_right)]):
        row = ev.loc[cid]
        gt = np.array(row["ego_future_xy"], np.float32).reshape(64, 2)
        alp = np.array(alpf.loc[cid, "pred_xy"], np.float32).reshape(64, 2)
        b = decode(preds["bc"].loc[cid, "pred_action"], row)
        g = decode(preds["grpo"].loc[cid, "pred_action"], row)
        dp = decode(preds["dpo"].loc[cid, "pred_action"], row)
        ade = lambda p: float(np.mean(np.linalg.norm(p - gt, axis=1)))
        speed = float(row["speed"])
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")); ax0.axis("off")
        ax0.set_title(f"{label} — clip {str(cid)[:8]}  v={speed:.1f} m/s", fontsize=14)
        ax1.plot(*np.vstack([o, gt]).T, "k-", lw=3, label="GT")
        ax1.plot(*np.vstack([o, alp]).T, "m--", lw=2, label=f"Alpamayo 1-cam  (ADE {ade(alp):.1f})")
        ax1.plot(*np.vstack([o, b]).T, "b--", lw=2, label=f"BC  (ADE {ade(b):.1f})")
        ax1.plot(*np.vstack([o, g]).T, "r--", lw=2.5, label=f"GRPO  (ADE {ade(g):.1f})")
        ax1.plot(*np.vstack([o, dp]).T, color="darkorange", ls="--", lw=2.5, label=f"Cosmos-DPO  (ADE {ade(dp):.1f})")
        ax1.set_xlabel("x (forward, m)"); ax1.set_ylabel("y (left, m)")
        ax1.set_title("trajectory, top-down", fontsize=14); ax1.legend(loc="best", fontsize=10); ax1.grid(alpha=0.3)
    fig.suptitle("GT vs pi0.5 BC / GRPO / Cosmos-DPO  (+ Alpamayo 1-cam)", fontsize=16, y=0.995)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return {"png": buf.getvalue()}


@app.local_entrypoint()
def dpo_bev(out: str = "bev_dpo.png",
            clip_left: str = "1bc6dae4-a3a9-4afe-a882-a84d8ca9e0d1",
            clip_right: str = "3eb19f65-b338-4331-8871-5836dd8cd302"):
    import pathlib
    r = bev_dpo.remote(clip_left=clip_left, clip_right=clip_right)
    pathlib.Path(out).write_bytes(r["png"]); print(f"saved {out}")


@app.function(image=viz_img, timeout=60 * 15, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def bev_grpo_best(nav: str = "turn left", gr_cap: float = 3.0, exclude: str = "", max_other: float = 12.0):
    """Single-clip BEV in `nav` category where GRPO is strictly closest to GT and all
    models point the SAME direction (bounded spread). Prefers a clear daytime clip."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet").set_index("clip_id")
    preds = {m: pd.read_parquet(f"/eval/pi_pred_{m}.parquet").set_index("clip_id") for m in ["bc", "grpo", "dpo"]}

    def decode(pa, row):
        act = torch.tensor(np.array(pa, np.float32).reshape(1, 1, 64, 2))
        hx = torch.tensor(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hr = torch.tensor(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hx, hr)
        return fut[0, 0, :, :2].numpy()

    def navcat(gt):
        fwd, lat = float(gt[-1, 0]), float(gt[-1, 1])
        if abs(lat) < 1: return "drive forward"
        if lat > 3: return "turn left"
        if lat < -3: return "turn right"
        return "bear left" if lat > 0 else "bear right"

    excl = set(x for x in exclude.split(",") if x)
    cands = []
    for cid in ev.index:
        if cid in excl or cid not in alpf.index or any(cid not in preds[m].index for m in preds):
            continue
        row = ev.loc[cid]
        gt = np.array(row["ego_future_xy"], np.float32).reshape(64, 2)
        if navcat(gt) != nav:
            continue
        trj = {"GRPO": decode(preds["grpo"].loc[cid, "pred_action"], row),
               "BC": decode(preds["bc"].loc[cid, "pred_action"], row),
               "Cosmos-DPO": decode(preds["dpo"].loc[cid, "pred_action"], row),
               "Alpamayo 1-cam": np.array(alpf.loc[cid, "pred_xy"], np.float32).reshape(64, 2)}
        ades = {k: float(np.mean(np.linalg.norm(v - gt, axis=1))) for k, v in trj.items()}
        others = min(ades["BC"], ades["Cosmos-DPO"], ades["Alpamayo 1-cam"])
        # ALL FOUR models must turn the SAME way as GT; GRPO strictly closest; spread bounded.
        sgn = np.sign(gt[-1, 1])
        same_dir = all(np.sign(v[-1, 1]) == sgn for v in trj.values())
        if ades["GRPO"] < others and ades["GRPO"] < gr_cap and same_dir and max(ades.values()) < max_other:
            img = Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")
            bright = float(np.asarray(img).mean())   # daytime ~ brighter
            cands.append({"cid": cid, "margin": others - ades["GRPO"], "gt": gt, "trj": trj,
                          "ades": ades, "speed": float(row["speed"]), "img": row["image_bytes"],
                          "bright": bright})
    if not cands:
        return {"error": f"no {nav} clip meeting same-direction + GRPO-best criteria"}
    # prefer daytime; among those, the fastest (clearest) all-agree turn
    bright = [c for c in cands if c["bright"] > 90] or cands
    bright.sort(key=lambda c: -c["speed"])
    print("all-4-agree candidates (cid, speed, GRPO_ade):",
          [(c["cid"][:8], round(c["speed"], 1), round(c["ades"]["GRPO"], 2)) for c in bright[:8]])
    best = bright[0]

    cid = best["cid"]; o = np.zeros((1, 2))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(Image.open(io.BytesIO(best["img"])).convert("RGB")); ax[0].axis("off")
    ax[0].set_title(f"{nav} — clip {str(cid)[:8]}  v={best['speed']:.1f} m/s", fontsize=15)
    ax[1].plot(*np.vstack([o, best["gt"]]).T, "k-", lw=3, label="GT")
    ax[1].plot(*np.vstack([o, best["trj"]["Alpamayo 1-cam"]]).T, "m--", lw=2,
               label=f"Alpamayo 1-cam (ADE {best['ades']['Alpamayo 1-cam']:.1f})")
    ax[1].plot(*np.vstack([o, best["trj"]["BC"]]).T, "b--", lw=2, label=f"BC (ADE {best['ades']['BC']:.1f})")
    ax[1].plot(*np.vstack([o, best["trj"]["Cosmos-DPO"]]).T, color="darkorange", ls="--", lw=2,
               label=f"Cosmos-DPO (ADE {best['ades']['Cosmos-DPO']:.1f})")
    ax[1].plot(*np.vstack([o, best["trj"]["GRPO"]]).T, "r-", lw=3, label=f"GRPO (ADE {best['ades']['GRPO']:.1f})")
    ax[1].set_xlabel("x (forward, m)"); ax[1].set_ylabel("y (left, m)")
    ax[1].set_title("trajectory, top-down", fontsize=15); ax[1].legend(loc="best", fontsize=11); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"{nav}: clip {cid} ades={ {k: round(v,2) for k,v in best['ades'].items()} }")
    return {"png": buf.getvalue(), "clip_id": str(cid), "ades": best["ades"]}


@app.function(image=viz_img, timeout=60 * 15, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def bev_single(clip_id: str):
    """Single-clip BEV (cam + top-down) styled like the pi-drive figures:
    GT black, Alpamayo 1-cam orange dotted, BC blue dashed, GRPO red, DPO green dashed."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet").set_index("clip_id")
    preds = {m: pd.read_parquet(f"/eval/pi_pred_{m}.parquet").set_index("clip_id") for m in ["bc", "grpo", "dpo"]}
    row = ev.loc[clip_id]
    gt = np.array(row["ego_future_xy"], np.float32).reshape(64, 2)

    def decode(pa):
        act = torch.tensor(np.array(pa, np.float32).reshape(1, 1, 64, 2))
        hx = torch.tensor(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hr = torch.tensor(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hx, hr)
        return fut[0, 0, :, :2].numpy()

    alp = np.array(alpf.loc[clip_id, "pred_xy"], np.float32).reshape(64, 2)
    b, g, dp = decode(preds["bc"].loc[clip_id, "pred_action"]), decode(preds["grpo"].loc[clip_id, "pred_action"]), decode(preds["dpo"].loc[clip_id, "pred_action"])
    ade = lambda p: float(np.mean(np.linalg.norm(p - gt, axis=1)))
    lat = float(gt[-1, 1])
    nav = "turn left" if lat > 3 else "turn right" if lat < -3 else "drive forward" if abs(lat) < 1 else ("bear left" if lat > 0 else "bear right")
    speed = float(row["speed"]); o = np.zeros((1, 2))
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].imshow(Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")); ax[0].axis("off")
    ax[0].set_title(f"{nav} — clip {str(clip_id)[:8]}  v={speed:.1f} m/s", fontsize=16)
    ax[1].plot(*np.vstack([o, gt]).T, "k-", lw=3, label="GT")
    ax[1].plot(*np.vstack([o, alp]).T, ":", color="orange", lw=2.5, label=f"Alpamayo 1-cam ({ade(alp):.1f})")
    ax[1].plot(*np.vstack([o, b]).T, "b--", lw=2, label=f"BC ({ade(b):.1f})")
    ax[1].plot(*np.vstack([o, g]).T, "r-", lw=2.5, label=f"GRPO ({ade(g):.1f})")
    ax[1].plot(*np.vstack([o, dp]).T, "--", color="green", lw=2, label=f"DPO ({ade(dp):.1f})")
    ax[1].set_xlabel("x (forward, m)", fontsize=12); ax[1].set_ylabel("y (left, m)", fontsize=12)
    ax[1].legend(loc="upper right", fontsize=13); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"{nav} {clip_id}: GRPO={ade(g):.1f} BC={ade(b):.1f} DPO={ade(dp):.1f} Alp1cam={ade(alp):.1f}")
    return {"png": buf.getvalue()}


@app.local_entrypoint()
def single_bev(clip_id: str, out: str = "bev_single.png"):
    import pathlib
    r = bev_single.remote(clip_id=clip_id)
    pathlib.Path(out).write_bytes(r["png"]); print(f"saved {out}")


@app.function(image=viz_img, timeout=60 * 15, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def bev_pair(clip_top: str, clip_bottom: str):
    """Two clips stacked top/bottom; each row = cam + top-down, pi-drive style."""
    import io, numpy as np, pandas as pd, torch
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    ev = pd.read_parquet("/eval/eval_set.parquet").set_index("clip_id")
    alpf = pd.read_parquet("/eval/alpamayo_results_frontonly.parquet").set_index("clip_id")
    preds = {m: pd.read_parquet(f"/eval/pi_pred_{m}.parquet").set_index("clip_id") for m in ["bc", "grpo", "dpo"]}

    def decode(pa, row):
        act = torch.tensor(np.array(pa, np.float32).reshape(1, 1, 64, 2))
        hx = torch.tensor(np.array(row["ego_history_xyz"], np.float32).reshape(1, 1, 16, 3))
        hr = torch.tensor(np.array(row["ego_history_rot"], np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hx, hr)
        return fut[0, 0, :, :2].numpy()

    o = np.zeros((1, 2))
    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    for i, cid in enumerate([clip_top, clip_bottom]):
        row = ev.loc[cid]
        gt = np.array(row["ego_future_xy"], np.float32).reshape(64, 2)
        alp = np.array(alpf.loc[cid, "pred_xy"], np.float32).reshape(64, 2)
        b, g, dp = decode(preds["bc"].loc[cid, "pred_action"], row), decode(preds["grpo"].loc[cid, "pred_action"], row), decode(preds["dpo"].loc[cid, "pred_action"], row)
        ade = lambda p: float(np.mean(np.linalg.norm(p - gt, axis=1)))
        lat = float(gt[-1, 1])
        nav = "turn left" if lat > 3 else "turn right" if lat < -3 else "drive forward" if abs(lat) < 1 else ("bear left" if lat > 0 else "bear right")
        speed = float(row["speed"])
        ax0, ax1 = axes[i, 0], axes[i, 1]
        ax0.imshow(Image.open(io.BytesIO(row["image_bytes"])).convert("RGB")); ax0.axis("off")
        ax0.set_title(f"{nav} — clip {str(cid)[:8]}  v={speed:.1f} m/s", fontsize=16)
        ax1.plot(*np.vstack([o, gt]).T, "k-", lw=3, label="GT")
        ax1.plot(*np.vstack([o, alp]).T, ":", color="orange", lw=2.5, label=f"Alpamayo 1-cam ({ade(alp):.1f})")
        ax1.plot(*np.vstack([o, b]).T, "b--", lw=2, label=f"BC ({ade(b):.1f})")
        ax1.plot(*np.vstack([o, g]).T, "r-", lw=2.5, label=f"GRPO ({ade(g):.1f})")
        ax1.plot(*np.vstack([o, dp]).T, "--", color="green", lw=2, label=f"DPO ({ade(dp):.1f})")
        ax1.set_xlabel("x (forward, m)", fontsize=12); ax1.set_ylabel("y (left, m)", fontsize=12)
        ax1.legend(loc="best", fontsize=12); ax1.grid(alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight"); plt.close(fig)
    return {"png": buf.getvalue()}


@app.local_entrypoint()
def pair_bev(clip_top: str = "aae4505a-cc41-4a90-bcd9-fb2ebdcfff0e",
             clip_bottom: str = "3eb19f65-b338-4331-8871-5836dd8cd302",
             out: str = "bev_pair.png"):
    import pathlib
    r = bev_pair.remote(clip_top=clip_top, clip_bottom=clip_bottom)
    pathlib.Path(out).write_bytes(r["png"]); print(f"saved {out}")


@app.local_entrypoint()
def grpo_best_bev(nav: str = "turn left", out: str = "bev_grpo_best.png"):
    import pathlib
    r = bev_grpo_best.remote(nav=nav)
    if "error" in r:
        print(r["error"]); return
    pathlib.Path(out).write_bytes(r["png"])
    print(f"saved {out}  clip {r['clip_id']}  ades={ {k: round(v,2) for k,v in r['ades'].items()} }")


# ==========================================================================
# Metric 2: Replan jitter
#   J1 build_jitter_set  (CPU, parallel)  -> /eval/jitter_set.parquet
#   J2 run_pi05_jitter   (H100, sharded)  -> /eval/jitter_pred_{model}_s{shard}.parquet
#   J3 jitter_score      (CPU)            -> per-model mean/median/peak (m)
# ==========================================================================
JIT_W = 30            # consecutive replans per clip
JIT_START_S = 3.0     # first emission time (>= history span; within 20s clip)


@app.function(image=alpamayo_img, timeout=60 * 60, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE}, memory=16 * 1024,
              max_containers=8)
def _jitter_clip(clip_id: str):
    """Build JIT_W consecutive emission frames for one clip: front image + state +
    ego history. Writes a per-clip parquet to the volume; returns count + first error.
    Loads the clip's camera+egomotion ONCE (single avdi, retried) then samples all
    emission frames from cache — avoids hammering HF under concurrency."""
    import io, os, time, numpy as np, pandas as pd
    from PIL import Image
    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.geometry.rotation import so3_to_yaw_torch

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    dt = 0.1
    rows = []
    first_err = None

    def _load(t0):
        for attempt in range(4):
            try:
                return load_physical_aiavdataset(clip_id, t0_us=t0, avdi=avdi, maybe_stream=True,
                                                 camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV],
                                                 num_frames=1)
            except Exception as e:
                msg = str(e).lower()
                if attempt < 3 and ("429" in msg or "too many" in msg or "timeout" in msg or "zip" in msg or "connection" in msg):
                    time.sleep(2 ** attempt + 1)
                    continue
                raise

    for i in range(JIT_W):
        t0 = int((JIT_START_S + i * dt) * 1_000_000)
        try:
            d = _load(t0)
            hist = d["ego_history_xyz"][0, 0].numpy()
            speed = float(np.linalg.norm(np.diff(hist, axis=0), axis=1)[-1] / dt)
            yaws = so3_to_yaw_torch(d["ego_history_rot"][0, 0]).numpy()
            hrate = float((yaws[-1] - yaws[-2]) / dt) if len(yaws) > 1 else 0.0
            img = d["image_frames"][0, 0].permute(1, 2, 0).numpy().astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(img).resize((640, 480), Image.LANCZOS).save(buf, "JPEG", quality=90)
            rows.append({"clip_id": clip_id, "frame_idx": i, "t0_us": t0,
                         "image_bytes": buf.getvalue(), "speed": speed, "heading_rate": hrate,
                         "ego_history_xyz": hist.flatten().tolist(),
                         "ego_history_rot": d["ego_history_rot"][0, 0].numpy().flatten().tolist()})
        except Exception as e:
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue
    os.makedirs("/eval/jitter_batches", exist_ok=True)
    if rows:
        pd.DataFrame(rows).to_parquet(f"/eval/jitter_batches/{clip_id}.parquet", index=False)
        EVAL_VOL.commit()
    return {"clip_id": clip_id, "n": len(rows), "err": first_err}


@app.function(image=alpamayo_img, timeout=60 * 60, volumes={"/eval": EVAL_VOL}, memory=32 * 1024)
def build_jitter_set():
    import glob, pandas as pd
    clip_ids = list(pd.read_parquet("/eval/eval_set.parquet")["clip_id"])
    print(f"Building jitter set: {len(clip_ids)} clips x {JIT_W} frames")
    total, done, errs = 0, 0, []
    for r in _jitter_clip.map(clip_ids):
        total += r["n"]; done += 1
        if r["n"] == 0 and r["err"]:
            errs.append(r["err"])
        if done % 25 == 0:
            print(f"  {done}/{len(clip_ids)} clips, {total} frames so far"
                  + (f" | sample err: {errs[-1]}" if errs else ""))
    if errs:
        from collections import Counter
        print("error summary:", Counter(e[:70] for e in errs).most_common(3))
    EVAL_VOL.reload()
    parts = sorted(glob.glob("/eval/jitter_batches/*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    df.to_parquet("/eval/jitter_set.parquet", index=False)
    EVAL_VOL.commit()
    print(f"jitter_set: {len(df)} frames across {df['clip_id'].nunique()} clips")
    return {"frames": len(df), "clips": int(df["clip_id"].nunique())}


_PI_JITTER = r'''
import json, sys, io
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
OPENPI_DIR = "/opt/openpi"

def _patch_openpi():
    dst = f"{OPENPI_DIR}/src/openpi/policies/driving_policy.py"
    with open(dst, "w") as f:
        f.write("""
import dataclasses, einops, numpy as np
from openpi import transforms
from openpi.models import model as _model
def _parse_image(image):
    if isinstance(image, dict) and 'bytes' in image:
        import io; from PIL import Image as _I
        image = np.array(_I.open(io.BytesIO(image['bytes'])))
    else:
        image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255*image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
@dataclasses.dataclass(frozen=True)
class DrivingInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05
    def __call__(self, data):
        base = _parse_image(data["observation/image"])
        out = {"state": np.asarray(data["observation/state"], dtype=np.float32),
               "image": {"base_0_rgb": base, "left_wrist_0_rgb": np.zeros_like(base),
                         "right_wrist_0_rgb": np.zeros_like(base)},
               "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.False_,
                              "right_wrist_0_rgb": np.False_}}
        if "actions" in data: out["actions"] = data["actions"]
        out["prompt"] = "drive"
        return out
@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data):
        a = np.asarray(data["actions"], dtype=np.float32)
        if a.ndim == 1: a = a[np.newaxis, :]
        return {"actions": a}
""")
    gp = f"{OPENPI_DIR}/src/openpi/models/gemma.py"
    c = open(gp).read()
    if "gemma_2b_lora_driving" not in c:
        c = c.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]')
        c = c.replace('    if variant == "gemma_300m_lora":',
            '    if variant == "gemma_2b_lora_driving":\n'
            '        return Config(\n'
            '            width=2048, depth=18, mlp_dim=16_384,\n'
            '            num_heads=8, num_kv_heads=1, head_dim=256,\n'
            '            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},\n'
            '        )\n'
            '    if variant == "gemma_300m_lora":')
        open(gp, "w").write(c)
    cp = f"{OPENPI_DIR}/src/openpi/training/config.py"
    c = open(cp).read()
    if "pi05_driving" not in c:
        c = c.replace("import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy")
        ddc = (
            "\n@dataclasses.dataclass(frozen=True)\n"
            "class LeRobotDrivingDataConfig(DataConfigFactory):\n"
            "    @override\n"
            "    def create(self, assets_dirs, model_config):\n"
            "        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform({\n"
            '            "observation/image": "observation.images.front",\n'
            '            "observation/state": "observation.state",\n'
            '            "actions": "action", "prompt": "prompt"})])\n'
            "        data_transforms = _transforms.Group(\n"
            "            inputs=[driving_policy.DrivingInputs(model_type=model_config.model_type)],\n"
            "            outputs=[driving_policy.DrivingOutputs()])\n"
            "        model_transforms = ModelTransformFactory()(model_config)\n"
            "        return dataclasses.replace(self.create_base_config(assets_dirs, model_config),\n"
            "            repack_transforms=repack_transform, data_transforms=data_transforms,\n"
            '            model_transforms=model_transforms, action_sequence_keys=("action",))\n\n')
        c = c.replace("@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
                      ddc + "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:")
        dtc = (
            "\n    TrainConfig(\n"
            '        name="pi05_driving",\n'
            "        model=pi0_config.Pi0Config(pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving", action_expert_variant="gemma_300m"),\n'
            "        data=LeRobotDrivingDataConfig(repo_id=\"markmusic/pi05-physical-av-bc\",\n"
            "            base_config=DataConfig(prompt_from_task=True)),\n"
            "        weight_loader=weight_loaders.CheckpointWeightLoader(\"gs://openpi-assets/checkpoints/pi05_base/params\"),\n"
            "        freeze_filter=pi0_config.Pi0Config(pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving", action_expert_variant="gemma_300m").get_freeze_filter(),\n'
            "        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=750, peak_lr=3e-5, decay_steps=15_000, decay_lr=3e-6),\n"
            "        optimizer=_optimizer.AdamW(b1=0.9, b2=0.999, clip_gradient_norm=1.0),\n"
            "        num_train_steps=15_000, batch_size=96, fsdp_devices=1,\n"
            "        save_interval=500, log_interval=50,\n"
            '        checkpoint_base_dir="/pcache/checkpoints"),\n')
        c = c.replace("    *polaris_config.get_polaris_configs(),\n]",
                      "    *polaris_config.get_polaris_configs()," + dtc + "]")
        open(cp, "w").write(c)
    print("openpi patched")

def main():
    args = json.loads(sys.argv[1])
    ckpt_local, out_path = args["ckpt_local"], args["out_path"]
    shard, nshards = args["shard"], args["nshards"]
    _patch_openpi()
    from openpi.training import config as _config
    from openpi.policies.policy_config import create_trained_policy
    import openpi.transforms as transforms
    cfg = _config.get_config("pi05_driving")
    repack = transforms.Group(inputs=[transforms.RepackTransform({
        "observation/image": "observation.images.front", "observation/state": "observation.state",
        "actions": "action", "prompt": "prompt"})])
    policy = create_trained_policy(cfg, ckpt_local, repack_transforms=repack, default_prompt="drive")
    print("policy loaded")
    t = pq.read_table("/eval/jitter_set.parquet")
    n = t.num_rows
    cid = t.column("clip_id").to_pylist(); fi = t.column("frame_idx").to_pylist()
    imgs = t.column("image_bytes").to_pylist(); sp = t.column("speed").to_pylist()
    hr = t.column("heading_rate").to_pylist()
    idxs = list(range(shard, n, nshards))
    preds = []
    for c, i in enumerate(idxs):
        image = np.array(Image.open(io.BytesIO(imgs[i])).convert("RGB"))
        obs = {"observation.images.front": image,
               "observation.state": np.array([sp[i], hr[i]], dtype=np.float32),
               "action": np.zeros(128, dtype=np.float32), "prompt": "drive"}
        a = np.asarray(policy.infer(obs)["actions"], dtype=np.float32).reshape(-1)[:128]
        preds.append({"clip_id": cid[i], "frame_idx": fi[i], "pred_action": a.tolist()})
        if (c + 1) % 100 == 0:
            print(f"  shard {shard}: {c+1}/{len(idxs)}")
    import pandas as pd
    pd.DataFrame(preds).to_parquet(out_path, index=False)
    print(f"wrote {len(preds)} preds -> {out_path}")

if __name__ == "__main__":
    main()
'''


@app.function(image=pi_img, gpu="H100", timeout=60 * 60, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/pcache": PI_CACHE}, memory=64 * 1024)
def run_pi05_jitter(model: str, shard: int = 0, nshards: int = 4):
    import os, json, shutil, subprocess
    from huggingface_hub import snapshot_download
    repo = BC_REPO if model == "bc" else GRPO_REPO
    ckpt_local = f"/pcache/eval_ckpt_{model}"
    params_dir = f"{ckpt_local}/params"
    assets_dir = f"{ckpt_local}/assets/markmusic/pi05-physical-av-bc"
    os.makedirs(assets_dir, exist_ok=True)
    if not os.path.exists(f"{params_dir}/_METADATA"):
        if model == "bc":
            snapshot_download(repo, local_dir=params_dir, repo_type="model")
        else:
            snapshot_download(repo, local_dir=ckpt_local, repo_type="model", allow_patterns=["params/**", "assets/**"])
        PI_CACHE.commit()
    if not os.path.exists(f"{assets_dir}/norm_stats.json"):
        ns = snapshot_download(GRPO_REPO, repo_type="model",
                               allow_patterns=["assets/markmusic/pi05-physical-av-bc/norm_stats.json"])
        shutil.copy2(f"{ns}/assets/markmusic/pi05-physical-av-bc/norm_stats.json", f"{assets_dir}/norm_stats.json")
    script = "/tmp/pi_jitter.py"; open(script, "w").write(_PI_JITTER)
    out_path = f"/eval/jitter_pred_{model}_s{shard}.parquet"
    args = json.dumps({"ckpt_local": ckpt_local, "out_path": out_path, "shard": shard, "nshards": nshards})
    r = subprocess.run([f"{OPENPI_DIR}/.venv/bin/python", "-u", script, args], cwd=OPENPI_DIR, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pi05 jitter {model} shard {shard} failed ({r.returncode})")
    EVAL_VOL.commit()
    return {"model": model, "shard": shard, "out": out_path}


@app.function(image=alpamayo_img, timeout=60 * 30, volumes={"/eval": EVAL_VOL}, memory=16 * 1024)
def jitter_score(nshards: int = 4):
    import glob, json, numpy as np, pandas as pd, torch
    from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
    aspace = UnicycleAccelCurvatureActionSpace()
    jset = pd.read_parquet("/eval/jitter_set.parquet")
    jset["key"] = jset["clip_id"] + "_" + jset["frame_idx"].astype(str)
    jset = jset.set_index("key")

    def decode(chunk, hx, hr):
        act = torch.tensor(np.array(chunk, np.float32).reshape(1, 1, 64, 2))
        hxt = torch.tensor(np.array(hx, np.float32).reshape(1, 1, 16, 3))
        hrt = torch.tensor(np.array(hr, np.float32).reshape(1, 1, 16, 3, 3))
        fut, _ = aspace.action_to_traj(act, hxt, hrt)
        return fut[0, 0, :, :2].numpy()

    out = {}
    for model in ["bc", "grpo"]:
        parts = sorted(glob.glob(f"/eval/jitter_pred_{model}_s*.parquet"))
        pred = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        pred["key"] = pred["clip_id"] + "_" + pred["frame_idx"].astype(str)
        pred = pred.set_index("key")
        clip_jit, clip_peak = [], []
        for cid, grp in pred.reset_index().groupby("clip_id"):
            frames = sorted(grp["frame_idx"].tolist())
            dec = {}
            for fr in frames:
                k = f"{cid}_{fr}"
                row = jset.loc[k]
                dec[fr] = (decode(pred.loc[k, "pred_action"], row["ego_history_xyz"], row["ego_history_rot"]),
                           np.array(row["ego_history_rot"], np.float32).reshape(16, 3, 3),
                           np.array(row["ego_history_xyz"], np.float32).reshape(16, 3))
            js = []
            for a, b in zip(frames[:-1], frames[1:]):
                if b != a + 1:
                    continue
                P_t = dec[a][0]                       # ego frame at t
                P_tp1 = dec[b][0]                     # ego frame at t+1
                R = dec[b][1][-2][:2, :2]             # frame-t orientation in frame-t+1
                tvec = dec[b][2][-2][:2]              # frame-t origin in frame-t+1
                P_t_in_tp1 = (R @ P_t.T).T + tvec
                d = np.linalg.norm(P_t_in_tp1[1:] - P_tp1[:-1], axis=1)  # 63 overlap waypoints
                js.append(float(d.mean()))
            if js:
                clip_jit.append(float(np.mean(js))); clip_peak.append(float(np.max(js)))
        out[model] = {"n_clips": len(clip_jit),
                      "mean_m": float(np.mean(clip_jit)), "median_m": float(np.median(clip_jit)),
                      "peak_m": float(np.mean(clip_peak))}
    print(json.dumps(out, indent=2))
    return out


@app.local_entrypoint()
def run_jitter(nshards: int = 4):
    import json
    print("=== J1 build jitter set ===")
    print(build_jitter_set.remote())
    print("\n=== J2 pi0.5 inference (BC+GRPO, sharded H100) ===")
    calls = []
    for model in ["bc", "grpo"]:
        for s in range(nshards):
            calls.append(run_pi05_jitter.spawn(model, s, nshards))
    for c in calls:
        c.get()
    print("\n=== J3 jitter score ===")
    print(json.dumps(jitter_score.remote(nshards=nshards), indent=2))


@app.function(image=alpamayo_img, timeout=600, secrets=[HF_SECRET],
              volumes={"/eval": EVAL_VOL, "/cache": HF_CACHE})
def dbg_jitter(clip_id: str):
    import physical_ai_av
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    for t0 in [3_000_000, 5_000_000]:
        d = load_physical_aiavdataset(clip_id, t0_us=t0, avdi=avdi, maybe_stream=True,
                                      camera_features=[avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV], num_frames=1)
        print(f"t0={t0}: image_frames {tuple(d['image_frames'].shape)} OK")
    return "ok"


@app.local_entrypoint()
def dbg(clip_id: str = "1bc6dae4-a3a9-4afe-a882-a84d8ca9e0d1"):
    print(dbg_jitter.remote(clip_id=clip_id))


@app.local_entrypoint()
def run_jitter_infer(nshards: int = 4):
    """J2 + J3 only (assumes /eval/jitter_set.parquet already built)."""
    import json
    calls = [run_pi05_jitter.spawn(m, s, nshards) for m in ["bc", "grpo"] for s in range(nshards)]
    for c in calls:
        c.get()
    print(json.dumps(jitter_score.remote(nshards=nshards), indent=2))
