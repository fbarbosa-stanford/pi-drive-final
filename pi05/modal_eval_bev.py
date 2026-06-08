"""BEV trajectory visualization: pi0.5 predictions vs ground truth.

Loads the trained BC checkpoint, runs inference on random eval clips,
converts (accel, curvature) actions back to XYZ trajectories via the
unicycle kinematic model, and renders side-by-side BEV videos.

Usage:
    modal run pi05/modal_eval_bev.py::visualize_bev --n-samples 10
"""

import modal

CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"
HF_CHECKPOINT_REPO = "markmusic/pi05-driving-bc-v2-checkpoint"

app = modal.App("pi05-bev-eval")

pi05_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang", "ffmpeg")
    .pip_install("uv")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
    )
    .pip_install("huggingface_hub", "pyarrow", "matplotlib")
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Inline helper script — runs inside the openpi venv's Python
# ---------------------------------------------------------------------------

_EVAL_SCRIPT = r'''
"""BEV eval helper — executed via /opt/openpi/.venv/bin/python"""
import glob
import io
import json
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from PIL import Image

OPENPI_DIR = "/opt/openpi"
CACHE_DIR = "/cache"


def _patch_openpi():
    """Minimal openpi patches for inference only."""

    driving_policy_dst = f"{OPENPI_DIR}/src/openpi/policies/driving_policy.py"
    with open(driving_policy_dst, "w") as f:
        f.write("""
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict) and 'bytes' in image:
        import io
        from PIL import Image as _PILImage
        image = np.array(_PILImage.open(io.BytesIO(image['bytes'])))
    else:
        image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image

@dataclasses.dataclass(frozen=True)
class DrivingInputs(transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI05
    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        inputs["prompt"] = "drive"
        return inputs

@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        return {"actions": actions}
""")

    # Patch gemma.py — add gemma_2b_lora_driving variant
    gemma_path = f"{OPENPI_DIR}/src/openpi/models/gemma.py"
    with open(gemma_path, "r") as f:
        content = f.read()
    if "gemma_2b_lora_driving" not in content:
        content = content.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]',
        )
        content = content.replace(
            '    if variant == "gemma_300m_lora":',
            '    if variant == "gemma_2b_lora_driving":\n'
            '        return Config(\n'
            '            width=2048, depth=18, mlp_dim=16_384,\n'
            '            num_heads=8, num_kv_heads=1, head_dim=256,\n'
            '            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},\n'
            '        )\n'
            '    if variant == "gemma_300m_lora":',
        )
        with open(gemma_path, "w") as f:
            f.write(content)

    # Patch config.py — add pi05_driving config
    config_path = f"{OPENPI_DIR}/src/openpi/training/config.py"
    with open(config_path, "r") as f:
        content = f.read()
    if "pi05_driving" not in content:
        content = content.replace(
            "import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy",
        )
        driving_data_config = (
            "\n@dataclasses.dataclass(frozen=True)\n"
            "class LeRobotDrivingDataConfig(DataConfigFactory):\n"
            "    @override\n"
            "    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:\n"
            "        repack_transform = _transforms.Group(\n"
            "            inputs=[_transforms.RepackTransform({\n"
            '                "observation/image": "observation.images.front",\n'
            '                "observation/state": "observation.state",\n'
            '                "actions": "action",\n'
            '                "prompt": "prompt",\n'
            "            })])\n"
            "        data_transforms = _transforms.Group(\n"
            "            inputs=[driving_policy.DrivingInputs(model_type=model_config.model_type)],\n"
            "            outputs=[driving_policy.DrivingOutputs()],\n"
            "        )\n"
            "        model_transforms = ModelTransformFactory()(model_config)\n"
            "        return dataclasses.replace(\n"
            "            self.create_base_config(assets_dirs, model_config),\n"
            "            repack_transforms=repack_transform,\n"
            "            data_transforms=data_transforms,\n"
            "            model_transforms=model_transforms,\n"
            '            action_sequence_keys=("action",),\n'
            "        )\n\n"
        )
        content = content.replace(
            "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
            driving_data_config + "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
        )
        driving_train_config = (
            "\n    TrainConfig(\n"
            '        name="pi05_driving",\n'
            "        model=pi0_config.Pi0Config(\n"
            "            pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving",\n'
            '            action_expert_variant="gemma_300m",\n'
            "        ),\n"
            "        data=LeRobotDrivingDataConfig(\n"
            '            repo_id="markmusic/pi05-physical-av-bc",\n'
            "            base_config=DataConfig(prompt_from_task=True),\n"
            "        ),\n"
            "        weight_loader=weight_loaders.CheckpointWeightLoader(\n"
            '            "gs://openpi-assets/checkpoints/pi05_base/params"\n'
            "        ),\n"
            "        freeze_filter=pi0_config.Pi0Config(\n"
            "            pi05=True, action_dim=128, action_horizon=1,\n"
            '            paligemma_variant="gemma_2b_lora_driving",\n'
            '            action_expert_variant="gemma_300m",\n'
            "        ).get_freeze_filter(),\n"
            "        lr_schedule=_optimizer.CosineDecaySchedule(\n"
            "            warmup_steps=750, peak_lr=3e-5, decay_steps=15_000, decay_lr=3e-6,\n"
            "        ),\n"
            "        optimizer=_optimizer.AdamW(b1=0.9, b2=0.999, clip_gradient_norm=1.0),\n"
            "        num_train_steps=15_000, batch_size=96, fsdp_devices=1,\n"
            "        save_interval=500, log_interval=50,\n"
            '        checkpoint_base_dir="/cache/checkpoints",\n'
            "    ),\n"
        )
        content = content.replace(
            "    *polaris_config.get_polaris_configs(),\n]",
            "    *polaris_config.get_polaris_configs()," + driving_train_config + "]",
        )
        with open(config_path, "w") as f:
            f.write(content)

    print("openpi patched for eval")


def actions_to_xy(actions_2d, speed, dt=0.1):
    """Integrate (accel, curvature) via unicycle model to XY trajectory."""
    v = speed
    heading = 0.0
    x, y = 0.0, 0.0
    pts = []
    for a, kappa in actions_2d:
        v = max(v + float(a) * dt, 0.0)
        heading += v * float(kappa) * dt
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        pts.append([x, y])
    return np.array(pts)


def main():
    args = json.loads(sys.argv[1])
    n_samples = args["n_samples"]
    seed = args["seed"]
    ckpt_local = args["ckpt_local"]
    output_tag = args.get("output_tag", "default")
    model_label = args.get("model_label", "pi0.5")

    _patch_openpi()

    from openpi.training import config as _config
    from openpi.policies.policy_config import create_trained_policy
    import openpi.transforms as transforms

    config = _config.get_config("pi05_driving")
    repack = transforms.Group(
        inputs=[transforms.RepackTransform({
            "observation/image": "observation.images.front",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })]
    )

    print("Loading policy...")
    policy = create_trained_policy(
        config, ckpt_local,
        repack_transforms=repack,
        default_prompt="drive",
    )
    print("Policy loaded.")

    # Load eval dataset
    eval_path = f"{CACHE_DIR}/hf/lerobot/markmusic/pi05-physical-av-bc-eval"
    eval_parquets = sorted(glob.glob(f"{eval_path}/data/**/*.parquet", recursive=True))
    print(f"Found {len(eval_parquets)} eval parquet files")

    tables = [pq.read_table(p) for p in eval_parquets]
    full_table = pa.concat_tables(tables)
    n_total = full_table.num_rows
    print(f"Total eval samples: {n_total}")

    rng = np.random.RandomState(seed)
    indices = rng.choice(n_total, size=min(n_samples, n_total), replace=False)
    indices.sort()

    output_dir = f"{CACHE_DIR}/bev_eval_{output_tag}"
    os.makedirs(output_dir, exist_ok=True)
    frames = []
    all_ade, all_fde = [], []

    for i, idx in enumerate(indices):
        row = full_table.slice(idx, 1)

        img_struct = row.column("observation.images.front")[0].as_py()
        img_bytes = img_struct["bytes"]
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(image)

        state = np.array(row.column("observation.state")[0].as_py(), dtype=np.float32)
        gt_actions_flat = np.array(row.column("action")[0].as_py(), dtype=np.float32)
        gt_actions = gt_actions_flat.reshape(64, 2)

        obs = {
            "observation.images.front": image,
            "observation.state": state,
            "action": gt_actions_flat,
            "prompt": "drive",
        }
        result = policy.infer(obs)
        pred_actions_flat = result["actions"]
        if pred_actions_flat.ndim == 1:
            pred_actions = pred_actions_flat.reshape(64, 2)
        else:
            pred_actions = pred_actions_flat.reshape(-1, 2)[:64]

        infer_ms = result["policy_timing"]["infer_ms"]
        print(f"  Sample {i+1}/{len(indices)} (idx={idx}): infer={infer_ms:.0f}ms")

        speed = float(state[0])
        gt_xy = actions_to_xy(gt_actions, speed)
        pred_xy = actions_to_xy(pred_actions, speed)

        ade = np.mean(np.linalg.norm(gt_xy - pred_xy, axis=1))
        fde = np.linalg.norm(gt_xy[-1] - pred_xy[-1])
        all_ade.append(ade)
        all_fde.append(fde)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        axes[0].imshow(image_np)
        axes[0].set_title(f"Front Camera (speed={speed:.1f} m/s)", fontsize=12)
        axes[0].axis("off")

        ax = axes[1]
        ax.plot(gt_xy[:, 1], gt_xy[:, 0], "g-o", markersize=2, linewidth=2, label="Ground Truth")
        ax.plot(pred_xy[:, 1], pred_xy[:, 0], "r-o", markersize=2, linewidth=2, label=f"{model_label} Predicted")
        ax.plot(0, 0, "k^", markersize=12, label="Ego (t=0)")

        for t_step in range(0, 64, 10):
            ax.plot(gt_xy[t_step, 1], gt_xy[t_step, 0], "gs", markersize=6)
            ax.plot(pred_xy[t_step, 1], pred_xy[t_step, 0], "rs", markersize=6)
            if t_step > 0:
                ax.annotate(f"{t_step/10:.0f}s", (gt_xy[t_step, 1]+0.3, gt_xy[t_step, 0]),
                           fontsize=8, color="green")

        ax.set_xlabel("Lateral (m)", fontsize=11)
        ax.set_ylabel("Forward (m)", fontsize=11)
        ax.set_title(f"{model_label} — ADE={ade:.2f}m, FDE={fde:.2f}m", fontsize=12)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

        # Fixed lateral range so straight trajectories don't collapse the plot
        all_lat = np.concatenate([gt_xy[:, 1], pred_xy[:, 1]])
        lat_center = np.mean(all_lat)
        lat_extent = max(np.ptp(all_lat), 1.0)
        fwd_extent = max(np.ptp(np.concatenate([gt_xy[:, 0], pred_xy[:, 0]])), 1.0)
        half_lat = max(fwd_extent * 0.4, lat_extent * 0.6, 15.0)
        ax.set_xlim(lat_center + half_lat, lat_center - half_lat)
        ax.set_aspect("equal")

        plt.tight_layout()
        frame_path = f"{output_dir}/frame_{i:03d}.png"
        fig.savefig(frame_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        frames.append(frame_path)

        print(f"    ADE={ade:.2f}m  FDE={fde:.2f}m  speed={speed:.1f}m/s")

    # Compose video
    video_path = f"{output_dir}/bev_eval.mp4"
    frame_list = f"{output_dir}/frames.txt"
    with open(frame_list, "w") as f:
        for fp in frames:
            f.write(f"file '{fp}'\nduration 2\n")
        f.write(f"file '{frames[-1]}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", frame_list, "-vf", "scale=1920:-2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        video_path,
    ], check=True)

    print(f"\nVideo saved: {video_path}")
    print(f"\n{'='*50}")
    print(f"Evaluated {len(indices)} samples from held-out eval set")
    print(f"Mean ADE: {np.mean(all_ade):.2f}m")
    print(f"Mean FDE: {np.mean(all_fde):.2f}m")
    print(f"Video: {video_path}")

    # Write results JSON for the outer Modal function to read
    import base64
    frame_data = []
    for fp in frames:
        with open(fp, "rb") as img_f:
            frame_data.append(base64.b64encode(img_f.read()).decode("ascii"))
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "video_path": video_path,
            "n_samples": len(indices),
            "mean_ade": float(np.mean(all_ade)),
            "mean_fde": float(np.mean(all_fde)),
            "frames_b64": frame_data,
        }, f)


if __name__ == "__main__":
    main()
'''


@app.function(
    image=pi05_image,
    gpu="H100",
    timeout=60 * 30,
    volumes={"/cache": cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def visualize_bev(n_samples: int = 10, seed: int = 42, checkpoint_repo: str = HF_CHECKPOINT_REPO):
    import json
    import os
    import subprocess
    import tempfile

    # Download checkpoint from HF into params/ subdirectory
    # create_trained_policy expects checkpoint_dir/params/ structure
    from huggingface_hub import snapshot_download
    repo_slug = checkpoint_repo.replace("/", "--")
    ckpt_local = f"{CACHE_DIR}/eval_ckpt_{repo_slug}"
    params_dir = f"{ckpt_local}/params"
    if not os.path.exists(f"{params_dir}/_METADATA"):
        print(f"Downloading checkpoint from HF: {checkpoint_repo}...")
        snapshot_download(
            repo_id=checkpoint_repo,
            local_dir=params_dir,
            repo_type="model",
        )
        cache_volume.commit()
    print(f"Checkpoint params at {params_dir}: {os.listdir(params_dir)}")

    # Copy norm_stats from training assets into checkpoint assets dir
    # create_trained_policy looks for checkpoint_dir/assets/<asset_id>/norm_stats.json
    import shutil
    train_assets = f"{OPENPI_DIR}/assets/pi05_driving/markmusic/pi05-physical-av-bc"
    ckpt_assets = f"{ckpt_local}/assets/markmusic/pi05-physical-av-bc"
    # Try from openpi assets first, fall back to training cache
    train_cache_assets = f"{CACHE_DIR}/checkpoints/pi05_driving/bc-coldstart-v2/assets/markmusic/pi05-physical-av-bc"
    for src in [train_assets, train_cache_assets]:
        norm_file = f"{src}/norm_stats.json"
        if os.path.exists(norm_file):
            os.makedirs(ckpt_assets, exist_ok=True)
            shutil.copy2(norm_file, f"{ckpt_assets}/norm_stats.json")
            print(f"Copied norm_stats from {src}")
            break
    else:
        # Compute norm stats on the fly
        print("No cached norm_stats found, computing from openpi assets dir...")
        # Check if they exist at the default openpi assets location
        for root, dirs, files in os.walk(f"{OPENPI_DIR}/assets"):
            if "norm_stats.json" in files:
                os.makedirs(ckpt_assets, exist_ok=True)
                shutil.copy2(os.path.join(root, "norm_stats.json"), f"{ckpt_assets}/norm_stats.json")
                print(f"Found norm_stats at {root}")
                break
        # Also check the training checkpoints on the volume
        for root, dirs, files in os.walk(f"{CACHE_DIR}/checkpoints"):
            if "norm_stats.json" in files:
                os.makedirs(ckpt_assets, exist_ok=True)
                shutil.copy2(os.path.join(root, "norm_stats.json"), f"{ckpt_assets}/norm_stats.json")
                print(f"Found norm_stats at {root}")
                break

    # Write and run the eval script via openpi's venv Python
    script_path = "/tmp/bev_eval_helper.py"
    with open(script_path, "w") as f:
        f.write(_EVAL_SCRIPT)

    output_tag = repo_slug
    if checkpoint_repo == HF_CHECKPOINT_REPO:
        model_label = "pi0.5 v2 (207K samples)"
    elif "pi05-physical-av-bc-checkpoint" in checkpoint_repo:
        model_label = "pi0.5 v1 (25K samples)"
    else:
        model_label = checkpoint_repo.split("/")[-1]
    args_json = json.dumps({
        "n_samples": n_samples,
        "seed": seed,
        "ckpt_local": ckpt_local,
        "output_tag": output_tag,
        "model_label": model_label,
    })

    result = subprocess.run(
        [f"{OPENPI_DIR}/.venv/bin/python", "-u", script_path, args_json],
        cwd=OPENPI_DIR,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Eval script failed with exit code {result.returncode}")

    cache_volume.commit()

    output_dir = f"{CACHE_DIR}/bev_eval_{output_tag}"
    results_file = f"{output_dir}/results.json"
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        # Return frame PNGs as bytes for local saving
        import base64
        frames_b64 = data.pop("frames_b64", [])
        data["frame_pngs"] = [base64.b64decode(b) for b in frames_b64]
        return data
    return {"output_dir": output_dir, "n_samples": n_samples}


@app.local_entrypoint()
def main(n_samples: int = 10, seed: int = 42, checkpoint_repo: str = HF_CHECKPOINT_REPO):
    import pathlib
    result = visualize_bev.remote(n_samples=n_samples, seed=seed, checkpoint_repo=checkpoint_repo)

    tag = checkpoint_repo.split("/")[-1]
    out_dir = pathlib.Path(f"pi05/runs/bev_eval_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_pngs = result.get("frame_pngs", [])
    for i, png_bytes in enumerate(frame_pngs):
        path = out_dir / f"frame_{i:03d}.png"
        path.write_bytes(png_bytes)
        print(f"Saved {path}")

    print(f"\nMean ADE: {result.get('mean_ade', '?'):.2f}m")
    print(f"Mean FDE: {result.get('mean_fde', '?'):.2f}m")
    print(f"Saved {len(frame_pngs)} frames to {out_dir}")
