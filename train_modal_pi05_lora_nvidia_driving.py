"""
π₀.5 fine-tuning on Modal (NVIDIA AV → LeRobot → lerobot-train).

Profiles
--------
generalize (recommended): Train/val split, expert-only finetune (stable), held-out metrics.
learn:                  Memorization smoke test (subset of episodes, expert-only).
lora:                   Minimal integration smoke test.

Launch:
  modal run --detach train_modal_pi05_lora_nvidia_driving.py::run_pi05_smoke_test --profile generalize

  modal run --detach train_modal_pi05_lora_nvidia_driving.py::run_pi05_smoke_test --profile generalize --skip-prepare

Eval only (after training):
  modal run --detach train_modal_pi05_lora_nvidia_driving.py::eval_main \\
    --output-dir /vol/runs/pi05_nvidia_smoke_generalize_<timestamp>
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import modal

APP_NAME = "pi05-lora-nvidia-driving"
VOLUME_NAME = "pi05-nvidia-driving"

app = modal.App(APP_NAME)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

_PKG_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        "ffmpeg",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
    )
    .pip_install(
        "physical_ai_av",
        "mediapy",
        "opencv-python-headless",
        "wandb",
        "huggingface_hub",
        "peft",
        "numpy",
        "pandas",
        "pillow",
    )
    .run_commands(
        "GIT_LFS_SKIP_SMUDGE=1 pip install "
        "'lerobot[pi,peft,dataset] @ git+https://github.com/huggingface/lerobot.git@main'"
    )
    .add_local_dir(_PKG_DIR, remote_path="/app/pi_05_drives")
)

DATASET_ROOT_LORA = "/vol/nvidia_av_lerobot"
DATASET_ROOT_LEARN = "/vol/nvidia_av_lerobot_learn"
DATASET_ROOT_GEN = "/vol/nvidia_av_lerobot_gen"
CHECKPOINT_ROOT = "/vol/runs/pi05_nvidia_smoke"
HF_CACHE_DIR = "/vol/hf_cache"
SPLIT_FILENAME = "meta/train_val_split.json"

ProfileName = Literal["learn", "lora", "generalize"]

PROFILE_DEFAULTS: dict[str, dict] = {
    # Train/val split. Expert-only + pi05 default LR (LoRA@5e-6 showed flat loss + rising grad_norm).
    "generalize": {
        "dataset_root": DATASET_ROOT_GEN,
        "repo_id": "local/nvidia_av_gen",
        "max_clips": 60,
        "frames_per_clip": 100,
        "clip_seed": 42,
        "val_ratio": 0.2,
        "min_val_episodes": 8,
        "train_episodes": None,  # filled after prepare from split file
        "steps": 12_000,
        "batch_size": 8,
        "log_freq": 50,
        "save_freq": 2_000,
        "chunk_size": 16,
        "n_action_steps": 16,
        "train_expert_only": True,
        "freeze_vision_encoder": True,
        "use_lora": False,
        "peft_r": 16,
        "optimizer_lr": 1e-5,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1_000,
        "scheduler_decay_steps": 12_000,
        "wandb_project": "pi05-nvidia-driving",
        "wandb_run_name": "pi05_gen_expert_stable",
        "ego_frame_actions": True,
        "run_val_eval": True,
    },
    "learn": {
        "dataset_root": DATASET_ROOT_LEARN,
        "repo_id": "local/nvidia_av_learn",
        "max_clips": 24,
        "frames_per_clip": 80,
        "clip_seed": 0,
        "val_ratio": 0.0,
        "min_val_episodes": 0,
        "train_episodes": list(range(16)),
        "steps": 8_000,
        "batch_size": 16,
        "log_freq": 10,
        "save_freq": 1_000,
        "chunk_size": 16,
        "n_action_steps": 16,
        "train_expert_only": True,
        "freeze_vision_encoder": True,
        "use_lora": False,
        "peft_r": 16,
        "optimizer_lr": 1e-5,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 200,
        "scheduler_decay_steps": 8_000,
        "wandb_project": "pi05-nvidia-driving-smoke",
        "wandb_run_name": "pi05_learn_memorize",
        "ego_frame_actions": True,
        "run_val_eval": False,
    },
    "lora": {
        "dataset_root": DATASET_ROOT_LORA,
        "repo_id": "local/nvidia_av_smoke",
        "max_clips": 8,
        "frames_per_clip": 120,
        "clip_seed": 0,
        "val_ratio": 0.0,
        "min_val_episodes": 0,
        "train_episodes": None,
        "steps": 2_500,
        "batch_size": 4,
        "log_freq": 25,
        "save_freq": 500,
        "chunk_size": 50,
        "n_action_steps": 50,
        "train_expert_only": False,
        "freeze_vision_encoder": False,
        "use_lora": True,
        "peft_r": 16,
        "optimizer_lr": None,
        "optimizer_grad_clip_norm": None,
        "scheduler_warmup_steps": None,
        "scheduler_decay_steps": None,
        "wandb_project": "pi05-nvidia-driving-smoke",
        "wandb_run_name": "pi05_lora_nvidia_av_smoke",
        "ego_frame_actions": False,
        "run_val_eval": False,
    },
}


def _setup_huggingface_auth() -> str:
    """Login and warm the HF cache (PaliGemma is gated)."""
    from huggingface_hub import hf_hub_download, login, whoami

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN missing on Modal. Update secret: "
            "modal secret create huggingface HF_TOKEN=<token> --force"
        )

    os.makedirs(HF_CACHE_DIR, exist_ok=True)
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
    os.environ["HF_HOME"] = HF_CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"

    login(token=hf_token, add_to_git_credential=False)
    user = whoami()
    print(f"HF authenticated as: {user.get('name', user)}")

    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        path = hf_hub_download(
            "google/paligemma-3b-pt-224",
            filename,
            token=hf_token,
        )
        print(f"Cached {filename} -> {path}")

    volume.commit()
    return hf_token


def _resize_rgb(frame, size: int = 224):
    import cv2

    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def _world_delta_to_ego(delta_xy, yaw: float):
    """Rotate XY delta into approximate ego frame (forward / lateral)."""
    import numpy as np

    c, s = np.cos(-yaw), np.sin(-yaw)
    return np.array(
        [c * delta_xy[0] - s * delta_xy[1], s * delta_xy[0] + c * delta_xy[1]],
        dtype=np.float32,
    )


def _ego_to_world_xy(dx_ego: float, dy_ego: float, yaw: float) -> tuple[float, float]:
    """Inverse of _world_delta_to_ego for visualization (ego action → world motion)."""
    import numpy as np

    c, s = np.cos(yaw), np.sin(yaw)
    wx = c * dx_ego - s * dy_ego
    wy = s * dx_ego + c * dy_ego
    return float(wx), float(wy)


def _motion_heading_rad(dx_ego: float, dy_ego: float, yaw: float) -> float:
    """World-frame direction of motion (radians); use for steering overlay."""
    import numpy as np

    wx, wy = _ego_to_world_xy(dx_ego, dy_ego, yaw)
    return float(np.arctan2(wy, wx + 1e-6))


def make_train_val_split(
    num_episodes: int,
    val_ratio: float,
    min_val_episodes: int,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """Hold out whole episodes (clips) for validation — no frame leakage across split."""
    if num_episodes < 2 or val_ratio <= 0:
        return list(range(num_episodes)), []

    n_val = max(min_val_episodes, int(round(num_episodes * val_ratio)))
    n_val = min(n_val, num_episodes // 2)
    rng = random.Random(seed)
    indices = list(range(num_episodes))
    rng.shuffle(indices)
    val_episodes = sorted(indices[:n_val])
    train_episodes = sorted(indices[n_val:])
    return train_episodes, val_episodes


def write_split_file(dataset_root: Path, train_episodes: list[int], val_episodes: list[int]) -> Path:
    split_path = dataset_root / SPLIT_FILENAME
    split_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train_episodes": train_episodes,
        "val_episodes": val_episodes,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    split_path.write_text(json.dumps(payload, indent=2))
    return split_path


def load_or_create_split(
    dataset_root: Path,
    repo_id: str,
    val_ratio: float,
    min_val_episodes: int,
    seed: int,
    force_rebuild: bool = False,
) -> tuple[list[int], list[int]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    split_path = dataset_root / SPLIT_FILENAME
    if split_path.exists() and not force_rebuild:
        data = json.loads(split_path.read_text())
        return data["train_episodes"], data["val_episodes"]

    meta = LeRobotDataset(repo_id=repo_id, root=dataset_root).meta
    train_eps, val_eps = make_train_val_split(
        meta.total_episodes, val_ratio, min_val_episodes, seed=seed
    )
    write_split_file(dataset_root, train_eps, val_eps)
    return train_eps, val_eps


def prepare_nvidia_lerobot_dataset(
    dataset_root: str,
    repo_id: str,
    max_clips: int = 8,
    frames_per_clip: int = 120,
    fps: int = 10,
    skip_prepare: bool = False,
    ego_frame_actions: bool = False,
    clip_seed: int = 42,
    val_ratio: float = 0.0,
    min_val_episodes: int = 0,
) -> str:
    """Convert NVIDIA AV clips into LeRobot format on the volume."""
    import numpy as np
    import physical_ai_av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    out_root = Path(dataset_root)
    if skip_prepare and (out_root / "meta" / "info.json").exists():
        print(f"Dataset already exists at {out_root}, skipping conversion.")
        if val_ratio > 0:
            train_eps, val_eps = load_or_create_split(
                out_root, repo_id, val_ratio, min_val_episodes, clip_seed
            )
            print(f"  split: {len(train_eps)} train / {len(val_eps)} val episodes")
        return str(out_root)

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
    camera = avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV

    all_clip_ids = list(avdi.clip_index.index)
    rng = random.Random(clip_seed)
    shuffled = all_clip_ids.copy()
    rng.shuffle(shuffled)
    clip_ids = shuffled[:max_clips]
    print(f"Converting {len(clip_ids)} NVIDIA AV clips -> {out_root} (seed={clip_seed})")
    print(f"  ego_frame_actions={ego_frame_actions}")

    features = {
        "observation.state": {"dtype": "float32", "shape": (8,), "names": None},
        "action": {"dtype": "float32", "shape": (3,), "names": None},
        "observation.images.base_0_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
    }

    if out_root.exists():
        import shutil

        shutil.rmtree(out_root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=out_root,
        use_videos=False,
    )

    dt_us = int(1_000_000 / fps)

    for clip_idx, clip_id in enumerate(clip_ids):
        print(f"[{clip_idx + 1}/{len(clip_ids)}] clip {clip_id}")
        try:
            egomotion = avdi.get_clip_feature(
                clip_id, feature=avdi.features.LABELS.EGOMOTION, maybe_stream=True
            )
            video = avdi.get_clip_feature(clip_id, feature=camera, maybe_stream=True)
        except Exception as exc:
            print(f"  skip clip (load failed): {exc}")
            continue

        timestamps_us = dt_us * np.arange(frames_per_clip)
        try:
            frames, actual_ts = video.decode_images_from_timestamps(timestamps_us)
            poses = egomotion(actual_ts)
        except Exception as exc:
            print(f"  skip clip (decode failed): {exc}")
            continue

        xyz = np.asarray(poses.pose.translation, dtype=np.float32)
        if xyz.ndim != 2 or xyz.shape[0] < 2:
            print("  skip clip (bad egomotion shape)")
            continue

        n = min(len(frames), len(xyz) - 1)
        for i in range(n):
            pos = xyz[i]
            nxt = xyz[i + 1]
            delta = (nxt - pos).astype(np.float32)
            vel = delta * fps
            yaw = float(np.arctan2(vel[1], vel[0] + 1e-6))

            if ego_frame_actions:
                delta_xy_ego = _world_delta_to_ego(delta[:2], yaw)
                action = np.array([delta_xy_ego[0], delta_xy_ego[1], delta[2]], dtype=np.float32)
            else:
                action = delta

            dataset.add_frame(
                {
                    "observation.state": np.concatenate(
                        [
                            pos,
                            vel,
                            np.array([yaw, np.linalg.norm(vel[:2])], dtype=np.float32),
                        ]
                    ).astype(np.float32),
                    "action": action,
                    "observation.images.base_0_rgb": _resize_rgb(frames[i]),
                    "task": "Drive safely following the road.",
                }
            )
        dataset.save_episode()

    dataset.finalize()
    if val_ratio > 0:
        train_eps, val_eps = make_train_val_split(
            dataset.meta.total_episodes, val_ratio, min_val_episodes, seed=clip_seed
        )
        split_path = write_split_file(out_root, train_eps, val_eps)
        print(f"Wrote split -> {split_path}")
        print(f"  train episodes ({len(train_eps)}): {train_eps[:8]}{'...' if len(train_eps) > 8 else ''}")
        print(f"  val episodes   ({len(val_eps)}): {val_eps}")

    volume.commit()
    print(
        f"Wrote {dataset.meta.total_episodes} episodes, "
        f"{dataset.meta.total_frames} frames -> {out_root}"
    )
    return str(out_root)


def _resolve_output_dir(resume: bool, output_dir: str | None, profile: str) -> str:
    if output_dir is not None:
        return output_dir
    if resume:
        return CHECKPOINT_ROOT
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{CHECKPOINT_ROOT}_{profile}_{ts}"


def train_pi05(
    *,
    dataset_root: str,
    repo_id: str,
    profile: str,
    steps: int,
    batch_size: int,
    log_freq: int,
    save_freq: int,
    chunk_size: int,
    n_action_steps: int,
    train_expert_only: bool,
    freeze_vision_encoder: bool,
    use_lora: bool,
    peft_r: int,
    train_episodes: list[int] | None,
    optimizer_lr: float | None,
    optimizer_grad_clip_norm: float | None,
    scheduler_warmup_steps: int | None,
    scheduler_decay_steps: int | None,
    resume: bool,
    output_dir: str | None,
    wandb_project: str,
    wandb_run_name: str,
) -> str:
    if not os.getenv("WANDB_API_KEY"):
        raise RuntimeError(
            "WANDB_API_KEY is missing. Create Modal secret 'wandb' with your API key."
        )
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN missing — call _setup_huggingface_auth() first.")

    env = os.environ.copy()
    env["HF_TOKEN"] = hf_token
    env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    env["HF_HOME"] = HF_CACHE_DIR
    env["HUGGINGFACE_HUB_CACHE"] = f"{HF_CACHE_DIR}/hub"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["WANDB_PROJECT"] = wandb_project

    out_dir = _resolve_output_dir(resume=resume, output_dir=output_dir, profile=profile)
    print(f"profile={profile} output_dir={out_dir} resume={resume}")

    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root}",
        "--policy.type=pi05",
        "--policy.pretrained_path=lerobot/pi05_base",
        "--policy.push_to_hub=false",
        "--policy.compile_model=false",
        "--policy.gradient_checkpointing=true",
        "--policy.dtype=bfloat16",
        f"--policy.train_expert_only={'true' if train_expert_only else 'false'}",
        f"--policy.freeze_vision_encoder={'true' if freeze_vision_encoder else 'false'}",
        "--policy.empty_cameras=2",
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}",
        f"--output_dir={out_dir}",
        f"--job_name={wandb_run_name}",
        "--policy.device=cuda",
        "--wandb.enable=true",
        f"--wandb.project={wandb_project}",
        "--wandb.mode=online",
        f"--batch_size={batch_size}",
        "--num_workers=4",
        f"--steps={steps}",
        f"--log_freq={log_freq}",
        f"--save_freq={save_freq}",
        "--policy.normalization_mapping={\"ACTION\": \"MEAN_STD\", \"STATE\": \"MEAN_STD\", \"VISUAL\": \"IDENTITY\"}",
    ]

    if train_episodes is not None:
        cmd.append(f"--dataset.episodes={json.dumps(train_episodes)}")

    if optimizer_lr is not None:
        cmd.append(f"--policy.optimizer_lr={optimizer_lr}")

    if optimizer_grad_clip_norm is not None:
        cmd.append(f"--policy.optimizer_grad_clip_norm={optimizer_grad_clip_norm}")

    if scheduler_warmup_steps is not None:
        cmd.append(f"--policy.scheduler_warmup_steps={scheduler_warmup_steps}")

    if scheduler_decay_steps is not None:
        cmd.append(f"--policy.scheduler_decay_steps={scheduler_decay_steps}")

    if use_lora:
        cmd.extend(["--peft.method=LORA", f"--peft.r={peft_r}"])

    if resume:
        cmd.append("--resume=true")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    volume.commit()
    print(f"Training done. Checkpoints: {out_dir}")
    return out_dir


def resolve_pretrained_checkpoint(checkpoint_dir: str, checkpoint_step: int | None = None) -> Path:
    """Resolve .../checkpoints/<step>/pretrained_model (last/ is often empty on Modal volumes)."""
    ckpt_root = Path(checkpoint_dir) / "checkpoints"
    if checkpoint_step is not None:
        ckpt = ckpt_root / f"{checkpoint_step:06d}" / "pretrained_model"
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"No checkpoint at {ckpt}")

    for candidate in (
        ckpt_root / "last" / "pretrained_model",
    ):
        if candidate.exists():
            return candidate

    step_dirs = sorted(
        (p for p in ckpt_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )
    for step_dir in reversed(step_dirs):
        ckpt = step_dir / "pretrained_model"
        if (ckpt / "model.safetensors").exists():
            return ckpt

    raise FileNotFoundError(f"No pretrained_model under {ckpt_root}")


def eval_pi05_on_val_episodes(
    *,
    checkpoint_dir: str,
    dataset_root: str,
    repo_id: str,
    val_episodes: list[int],
    batch_size: int = 8,
    max_batches: int | None = 200,
    checkpoint_step: int | None = None,
) -> dict[str, float]:
    """Open-loop eval on held-out episodes: flow loss + unnormalized action MSE."""
    import torch
    from torch.utils.data import DataLoader

    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    ckpt = resolve_pretrained_checkpoint(checkpoint_dir, checkpoint_step=checkpoint_step)

    print(f"Evaluating checkpoint {ckpt} on val episodes {val_episodes}")
    policy = PI05Policy.from_pretrained(ckpt, device="cuda")
    policy.eval()

    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    print(f"delta_timestamps action steps: {len(delta_timestamps.get('action', [])) if delta_timestamps else 0}")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=dataset_root,
        episodes=val_episodes,
        delta_timestamps=delta_timestamps,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=ckpt,
        dataset_stats=dataset.meta.stats,
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    flow_loss_sum = 0.0
    action_mse_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            flow_loss_sum += loss.item()

            pred = policy.predict_action_chunk(batch)
            action_dim = policy.config.output_features[ACTION].shape[0]
            pred_first = pred[:, 0, :action_dim]
            gt_first = batch[ACTION][:, 0, :action_dim]
            pred_phys = postprocessor(pred_first)
            gt_phys = postprocessor(gt_first)
            action_mse_sum += torch.mean((pred_phys - gt_phys) ** 2).item()
            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Validation dataloader produced zero batches")

    metrics = {
        "val_flow_loss": flow_loss_sum / n_batches,
        "val_action_mse_step0": action_mse_sum / n_batches,
        "val_batches": float(n_batches),
        "val_episodes": float(len(val_episodes)),
    }
    print("Validation metrics:", metrics)
    return metrics


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def run_pi05_smoke_test(
    profile: ProfileName = "generalize",
    max_clips: int | None = None,
    frames_per_clip: int | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    log_freq: int | None = None,
    save_freq: int | None = None,
    skip_prepare: bool = False,
    resume: bool = False,
    output_dir: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> None:
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile {profile!r}. Choose from: {list(PROFILE_DEFAULTS)}")

    cfg = PROFILE_DEFAULTS[profile].copy()

    def _get(key: str, override):
        return override if override is not None else cfg[key]

    _setup_huggingface_auth()

    dataset_root = prepare_nvidia_lerobot_dataset(
        dataset_root=cfg["dataset_root"],
        repo_id=cfg["repo_id"],
        max_clips=_get("max_clips", max_clips),
        frames_per_clip=_get("frames_per_clip", frames_per_clip),
        skip_prepare=skip_prepare,
        ego_frame_actions=cfg["ego_frame_actions"],
        clip_seed=cfg["clip_seed"],
        val_ratio=cfg["val_ratio"],
        min_val_episodes=cfg["min_val_episodes"],
    )

    root_path = Path(dataset_root)
    train_episodes = cfg["train_episodes"]
    val_episodes: list[int] = []

    if cfg["val_ratio"] > 0:
        train_episodes, val_episodes = load_or_create_split(
            root_path,
            cfg["repo_id"],
            cfg["val_ratio"],
            cfg["min_val_episodes"],
            cfg["clip_seed"],
        )
        print(f"Training on {len(train_episodes)} episodes; val holdout: {len(val_episodes)} episodes")

    out_dir = train_pi05(
        dataset_root=dataset_root,
        repo_id=cfg["repo_id"],
        profile=profile,
        steps=_get("steps", steps),
        batch_size=_get("batch_size", batch_size),
        log_freq=_get("log_freq", log_freq),
        save_freq=_get("save_freq", save_freq),
        chunk_size=cfg["chunk_size"],
        n_action_steps=cfg["n_action_steps"],
        train_expert_only=cfg["train_expert_only"],
        freeze_vision_encoder=cfg["freeze_vision_encoder"],
        use_lora=cfg["use_lora"],
        peft_r=cfg["peft_r"],
        train_episodes=train_episodes,
        optimizer_lr=cfg["optimizer_lr"],
        optimizer_grad_clip_norm=cfg.get("optimizer_grad_clip_norm"),
        scheduler_warmup_steps=cfg["scheduler_warmup_steps"],
        scheduler_decay_steps=cfg["scheduler_decay_steps"],
        resume=resume,
        output_dir=output_dir,
        wandb_project=_get("wandb_project", wandb_project),
        wandb_run_name=_get("wandb_run_name", wandb_run_name),
    )

    if cfg["run_val_eval"] and val_episodes:
        metrics = eval_pi05_on_val_episodes(
            checkpoint_dir=out_dir,
            dataset_root=dataset_root,
            repo_id=cfg["repo_id"],
            val_episodes=val_episodes,
            batch_size=_get("batch_size", batch_size),
        )
        metrics_path = Path(out_dir) / "val_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Wrote {metrics_path}")

        try:
            import wandb

            wandb.init(
                project=_get("wandb_project", wandb_project),
                name=f"{_get('wandb_run_name', wandb_run_name)}_val",
                job_type="eval",
            )
            wandb.log({k: v for k, v in metrics.items()})
            wandb.finish()
        except Exception as exc:
            print(f"W&B val log skipped: {exc}")

        volume.commit()


run_pi05_lora_smoke_test = run_pi05_smoke_test


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
)
def run_pi05_eval(
    output_dir: str,
    profile: ProfileName = "generalize",
    batch_size: int | None = None,
    max_batches: int | None = 200,
    skip_prepare: bool = True,
    checkpoint_step: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
) -> dict[str, float]:
    """Open-loop val eval only (no training). Requires an existing checkpoint under output_dir."""
    if profile not in PROFILE_DEFAULTS:
        raise ValueError(f"Unknown profile {profile!r}. Choose from: {list(PROFILE_DEFAULTS)}")

    cfg = PROFILE_DEFAULTS[profile].copy()
    if not cfg["run_val_eval"] or cfg["val_ratio"] <= 0:
        raise ValueError(
            f"Profile {profile!r} has no val split (run_val_eval={cfg['run_val_eval']}, "
            f"val_ratio={cfg['val_ratio']}). Use profile='generalize'."
        )

    _setup_huggingface_auth()

    dataset_root = prepare_nvidia_lerobot_dataset(
        dataset_root=cfg["dataset_root"],
        repo_id=cfg["repo_id"],
        max_clips=cfg["max_clips"],
        frames_per_clip=cfg["frames_per_clip"],
        skip_prepare=skip_prepare,
        ego_frame_actions=cfg["ego_frame_actions"],
        clip_seed=cfg["clip_seed"],
        val_ratio=cfg["val_ratio"],
        min_val_episodes=cfg["min_val_episodes"],
    )

    _, val_episodes = load_or_create_split(
        Path(dataset_root),
        cfg["repo_id"],
        cfg["val_ratio"],
        cfg["min_val_episodes"],
        cfg["clip_seed"],
    )
    print(f"Val holdout: {len(val_episodes)} episodes -> {val_episodes}")

    metrics = eval_pi05_on_val_episodes(
        checkpoint_dir=output_dir,
        dataset_root=dataset_root,
        repo_id=cfg["repo_id"],
        val_episodes=val_episodes,
        batch_size=batch_size if batch_size is not None else cfg["batch_size"],
        max_batches=max_batches,
        checkpoint_step=checkpoint_step,
    )

    metrics_path = Path(output_dir) / "val_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")

    project = wandb_project or cfg["wandb_project"]
    run_name = wandb_run_name or cfg["wandb_run_name"]
    try:
        import wandb

        wandb.init(project=project, name=f"{run_name}_val", job_type="eval")
        wandb.log({k: v for k, v in metrics.items()})
        wandb.finish()
    except Exception as exc:
        print(f"W&B val log skipped: {exc}")

    volume.commit()
    return metrics


@app.local_entrypoint()
def main(
    profile: ProfileName = "generalize",
    max_clips: int | None = None,
    frames_per_clip: int | None = None,
    steps: int | None = None,
    batch_size: int | None = None,
    skip_prepare: bool = False,
    resume: bool = False,
    output_dir: str | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    call = run_pi05_smoke_test.spawn(
        profile=profile,
        max_clips=max_clips,
        frames_per_clip=frames_per_clip,
        steps=steps,
        batch_size=batch_size,
        skip_prepare=skip_prepare,
        resume=resume,
        output_dir=output_dir,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    print(f"Spawned profile={profile!r} on Modal. call_id={call.object_id}")
    print("Dashboard: https://modal.com/apps/fbarbosa/main")


@app.local_entrypoint()
def eval_main(
    output_dir: str,
    profile: ProfileName = "generalize",
    batch_size: int | None = None,
    max_batches: int | None = 200,
    skip_prepare: bool = True,
    checkpoint_step: int | None = None,
    wandb_project: str | None = None,
    wandb_run_name: str | None = None,
):
    """Eval-only: pass the training run directory on the Modal volume (e.g. /vol/runs/pi05_nvidia_smoke_generalize_...)."""
    metrics = run_pi05_eval.remote(
        output_dir=output_dir,
        profile=profile,
        batch_size=batch_size,
        max_batches=max_batches,
        skip_prepare=skip_prepare,
        checkpoint_step=checkpoint_step,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )
    print("Validation metrics:", metrics)


DEFAULT_INFER_CHECKPOINT_RUN = (
    "/vol/runs/pi05_nvidia_smoke_generalize_20260519_212805"
)
INFER_TASK = "Drive safely following the road."


def _classify_pedal(dx_forward: float) -> tuple[str, tuple[int, int, int]]:
    """Map forward ego delta (m per 0.1s @ 10Hz) to a driving pedal label and BGR color."""
    if dx_forward > 0.05:
        return "GAS", (0, 210, 0)
    if dx_forward > 0.012:
        return "CRUISE", (0, 220, 180)
    if dx_forward >= -0.012:
        return "STOP", (0, 160, 255)
    if dx_forward >= -0.05:
        return "BRAKE", (0, 70, 255)
    return "REVERSE", (0, 0, 220)


def _draw_road_path_overlay(
    img,
    dx_forward: float,
    steer_angle_rad: float,
    color_bgr: tuple[int, int, int],
    *,
    alpha: float = 0.38,
    yaw_rate_rad: float = 0.0,
) -> tuple[int, int]:
    """Perspective path on road: steer from heading/yaw-rate, depth from forward dx."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    # ego-frame dy is ~0 in labels; steer from heading + turn rate (visible on curves).
    # Negated for front-camera image coords (+x right): left turn → path aims left on screen.
    steer_px = -(np.sin(steer_angle_rad) * 95.0 + yaw_rate_rad * 900.0)
    vp_x = int(w * 0.5 + np.clip(steer_px, -w * 0.40, w * 0.40))
    vp_y = int(h * 0.30)
    base_y = h - 6
    base_half = int(70 + min(abs(steer_px) * 0.35, 50.0))

    forward_px = float(np.clip(dx_forward * 520.0, -h * 0.12, h * 0.50))
    target_x = int(np.clip(vp_x - np.sin(steer_angle_rad) * 35.0, 36, w - 36))
    target_y = int(np.clip(vp_y + h * 0.28 - forward_px, vp_y + 24, base_y - 28))

    overlay = img.copy()
    lane = np.array(
        [
            [w // 2 - base_half, base_y],
            [w // 2 + base_half, base_y],
            [target_x + 32, target_y],
            [target_x - 32, target_y],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(overlay, [lane], color_bgr)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    cv2.polylines(img, [lane], isClosed=True, color=color_bgr, thickness=2)
    cv2.arrowedLine(
        img, (w // 2, base_y), (target_x, target_y), color_bgr, 4, tipLength=0.18, line_type=cv2.LINE_AA
    )
    return target_x, target_y


def _draw_pedal_hud(
    img,
    *,
    pred_label: str,
    gt_label: str,
    pred_color: tuple[int, int, int],
    gt_color: tuple[int, int, int],
    pred_dx: float,
    gt_dx: float,
) -> None:
    import cv2

    h, w = img.shape[:2]
    box_w, box_h = 200, 88
    x0 = w - box_w - 12
    y0 = 12
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (30, 30, 30), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (180, 180, 180), 2)
    cv2.putText(img, "PRED", (x0 + 10, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, pred_label, (x0 + 70, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2)
    cv2.putText(img, f"{pred_dx:+.3f} m/step", (x0 + 10, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, pred_color, 1)
    cv2.putText(img, "GT", (x0 + 10, y0 + 74), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(img, gt_label, (x0 + 70, y0 + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gt_color, 2)

    # Large bottom-center pedal badge (prediction)
    badge_scale = 1.1 if pred_label == "GAS" else 0.95
    cv2.putText(
        img,
        pred_label,
        (w // 2 - 60, h - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        badge_scale,
        pred_color,
        3,
        cv2.LINE_AA,
    )


def _infer_single_episode(
    *,
    policy,
    preprocessor,
    postprocessor,
    dataset_root: str,
    repo_id: str,
    episode: int,
    split_tag: str,
    out_dir: Path,
    ckpt: Path,
    action_dim: int,
    max_frames: int | None,
    frame_stride: int,
) -> dict:
    """Render one episode overlay video; policy already loaded."""
    import cv2
    import mediapy as media
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import ACTION
    from torch.utils.data import DataLoader

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[episode])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    video_path = out_dir / f"{split_tag}_episode_{episode:03d}.mp4"
    json_path = out_dir / f"{split_tag}_episode_{episode:03d}.jsonl"

    frames_out: list = []
    records: list[dict] = []
    prev_yaw: float | None = None

    with torch.inference_mode():
        for frame_idx, batch in enumerate(loader):
            if frame_idx % frame_stride != 0:
                continue
            if max_frames is not None and len(frames_out) >= max_frames:
                break

            batch = preprocessor(batch)
            pred = policy.predict_action_chunk(batch)
            pred_first = pred[:, 0, :action_dim]

            gt_action = batch[ACTION]
            if gt_action.dim() == 3:
                gt_first = gt_action[:, 0, :action_dim]
            else:
                gt_first = gt_action[:, :action_dim]

            pred_phys = postprocessor(pred_first).cpu().numpy()[0]
            gt_phys = postprocessor(gt_first).cpu().numpy()[0]

            raw = dataset[frame_idx]
            state = raw["observation.state"]
            if hasattr(state, "cpu"):
                state = state.cpu().numpy().astype(float)
            else:
                state = np.asarray(state, dtype=float)
            yaw = float(state[6])

            if prev_yaw is None:
                yaw_rate = 0.0
            else:
                yaw_rate = float(np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw)))
            prev_yaw = yaw

            gt_heading = _motion_heading_rad(float(gt_phys[0]), float(gt_phys[1]), yaw)
            pred_heading = _motion_heading_rad(float(pred_phys[0]), float(pred_phys[1]), yaw)

            img_key = "observation.images.base_0_rgb"
            img = batch[img_key][0]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            arr = img.cpu().numpy()
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            pred_str = f"pred d=({pred_phys[0]:+.3f},{pred_phys[1]:+.3f},{pred_phys[2]:+.3f})"
            gt_str = f"gt   d=({gt_phys[0]:+.3f},{gt_phys[1]:+.3f},{gt_phys[2]:+.3f})"
            err = float(np.mean((pred_phys - gt_phys) ** 2))

            cv2.putText(
                arr, f"{split_tag} ep {episode} frame {frame_idx}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
            )
            cv2.putText(arr, pred_str, (8, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.putText(arr, gt_str, (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            cv2.putText(arr, f"mse={err:.4f}", (8, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            pred_dx, pred_dy = float(pred_phys[0]), float(pred_phys[1])
            gt_dx, gt_dy = float(gt_phys[0]), float(gt_phys[1])
            pred_pedal, pred_pedal_color = _classify_pedal(pred_dx)
            gt_pedal, gt_pedal_color = _classify_pedal(gt_dx)

            _draw_road_path_overlay(
                arr, gt_dx, gt_heading, (0, 140, 255), alpha=0.28, yaw_rate_rad=yaw_rate
            )
            _draw_road_path_overlay(
                arr, pred_dx, pred_heading, (255, 200, 0), alpha=0.40, yaw_rate_rad=yaw_rate
            )
            _draw_pedal_hud(
                arr,
                pred_label=pred_pedal,
                gt_label=gt_pedal,
                pred_color=pred_pedal_color,
                gt_color=gt_pedal_color,
                pred_dx=pred_dx,
                gt_dx=gt_dx,
            )

            h, w = arr.shape[:2]
            cv2.putText(
                arr,
                f"yaw={np.degrees(yaw):+.0f} dyaw={np.degrees(yaw_rate):+.1f}/step",
                (8, 124),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (180, 255, 180),
                1,
            )
            cv2.putText(
                arr,
                "road steer: heading (ego dy~0 in labels) | pedal: forward dx",
                (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (220, 220, 220),
                1,
            )

            frames_out.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            records.append(
                {
                    "episode": episode,
                    "split": split_tag,
                    "frame": frame_idx,
                    "yaw_rad": yaw,
                    "yaw_rate_rad": yaw_rate,
                    "gt_heading_rad": gt_heading,
                    "pred_heading_rad": pred_heading,
                    "pred_delta_ego": pred_phys.tolist(),
                    "gt_delta_ego": gt_phys.tolist(),
                    "pred_pedal": pred_pedal,
                    "gt_pedal": gt_pedal,
                    "mse": err,
                }
            )

    if not frames_out:
        raise RuntimeError(f"No frames processed for episode {episode}")

    media.write_video(str(video_path), frames_out, fps=10)
    with json_path.open("w") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")

    mse_mean = float(np.mean([r["mse"] for r in records]))
    return {
        "episode": episode,
        "split": split_tag,
        "checkpoint": str(ckpt),
        "video_path": str(video_path),
        "predictions_path": str(json_path),
        "num_frames": len(frames_out),
        "mean_step0_mse": mse_mean,
        "task": INFER_TASK,
    }


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
)
def run_clip_inference(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int | None = 12000,
    dataset_root: str = DATASET_ROOT_GEN,
    repo_id: str = "local/nvidia_av_gen",
    val_episode: int | None = None,
    clip_seed: int = 42,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> dict:
    """Single val (or chosen) episode → overlay video."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    _setup_huggingface_auth()

    root_path = Path(dataset_root)
    _, val_episodes = load_or_create_split(
        root_path, repo_id, val_ratio=0.2, min_val_episodes=8, seed=42
    )
    rng = random.Random(clip_seed)
    episode = val_episode if val_episode is not None else rng.choice(val_episodes)
    print(f"Val episode {episode} (pool size {len(val_episodes)})")

    ckpt = resolve_pretrained_checkpoint(checkpoint_run_dir, checkpoint_step=checkpoint_step)
    policy = PI05Policy.from_pretrained(ckpt, device="cuda")
    policy.eval()

    meta_ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        dataset_stats=meta_ds.meta.stats,
    )
    action_dim = policy.config.output_features[ACTION].shape[0]
    out_dir = Path(checkpoint_run_dir) / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _infer_single_episode(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset_root=dataset_root,
        repo_id=repo_id,
        episode=episode,
        split_tag="val",
        out_dir=out_dir,
        ckpt=ckpt,
        action_dim=action_dim,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    (out_dir / f"val_episode_{episode:03d}_summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", summary)
    volume.commit()
    return summary


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes={"/vol": volume},
    secrets=[
        modal.Secret.from_name("huggingface"),
    ],
)
def run_batch_inference(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int | None = 12000,
    dataset_root: str = DATASET_ROOT_GEN,
    repo_id: str = "local/nvidia_av_gen",
    include_val: bool = True,
    include_train: bool = False,
    max_episodes: int | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
) -> dict:
    """Render overlay videos for many episodes (loads policy once). Default: val holdout only (unseen)."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION

    _setup_huggingface_auth()

    root_path = Path(dataset_root)
    train_episodes, val_episodes = load_or_create_split(
        root_path, repo_id, val_ratio=0.2, min_val_episodes=8, seed=42
    )

    jobs: list[tuple[str, int]] = []
    if include_val:
        jobs.extend(("val", ep) for ep in sorted(val_episodes))
    if include_train:
        jobs.extend(("train", ep) for ep in sorted(train_episodes))
    if max_episodes is not None:
        jobs = jobs[:max_episodes]

    split_desc = []
    if include_val:
        split_desc.append(f"{len(val_episodes)} val (unseen)")
    if include_train:
        split_desc.append(f"{len(train_episodes)} train (seen)")
    print(f"Batch inference: {len(jobs)} episodes to render — pool has {', '.join(split_desc)}")

    ckpt = resolve_pretrained_checkpoint(checkpoint_run_dir, checkpoint_step=checkpoint_step)
    policy = PI05Policy.from_pretrained(ckpt, device="cuda")
    policy.eval()

    meta_ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        dataset_stats=meta_ds.meta.stats,
    )
    action_dim = policy.config.output_features[ACTION].shape[0]

    out_dir = Path(checkpoint_run_dir) / "inference" / ("batch_val" if not include_train else "batch")
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    for idx, (split_tag, episode) in enumerate(jobs):
        print(f"[{idx + 1}/{len(jobs)}] {split_tag} episode {episode}")
        try:
            summary = _infer_single_episode(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset_root=dataset_root,
                repo_id=repo_id,
                episode=episode,
                split_tag=split_tag,
                out_dir=out_dir,
                ckpt=ckpt,
                action_dim=action_dim,
                max_frames=max_frames,
                frame_stride=frame_stride,
            )
            manifest.append(summary)
        except Exception as exc:
            print(f"  FAILED episode {episode}: {exc}")
            manifest.append({"episode": episode, "split": split_tag, "error": str(exc)})

        if (idx + 1) % 3 == 0:
            volume.commit()

    mse_vals = [m["mean_step0_mse"] for m in manifest if "mean_step0_mse" in m]
    batch_summary = {
        "num_episodes": len(jobs),
        "num_ok": len(mse_vals),
        "num_failed": len(jobs) - len(mse_vals),
        "mean_mse_across_episodes": float(sum(mse_vals) / len(mse_vals)) if mse_vals else None,
        "output_dir": str(out_dir),
        "episodes": manifest,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(batch_summary, indent=2))
    print("Batch done:", batch_summary)
    volume.commit()
    return batch_summary


@app.local_entrypoint()
def infer_main(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    val_episode: int | None = None,
    clip_seed: int = 42,
    max_frames: int | None = None,
    frame_stride: int = 1,
):
    summary = run_clip_inference.remote(
        checkpoint_run_dir=checkpoint_run_dir,
        checkpoint_step=checkpoint_step,
        val_episode=val_episode,
        clip_seed=clip_seed,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    print(json.dumps(summary, indent=2))
    rel = Path(summary["video_path"]).relative_to("/vol")
    print(f"\nDownload:\n  modal volume get {VOLUME_NAME} {rel} ./")


@app.local_entrypoint()
def infer_batch_main(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    include_val: bool = True,
    include_train: bool = False,
    max_episodes: int | None = None,
    max_frames: int | None = None,
    frame_stride: int = 1,
):
    """Render videos for val holdout episodes (unseen). ~11 clips ≈ 30–45 min on H100."""
    call = run_batch_inference.spawn(
        checkpoint_run_dir=checkpoint_run_dir,
        checkpoint_step=checkpoint_step,
        include_val=include_val,
        include_train=include_train,
        max_episodes=max_episodes,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    print(f"Spawned batch inference. call_id={call.object_id}")
    print("Dashboard: https://modal.com/apps/fbarbosa/main")
    print(
        f"\nOutputs → {checkpoint_run_dir}/inference/batch/\n"
        f"  manifest.json + train_episode_*.mp4 + val_episode_*.mp4\n"
        f"\nDownload when done:\n"
        f"  modal volume get {VOLUME_NAME} "
        f"runs/pi05_nvidia_smoke_generalize_20260519_212805/inference/batch_val ./inference_batch_val"
    )


CADDY_DATASET_VOL = "/vol/caddy_lerobot"
CADDY_REPO_ID = "local/caddy_pi05"
_caddy_session_dir = os.environ.get("CADDY_SESSION_DIR", "").strip()
_caddy_image = (
    image.add_local_dir(_caddy_session_dir, remote_path="/root/caddy_session")
    if _caddy_session_dir
    else image
)


@app.function(
    image=_caddy_image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_caddy_on_modal(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    max_frames: int | None = 300,
    frame_stride: int = 2,
    max_duration_s: float | None = 60.0,
    full_video: bool = False,
    session_mount_path: str = "/root/caddy_session",
    dataset_root: str = CADDY_DATASET_VOL,
    skip_prepare: bool = False,
) -> dict:
    """Zero-shot NVIDIA π₀.5 on a Caddy session (set CADDY_SESSION_DIR when launching)."""
    import sys

    sys.path.insert(0, "/app/pi_05_drives")
    from infer_caddy_pi05 import run_caddy_zero_shot
    from prepare_caddy_lerobot import prepare_caddy_session

    if full_video:
        max_duration_s = None
        max_frames = None

    if not skip_prepare:
        if not Path(session_mount_path).is_dir():
            raise FileNotFoundError(
                f"Caddy session not at {session_mount_path}. "
                "Launch with: CADDY_SESSION_DIR=/path/to/Caddy-Training-Data-* "
                "modal run train_modal_pi05_lora_nvidia_driving.py::caddy_main"
            )
        prepare_caddy_session(
            session_mount_path,
            dataset_root,
            repo_id=CADDY_REPO_ID,
            max_duration_s=max_duration_s,
        )
        volume.commit()
    summary = run_caddy_zero_shot(
        dataset_root,
        checkpoint_run_dir,
        checkpoint_step=checkpoint_step,
        session_dir=session_mount_path if Path(session_mount_path).is_dir() else None,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )
    volume.commit()
    return summary


@app.local_entrypoint()
def caddy_main(
    checkpoint_run: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    max_frames: int | None = 300,
    frame_stride: int = 1,
    max_duration_s: float | None = 60.0,
    full_video: bool = False,
    skip_prepare: bool = False,
):
    summary = run_caddy_on_modal.remote(
        checkpoint_run_dir=checkpoint_run,
        checkpoint_step=checkpoint_step,
        max_frames=max_frames,
        frame_stride=frame_stride,
        max_duration_s=max_duration_s,
        full_video=full_video,
        skip_prepare=skip_prepare,
    )
    print(summary)
    print(
        f"\nDownload:\n"
        f"  modal volume get {VOLUME_NAME} "
        f"runs/pi05_nvidia_smoke_generalize_20260519_212805/inference/caddy_zero_shot ./caddy_zero_shot"
    )

