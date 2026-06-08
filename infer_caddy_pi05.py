"""
Zero-shot π₀.5 (NVIDIA-finetuned) on a Caddy LeRobot dataset + overlay video.

Prepare first:
  python prepare_caddy_lerobot.py --session /path/to/Caddy-Training-Data-* --out ./caddy_lerobot

Local (needs CUDA + lerobot + checkpoint):
  python infer_caddy_pi05.py --dataset-root ./caddy_lerobot --checkpoint-dir /path/to/pretrained_model

Modal (mounts session, prepares on GPU volume, runs infer):
  modal run infer_caddy_pi05.py --session-path /Users/.../Caddy-Training-Data-2026-05-17_18-17-32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import modal

from prepare_caddy_lerobot import prepare_caddy_session
from train_modal_pi05_lora_nvidia_driving import (
    APP_NAME,
    DEFAULT_INFER_CHECKPOINT_RUN,
    _classify_pedal,
    _draw_pedal_hud,
    _draw_road_path_overlay,
    _motion_heading_rad,
    _setup_huggingface_auth,
    image,
    resolve_pretrained_checkpoint,
    volume,
)

CADDY_DATASET_VOL = "/vol/caddy_lerobot"
CADDY_REPO_ID = "local/caddy_pi05"


def _load_control_rows(session_dir: Path) -> list[dict]:
    path = session_dir / "control.jsonl"
    if not path.is_file():
        return []
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if str(row.get("_schema", "")).startswith("caddy."):
                continue
            rows.append(row)
    rows.sort(key=lambda r: float(r["rel_t"]))
    return rows


def _interp_control(rows: list[dict], t: float) -> tuple[float, float, float]:
    """steer_deg, gas_frac, mph at rel_t."""
    if not rows:
        return 0.0, 0.0, 0.0
    if t <= float(rows[0]["rel_t"]):
        r = rows[0]
        return float(r.get("steer_deg", 0)), float(r.get("gas_frac", 0)), float(r.get("mph", 0))
    if t >= float(rows[-1]["rel_t"]):
        r = rows[-1]
        return float(r.get("steer_deg", 0)), float(r.get("gas_frac", 0)), float(r.get("mph", 0))
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        t0, t1 = float(a["rel_t"]), float(b["rel_t"])
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            steer = (1 - u) * float(a.get("steer_deg", 0)) + u * float(b.get("steer_deg", 0))
            gas = (1 - u) * float(a.get("gas_frac", 0)) + u * float(b.get("gas_frac", 0))
            mph = (1 - u) * float(a.get("mph", 0)) + u * float(b.get("mph", 0))
            return steer, gas, mph
    return 0.0, 0.0, 0.0


def run_caddy_zero_shot(
    dataset_root: str,
    checkpoint_run_dir: str,
    *,
    repo_id: str = CADDY_REPO_ID,
    checkpoint_step: int | None = 12000,
    checkpoint_path: str | Path | None = None,
    session_dir: str | None = None,
    max_frames: int | None = 300,
    frame_stride: int = 2,
    output_subdir: str = "caddy_zero_shot",
) -> dict:
    """Run π₀.5 on episode 0; optional control.jsonl overlay from raw session."""
    import cv2
    import mediapy as media
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.utils.constants import ACTION
    from torch.utils.data import DataLoader

    _setup_huggingface_auth()

    if checkpoint_path is not None:
        ckpt = Path(checkpoint_path)
    else:
        ckpt = resolve_pretrained_checkpoint(checkpoint_run_dir, checkpoint_step=checkpoint_step)
    print(f"Checkpoint: {ckpt}")
    policy = PI05Policy.from_pretrained(ckpt, device="cuda")
    policy.eval()

    meta_ds = LeRobotDataset(repo_id=repo_id, root=dataset_root)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(ckpt),
        dataset_stats=meta_ds.meta.stats,
    )
    action_dim = policy.config.output_features[ACTION].shape[0]

    out_dir = Path(checkpoint_run_dir) / "inference" / output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    controls = _load_control_rows(Path(session_dir)) if session_dir else []
    fps = meta_ds.meta.fps

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root, episodes=[0])
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    video_path = out_dir / "caddy_ep000_zero_shot.mp4"
    json_path = out_dir / "caddy_ep000_zero_shot.jsonl"

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
            gt_first = gt_action[:, 0, :action_dim] if gt_action.dim() == 3 else gt_action[:, :action_dim]

            pred_phys = postprocessor(pred_first).cpu().numpy()[0]
            gt_phys = postprocessor(gt_first).cpu().numpy()[0]

            raw = dataset[frame_idx]
            state = raw["observation.state"]
            state = np.asarray(
                state.cpu().numpy() if hasattr(state, "cpu") else state, dtype=float
            )
            yaw = float(state[6])
            yaw_rate = 0.0 if prev_yaw is None else float(
                np.arctan2(np.sin(yaw - prev_yaw), np.cos(yaw - prev_yaw))
            )
            prev_yaw = yaw

            t = frame_idx / fps
            steer_deg, gas_frac, mph = _interp_control(controls, t)

            pred_heading = _motion_heading_rad(float(pred_phys[0]), float(pred_phys[1]), yaw)
            gt_heading = _motion_heading_rad(float(gt_phys[0]), float(gt_phys[1]), yaw)

            img = batch["observation.images.base_0_rgb"][0]
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0)
            arr = img.cpu().numpy()
            if arr.min() < 0:
                arr = (arr + 1.0) / 2.0
            arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

            pred_dx = float(pred_phys[0])
            gt_dx = float(gt_phys[0])
            pred_pedal, pred_color = _classify_pedal(pred_dx)
            gt_pedal, gt_color = _classify_pedal(gt_dx)

            _draw_road_path_overlay(arr, gt_dx, gt_heading, (0, 140, 255), alpha=0.28, yaw_rate_rad=yaw_rate)
            _draw_road_path_overlay(arr, pred_dx, pred_heading, (255, 200, 0), alpha=0.40, yaw_rate_rad=yaw_rate)
            _draw_pedal_hud(
                arr,
                pred_label=pred_pedal,
                gt_label=gt_pedal,
                pred_color=pred_color,
                gt_color=gt_color,
                pred_dx=pred_dx,
                gt_dx=gt_dx,
            )

            h, w = arr.shape[:2]
            cv2.putText(
                arr,
                f"CADDY zero-shot (NVIDIA pi05)  t={t:.1f}s",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                arr,
                f"control steer={steer_deg:+.0f}deg  gas={gas_frac:.2f}  {mph:.1f}mph",
                (8, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (180, 255, 255),
                1,
            )
            cv2.putText(
                arr,
                f"pred d=({pred_phys[0]:+.3f},{pred_phys[1]:+.3f},{pred_phys[2]:+.3f})",
                (8, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 220, 255),
                1,
            )
            cv2.putText(
                arr,
                f"gt   d=({gt_phys[0]:+.3f},{gt_phys[1]:+.3f},{gt_phys[2]:+.3f})",
                (8, 98),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 200, 0),
                1,
            )
            cv2.putText(
                arr,
                "yellow=pred motion  blue=logged motion  cyan=actual steer/gas",
                (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (200, 200, 200),
                1,
            )

            frames_out.append(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
            records.append(
                {
                    "frame": frame_idx,
                    "t_s": t,
                    "pred_delta_ego": pred_phys.tolist(),
                    "gt_delta_ego": gt_phys.tolist(),
                    "steer_deg": steer_deg,
                    "gas_frac": gas_frac,
                    "mph": mph,
                    "pred_pedal": pred_pedal,
                    "gt_pedal": gt_pedal,
                    "mse": float(np.mean((pred_phys - gt_phys) ** 2)),
                }
            )

    media.write_video(str(video_path), frames_out, fps=fps)
    with json_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    mse_mean = float(np.mean([r["mse"] for r in records])) if records else float("nan")
    summary = {
        "video_path": str(video_path),
        "predictions_path": str(json_path),
        "num_frames": len(records),
        "mean_mse": mse_mean,
        "checkpoint": str(ckpt),
        "dataset_root": dataset_root,
        "note": "Zero-shot NVIDIA-highway pi05 on golf-cart; high MSE expected.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", summary)
    return summary


app_modal = modal.App(f"{APP_NAME}-caddy")

# Set CADDY_SESSION_DIR when building/running so the session is baked into the image:
#   CADDY_SESSION_DIR=/path/to/Caddy-Training-Data-* modal run infer_caddy_pi05.py
import os as _os

_caddy_session_dir = _os.environ.get("CADDY_SESSION_DIR", "").strip()
_infer_image = (
    image.add_local_dir(_caddy_session_dir, remote_path="/root/caddy_session")
    if _caddy_session_dir
    else image
)


@app_modal.function(
    image=_infer_image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface")],
)
def run_caddy_on_modal(
    checkpoint_run_dir: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    max_frames: int = 300,
    frame_stride: int = 2,
    max_duration_s: float = 60.0,
    session_mount_path: str = "/root/caddy_session",
    dataset_root: str = CADDY_DATASET_VOL,
    skip_prepare: bool = False,
) -> dict:
    if not skip_prepare:
        if not Path(session_mount_path).is_dir():
            raise FileNotFoundError(
                f"Caddy session not at {session_mount_path}. "
                "Re-run with CADDY_SESSION_DIR=/path/to/session modal run infer_caddy_pi05.py"
            )
        prepare_caddy_session(
            session_mount_path,
            dataset_root,
            repo_id=CADDY_REPO_ID,
            max_duration_s=max_duration_s,
        )
        volume.commit()
    return run_caddy_zero_shot(
        dataset_root,
        checkpoint_run_dir,
        checkpoint_step=checkpoint_step,
        session_dir=session_mount_path if Path(session_mount_path).is_dir() else None,
        max_frames=max_frames,
        frame_stride=frame_stride,
    )


@app_modal.local_entrypoint()
def main(
    checkpoint_run: str = DEFAULT_INFER_CHECKPOINT_RUN,
    checkpoint_step: int = 12000,
    max_frames: int = 300,
    frame_stride: int = 2,
    max_duration_s: float = 60.0,
    skip_prepare: bool = False,
):
    summary = run_caddy_on_modal.remote(
        checkpoint_run_dir=checkpoint_run,
        checkpoint_step=checkpoint_step,
        max_frames=max_frames,
        frame_stride=frame_stride,
        max_duration_s=max_duration_s,
        skip_prepare=skip_prepare,
    )
    print(summary)


def cli_local() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Path to checkpoints/NNNNNN/pretrained_model",
    )
    p.add_argument("--session", type=Path, default=None, help="For control.jsonl overlay")
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--frame-stride", type=int, default=2)
    args = p.parse_args()
    run_caddy_zero_shot(
        str(args.dataset_root.resolve()),
        str(args.checkpoint_dir.resolve().parents[2]),
        session_dir=str(args.session) if args.session else None,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
        checkpoint_path=args.checkpoint_dir.resolve(),
        checkpoint_step=None,
    )


if __name__ == "__main__":
    cli_local()
