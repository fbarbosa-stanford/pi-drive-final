"""
Convert a Caddy training session (front video + ego.jsonl) into LeRobot v3 for π₀.5.

Uses alpamayo-frame pose from ego.jsonl (same frame as NVIDIA PhysicalAI AV labels).

  python prepare_caddy_lerobot.py \\
    --session /path/to/Caddy-Training-Data-2026-05-17_18-17-32 \\
    --out ./caddy_lerobot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_modal_pi05_lora_nvidia_driving import _resize_rgb, _world_delta_to_ego

TASK = "Drive safely following the road."
DEFAULT_CAMERA = "front_noaudio.mp4"


def _load_jsonl_rows(path: Path) -> list[dict]:
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


def _interp_scalar(rows: list[dict], t: float, key: str, default: float = 0.0) -> float:
    if not rows:
        return default
    if t <= float(rows[0]["rel_t"]):
        return float(rows[0].get(key, default))
    if t >= float(rows[-1]["rel_t"]):
        return float(rows[-1].get(key, default))
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        t0, t1 = float(a["rel_t"]), float(b["rel_t"])
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            va = float(a.get(key, default))
            vb = float(b.get(key, default))
            return (1 - u) * va + u * vb
    return default


def _ego_alp(row: dict) -> dict:
    """Normalized alpamayo pose block from one ego row."""
    alp = row.get("alpamayo") or {}
    return {
        "xyz_m": alp.get("xyz_m", [0.0, 0.0, 0.0]),
        "yaw_rad": float(alp.get("yaw_rad", row.get("yaw_rad", 0.0))),
        "speed_mps": float(alp.get("speed_mps", row.get("speed_mps", 0.0))),
    }


def _interp_ego(rows: list[dict], t: float) -> dict | None:
    """Linear interp of alpamayo xyz + yaw between ego rows."""
    if not rows:
        return None
    if t <= float(rows[0]["rel_t"]):
        return _ego_alp(rows[0])
    if t >= float(rows[-1]["rel_t"]):
        return _ego_alp(rows[-1])
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        t0, t1 = float(a["rel_t"]), float(b["rel_t"])
        if t0 <= t <= t1:
            u = (t - t0) / max(t1 - t0, 1e-9)
            alp_a = a.get("alpamayo") or {}
            alp_b = b.get("alpamayo") or {}
            xyz = [
                (1 - u) * alp_a["xyz_m"][k] + u * alp_b["xyz_m"][k]
                for k in range(3)
            ]
            yaw = (1 - u) * float(alp_a.get("yaw_rad", a.get("yaw_rad", 0))) + u * float(
                alp_b.get("yaw_rad", b.get("yaw_rad", 0))
            )
            speed = (1 - u) * float(alp_a.get("speed_mps", a.get("speed_mps", 0))) + u * float(
                alp_b.get("speed_mps", b.get("speed_mps", 0))
            )
            return {"xyz_m": xyz, "yaw_rad": yaw, "speed_mps": speed}
    return None


def prepare_caddy_session(
    session_dir: str | Path,
    out_root: str | Path,
    *,
    repo_id: str = "local/caddy_pi05",
    fps: int = 10,
    video_name: str = DEFAULT_CAMERA,
    max_duration_s: float | None = None,
) -> Path:
    """Build one-episode LeRobot dataset from a Caddy session directory."""
    import cv2
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    session = Path(session_dir).resolve()
    out_root = Path(out_root).resolve()
    video_path = session / video_name
    ego_path = session / "ego.jsonl"
    if not video_path.is_file():
        raise FileNotFoundError(f"Missing video: {video_path}")
    if not ego_path.is_file():
        raise FileNotFoundError(f"Missing ego.jsonl: {ego_path}")

    egos = _load_jsonl_rows(ego_path)
    if len(egos) < 4:
        raise ValueError(f"Too few ego rows in {ego_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = n_video / video_fps
    if max_duration_s is not None:
        duration_s = min(duration_s, max_duration_s)
    n_frames = int(duration_s * fps)
    print(f"Session {session.name}: video {video_fps:.1f} fps, {n_video} frames")
    print(f"  ego rows={len(egos)}, export {n_frames} frames @ {fps} Hz ({duration_s:.1f}s)")

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

    samples: list[dict] = []
    for i in range(n_frames):
        t = i / fps
        ego = _interp_ego(egos, t)
        if ego is None:
            continue
        pos = np.asarray(ego["xyz_m"], dtype=np.float32)
        yaw = float(ego["yaw_rad"])
        speed = float(ego["speed_mps"])

        video_idx = int(round(t * video_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, video_idx)
        ok, frame = cap.read()
        if not ok:
            break
        samples.append(
            {
                "t": t,
                "pos": pos,
                "yaw": yaw,
                "speed": speed,
                "image": _resize_rgb(frame),
            }
        )

    cap.release()
    if len(samples) < 2:
        raise RuntimeError("Not enough aligned frames")

    for i in range(len(samples) - 1):
        pos = samples[i]["pos"]
        nxt = samples[i + 1]["pos"]
        delta = (nxt - pos).astype(np.float32)
        vel = delta * fps
        yaw = float(samples[i]["yaw"])
        delta_xy_ego = _world_delta_to_ego(delta[:2], yaw)
        action = np.array([delta_xy_ego[0], delta_xy_ego[1], delta[2]], dtype=np.float32)
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
                "observation.images.base_0_rgb": samples[i]["image"],
                "task": TASK,
            }
        )
    dataset.save_episode()
    dataset.finalize()
    print(f"Wrote LeRobot dataset -> {out_root} ({len(samples) - 1} frames)")
    return out_root


def main() -> None:
    p = argparse.ArgumentParser(description="Caddy session → LeRobot v3 (π₀.5 schema)")
    p.add_argument("--session", type=Path, required=True, help="Caddy-Training-Data-* directory")
    p.add_argument("--out", type=Path, required=True, help="Output LeRobot root")
    p.add_argument("--repo-id", default="local/caddy_pi05")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--video", default=DEFAULT_CAMERA)
    p.add_argument("--max-duration-s", type=float, default=None)
    args = p.parse_args()
    prepare_caddy_session(
        args.session,
        args.out,
        repo_id=args.repo_id,
        fps=args.fps,
        video_name=args.video,
        max_duration_s=args.max_duration_s,
    )


if __name__ == "__main__":
    main()
