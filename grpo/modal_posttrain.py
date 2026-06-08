"""
Flow-GRPO post-training on Mark's BC checkpoint (openpi / JAX).

Checkpoint: https://huggingface.co/markmusic/pi05-driving-bc-v2-checkpoint
Dataset:    https://huggingface.co/datasets/markmusic/pi05-physical-av-bc

Prerequisites (Modal secrets): huggingface, wandb (optional)

  # Download BC checkpoint to volume (~7GB)
  modal run grpo/modal_posttrain.py::download_bc_checkpoint

  # One GPU step: load policy, sample group, composite rank, flow losses for objective
  modal run grpo/modal_posttrain.py::smoke_grpo_step --group-size 8

  # GRPO train loop (ranking + flow_losses; policy loss = grpo/objective.py — you implement)
  modal run --detach grpo/modal_posttrain.py::train_grpo --num-steps 100

  modal run grpo/modal_posttrain.py::compare_fifty --num-samples 50
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Modal mounts this file to /root/; package code lives under /app/pi_05_drives
_PKG_ROOT = "/app/pi_05_drives"
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import modal

HF_BC_CHECKPOINT_REPO = "markmusic/pi05-driving-bc-v2-checkpoint"
HF_BC_DATASET_REPO = "markmusic/pi05-physical-av-bc"
HF_BC_EVAL_DATASET_REPO = "markmusic/pi05-physical-av-bc-eval"
OPENPI_CONFIG_NAME = "pi05_driving"

APP_NAME = "pi05-grpo-posttrain"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"
_PKG_DIR = Path(__file__).resolve().parents[1]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    .pip_install("uv", "huggingface_hub", "numpy", "matplotlib", "pillow", "pyarrow",
                 "imageio", "imageio-ffmpeg")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
    )
    .env({"PYTHONPATH": "/app/pi_05_drives:/opt/openpi/src"})
    .add_local_dir(
        _PKG_DIR, remote_path="/app/pi_05_drives", copy=True,
        # Only ship code — the project dir also holds ~9GB of checkpoints/outputs that
        # otherwise stall the upload ("Fetching files ...") and never start the job.
        ignore=[
            "checkpoints", "downloaded_ckpt", "caddy_lerobot", "caddy_zero_shot_full",
            "caddy_zero_shot_out", "inference_batch_val", "inference_output",
            "viz", "viz_images", "viz_local", ".git", "__pycache__",
            "*.mp4", "*.png", "*.pyc", ".DS_Store", "**/.DS_Store",
        ],
    )
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}
app = modal.App(APP_NAME)

CKPT_LOCAL = f"{CACHE_DIR}/checkpoints/bc_hf/{HF_BC_CHECKPOINT_REPO.replace('/', '--')}"


def ensure_bc_checkpoint(
    cache_dir: str = CACHE_DIR,
    *,
    repo_id: str = HF_BC_CHECKPOINT_REPO,
    force: bool = False,
) -> str:
    """Download BC checkpoint to volume if missing."""
    from huggingface_hub import snapshot_download

    local = f"{cache_dir}/checkpoints/bc_hf/{repo_id.replace('/', '--')}"
    if os.path.isdir(local) and not force:
        print(f"Checkpoint already at {local}")
        return local
    path = snapshot_download(repo_id=repo_id, repo_type="model", local_dir=local)
    print(f"Downloaded checkpoint to {path}")
    return str(local)


def ensure_bc_dataset(
    cache_dir: str = CACHE_DIR,
    force: bool = False,
) -> tuple[str, str, bool]:
    """Download LeRobot dataset. Returns (local_path, repo_id, use_holdout_split)."""
    from grpo.dataset_paths import resolve_bc_dataset

    return resolve_bc_dataset(cache_dir, force=force)


def _setup_openpi_env(ckpt_params_dir: str) -> dict[str, str]:
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    params_dir = ckpt_params_dir if ckpt_params_dir.rstrip("/").endswith("params") else f"{ckpt_params_dir}/params"
    return {
        **os.environ,
        "PI05_BC_CHECKPOINT_PARAMS": params_dir,
        "HF_HOME": f"{CACHE_DIR}/hf",
        "OPENPI_DIR": OPENPI_DIR,
        "HF_HUB_DISABLE_XET": "1",  # avoid hf_xet 'request_headers' download crash
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # on-demand alloc; frees GPU promptly (fixes progressive OOM)
        "PYTHONUNBUFFERED": "1",
    }


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_bc_checkpoint(
    repo_id: str = HF_BC_CHECKPOINT_REPO,
    force: bool = False,
) -> str:
    """Download orbax params from HuggingFace to Modal volume."""
    path = ensure_bc_checkpoint(CACHE_DIR, repo_id=repo_id, force=force)
    cache_volume.commit()
    return path


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def introspect_checkpoint(repo_id: str = HF_BC_CHECKPOINT_REPO) -> dict:
    """CPU-only: read orbax param shapes (no weight load, no GPU) to recover geometry.

    Reports action_in_proj / action_out_proj / state_proj shapes -> action_dim and
    state_dim, plus dataset action shape -> action_horizon = action_flat / action_dim.
    """
    from openpi_patches.patch_openpi import prepend_openpi_venv

    ckpt = ensure_bc_checkpoint(CACHE_DIR, repo_id=repo_id)
    prepend_openpi_venv(OPENPI_DIR)
    import orbax.checkpoint as ocp

    # Mark's HF checkpoint is a raw orbax pytree (manifest.ocdbt at root).
    root = Path(ckpt)
    params_root = root / "params" if (root / "params" / "manifest.ocdbt").exists() else root
    meta = ocp.PyTreeCheckpointer().metadata(params_root)

    def _flatten(tree, prefix=""):
        out = {}
        if hasattr(tree, "items"):
            for k, v in tree.items():
                out.update(_flatten(v, f"{prefix}/{k}" if prefix else str(k)))
        else:
            shape = getattr(tree, "shape", None)
            out[prefix] = list(shape) if shape is not None else str(type(tree).__name__)
        return out

    flat = _flatten(meta)
    interesting = {
        k: v for k, v in flat.items()
        if any(t in k for t in ("action_in_proj", "action_out_proj", "state_proj", "action_time", "time_mlp"))
    }

    def _find_dim(substr: str, axis: int):
        for k, v in flat.items():
            if substr in k and isinstance(v, list) and len(v) >= abs(axis + 1):
                return k, v
        return None, None

    ain_k, ain_v = _find_dim("action_in_proj", 0)      # (action_dim, width)
    sp_k, sp_v = _find_dim("state_proj", 0)            # (state_dim, width) in pi05
    action_dim = ain_v[0] if ain_v else None

    # Dataset action shape -> horizon.
    train_local = f"{CACHE_DIR}/hf/lerobot/{HF_BC_DATASET_REPO}"
    info_action = None
    info_path = Path(train_local) / "meta" / "info.json"
    if info_path.exists():
        feats = json.loads(info_path.read_text()).get("features", {})
        info_action = feats.get("action", {}).get("shape")
    action_flat = info_action[0] if info_action else None
    horizon = (action_flat // action_dim) if (action_flat and action_dim) else None

    report = {
        "repo_id": repo_id,
        "n_params": len(flat),
        "action_in_proj": {ain_k: ain_v},
        "state_proj": {sp_k: sp_v},
        "interesting_shapes": interesting,
        "inferred_action_dim": action_dim,
        "dataset_action_shape": info_action,
        "inferred_action_horizon": horizon,
    }
    print(json.dumps(report, indent=2))
    return report


def _project_path(xy, W, H, f, cam_h, horizon_frac, max_dist=40.0):
    """Approx pinhole projection of BEV ground points (X fwd, Y left, meters) -> image px.

    No dataset intrinsics exist, so f / cam_h / horizon are tunable. We only project the
    near lookahead (X in [1, max_dist]) so the path lies on the visible road instead of
    compressing onto the horizon (these are high-speed clips with ~200m trajectories).
    Points kept in original order so the polyline draws cleanly.
    """
    import numpy as np

    xy = np.asarray(xy, np.float32)
    X, Y = xy[:, 0], xy[:, 1]
    cx, cy = W / 2.0, H * horizon_frac
    out = []
    for i in range(len(X)):
        if not (1.0 < X[i] < max_dist):
            continue
        u = cx - f * Y[i] / X[i]
        v = cy + f * cam_h / X[i]
        if 0 <= u < W and cy <= v < H * 0.97:
            out.append((u, v))
    return np.asarray(out, np.float32) if out else np.zeros((0, 2), np.float32)


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def viz_overlay(
    num_samples: int = 40,
    seed: int = 123,
    grpo_ckpt_dir: str = f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/grpo-50ep-fixed/50",
    dpo_ckpt_dir: str = f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/dpo-50ep-fixed/50",
    focal: float = 480.0,
    cam_h: float = 1.5,
    horizon_frac: float = 0.46,
    fps: int = 4,
    tag: str = "overlay",
) -> dict:
    """Project GT/BC/GRPO/DPO predicted paths ONTO each frame's camera image and encode
    the overlaid frames into a slideshow mp4 (+ per-frame PNGs). No real video clips exist
    in this dataset, so this flips through N independent samples with their overlays.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    bc_ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(bc_ckpt)

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    from grpo.eval_indices import pick_eval_holdout_indices
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout
    from grpo.parquet_frames import FrameSource
    from rewards.action_space import actions_to_xyz
    from rewards.flat_actions import unflatten_actions

    config = _config.get_config(OPENPI_CONFIG_NAME)
    dc = config.data.create(config.assets_dirs, config.model)
    frames = FrameSource(dataset_root, dc)
    ah, ad = config.model.action_horizon, config.model.action_dim

    _ensure_checkpoint_layout(bc_ckpt, OPENPI_DIR)

    def _maybe_load(path):
        if os.path.isdir(f"{path}/params"):
            _ensure_checkpoint_layout(path, OPENPI_DIR)
            return policy_config.create_trained_policy(config, path)
        return None

    bc = policy_config.create_trained_policy(config, bc_ckpt)
    grpo = _maybe_load(grpo_ckpt_dir)
    dpo = _maybe_load(dpo_ckpt_dir)
    print(f"[overlay] BC + grpo={grpo is not None} dpo={dpo is not None}", flush=True)

    indices = pick_eval_holdout_indices(len(frames), num_samples, seed=seed)
    outdir = Path(CACHE_DIR) / "viz" / f"{tag}_{num_samples}"
    outdir.mkdir(parents=True, exist_ok=True)

    series = [("GT", "lime", "gt"), ("BC", "deepskyblue", bc), ("GRPO", "red", grpo), ("DPO", "orange", dpo)]
    for k, idx in enumerate(indices):
        r = frames.row(idx)
        img = np.asarray(r["observation.images.front"]).astype(np.uint8)
        H, W = img.shape[:2]
        obs = {"observation/image": img, "observation/state": r["observation.state"], "prompt": r["prompt"]}
        speed = float(r["observation.state"][0]) if r["observation.state"].size else 5.0
        noise = np.random.default_rng(seed + idx).standard_normal((1, ah, ad), dtype=np.float32)
        gt = np.asarray(r["action"], np.float32).reshape(-1)[:128]

        fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
        axp = fig.add_axes([0, 0, 1, 1])
        axp.imshow(img)
        axp.axis("off")
        for name, color, policy in series:
            if policy == "gt":
                flat = gt
            elif policy is None:
                continue  # model not loaded -> don't draw / don't legend it
            else:
                flat = np.asarray(policy.infer(obs, noise=noise)["actions"], np.float32).reshape(-1)[:128]
            xyz = actions_to_xyz(unflatten_actions(flat), action_format="accel_curvature",
                                 dt=0.1, initial_speed=speed, initial_yaw=0.0)
            uv = _project_path(xyz[:, :2], W, H, focal, cam_h, horizon_frac)
            if len(uv) >= 2:
                axp.plot(uv[:, 0], uv[:, 1], "-", color=color, lw=3, alpha=0.9, label=name)
        axp.text(8, 22, f"{r['prompt'][:28]}  v={speed:.1f} m/s", color="white", fontsize=11,
                 bbox=dict(facecolor="black", alpha=0.5, pad=2))
        axp.legend(loc="lower right", fontsize=10, framealpha=0.6)
        axp.set_xlim(0, W)
        axp.set_ylim(H, 0)
        png = outdir / f"frame_{k:03d}_idx{idx}.png"
        fig.savefig(png, dpi=100)
        plt.close(fig)
        if k % 5 == 0:
            print(f"[overlay] {k + 1}/{len(indices)}", flush=True)

    cache_volume.commit()
    print(f"[overlay] wrote {len(indices)} frames to {outdir}", flush=True)
    return {"outdir": str(outdir), "n": len(indices),
            "have_grpo": grpo is not None, "have_dpo": dpo is not None}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def viz_compare(
    num_samples: int = 10,
    seed: int = 123,
    grpo_ckpt_dir: str = f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/grpo-50ep-fixed/50",
    dpo_ckpt_dir: str = f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/dpo-50ep-fixed/50",
    tag: str = "compare",
) -> dict:
    """Render N held-out clips: camera image + top-down GT / BC / GRPO / DPO trajectories.

    Any of grpo/dpo checkpoints that are missing are simply skipped. Saves PNGs to the
    volume under /cache/viz/<tag>_<N>/. Fetch locally with:
        modal volume get pi05-cache viz/<tag>_<N> ./viz_local
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    bc_ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(bc_ckpt)  # real norm stats + correct ckpt path

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    from grpo.eval_indices import pick_eval_holdout_indices
    from grpo.eval_metrics import eval_candidate_vs_references
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout
    from grpo.parquet_frames import FrameSource
    from rewards.action_space import actions_to_xyz
    from rewards.flat_actions import unflatten_actions

    config = _config.get_config(OPENPI_CONFIG_NAME)
    dc = config.data.create(config.assets_dirs, config.model)
    frames = FrameSource(dataset_root, dc)
    ah, ad = config.model.action_horizon, config.model.action_dim

    _ensure_checkpoint_layout(bc_ckpt, OPENPI_DIR)

    def _maybe_load(path):
        if os.path.isdir(f"{path}/params"):
            _ensure_checkpoint_layout(path, OPENPI_DIR)
            return policy_config.create_trained_policy(config, path)
        return None

    print(f"[viz] loading BC; grpo={grpo_ckpt_dir}; dpo={dpo_ckpt_dir}", flush=True)
    bc = policy_config.create_trained_policy(config, bc_ckpt)
    grpo = _maybe_load(grpo_ckpt_dir)
    dpo = _maybe_load(dpo_ckpt_dir)
    print(f"[viz] loaded BC, grpo={grpo is not None}, dpo={dpo is not None}", flush=True)

    indices = pick_eval_holdout_indices(len(frames), num_samples, seed=seed)
    outdir = Path(CACHE_DIR) / "viz" / f"{tag}_{num_samples}"
    outdir.mkdir(parents=True, exist_ok=True)

    def traj(flat, speed):
        return actions_to_xyz(unflatten_actions(flat), action_format="accel_curvature",
                              dt=0.1, initial_speed=speed, initial_yaw=0.0)

    rows = []
    for k, idx in enumerate(indices):
        r = frames.row(idx)
        obs = {"observation/image": r["observation.images.front"],
               "observation/state": r["observation.state"], "prompt": r["prompt"]}
        speed = float(r["observation.state"][0]) if r["observation.state"].size else 5.0
        noise = np.random.default_rng(seed + idx).standard_normal((1, ah, ad), dtype=np.float32)
        gt = np.asarray(r["action"], np.float32).reshape(-1)[:128]
        bc_flat = np.asarray(bc.infer(obs, noise=noise)["actions"], np.float32).reshape(-1)[:128]
        gt_xyz, bc_xyz = traj(gt, speed), traj(bc_flat, speed)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.imshow(np.asarray(r["observation.images.front"]).astype(np.uint8))
        a1.set_title(f"clip {idx} — '{r['prompt'][:32]}'  v={speed:.1f}")
        a1.axis("off")
        a2.plot(gt_xyz[:, 0], gt_xyz[:, 1], "k-", lw=2.5, label="GT")
        a2.plot(bc_xyz[:, 0], bc_xyz[:, 1], "b--", lw=2, label="BC (pretrained)")
        bc_m = eval_candidate_vs_references(bc_flat, gt_flat=gt, initial_speed=speed)
        row = {"i": k, "index": int(idx), "prompt": r["prompt"][:40],
               "bc_ade_m": bc_m.get("gt_ade_m"), "bc_fde_m": bc_m.get("gt_fde_m"),
               "bc_val_loss": bc_m.get("gt_action_mse")}
        if grpo is not None:
            gr_flat = np.asarray(grpo.infer(obs, noise=noise)["actions"], np.float32).reshape(-1)[:128]
            a2.plot(*traj(gr_flat, speed).T[:2], "r--", lw=2, label="GRPO")
            gr_m = eval_candidate_vs_references(gr_flat, gt_flat=gt, initial_speed=speed)
            row["grpo_ade_m"] = gr_m.get("gt_ade_m")
            row["grpo_fde_m"] = gr_m.get("gt_fde_m")
            row["grpo_val_loss"] = gr_m.get("gt_action_mse")
        if dpo is not None:
            dp_flat = np.asarray(dpo.infer(obs, noise=noise)["actions"], np.float32).reshape(-1)[:128]
            a2.plot(*traj(dp_flat, speed).T[:2], "g--", lw=2, label="DPO")
            dp_m = eval_candidate_vs_references(dp_flat, gt_flat=gt, initial_speed=speed)
            row["dpo_ade_m"] = dp_m.get("gt_ade_m")
            row["dpo_fde_m"] = dp_m.get("gt_fde_m")
            row["dpo_val_loss"] = dp_m.get("gt_action_mse")
        a2.set_aspect("equal", adjustable="datalim")
        a2.legend(loc="best")
        a2.grid(True, alpha=0.3)
        a2.set_title("trajectory, top-down (m)")
        a2.set_xlabel("x (forward)")
        a2.set_ylabel("y (left)")
        fig.tight_layout()
        fig.savefig(outdir / f"clip_{k:02d}_idx{idx}.png", dpi=90)
        plt.close(fig)
        rows.append(row)
        print(json.dumps(row), flush=True)

    # Validation summary: mean held-out ADE (m) and validation loss (action-space MSE vs GT).
    def _mean(key):
        vals = [rr[key] for rr in rows if rr.get(key) is not None and np.isfinite(rr[key])]
        return float(np.mean(vals)) if vals else None

    summary = {"n": len(rows)}
    for m in ("bc", "grpo", "dpo"):
        summary[f"{m}_ade_m_mean"] = _mean(f"{m}_ade_m")
        summary[f"{m}_val_loss_mean"] = _mean(f"{m}_val_loss")  # action MSE vs GT

    cache_volume.commit()
    report = {"outdir": str(outdir), "n": len(indices),
              "have_grpo": bool(grpo), "have_dpo": bool(dpo),
              "summary": summary, "rows": rows}
    (outdir / "summary.json").write_text(json.dumps(report, indent=2))
    cache_volume.commit()
    print("[viz] VALIDATION SUMMARY:", json.dumps(summary), flush=True)
    print(f"[viz] wrote {len(indices)} PNGs to {outdir}", flush=True)
    return report


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 4,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def gen_judge_candidates(
    num_clips: int = 1500,
    group_size: int = 3,
    seed: int = 42,
    tag: str = "cosmos_g3",
) -> dict:
    """Stage A of the Cosmos-judge DPO pipeline.

    For each training clip: sample ``group_size`` candidate action chunks from BC, render a
    (camera | top-down BEV with numbered candidate paths) PNG for the VLM judge, and save the
    NORMALIZED candidate actions. Stage B (``dpo/modal_cosmos_judge.py::judge_candidates``)
    ranks the PNGs; Stage C trains DPO on the chosen/rejected pairs.
    """
    import json
    import os

    import numpy as np

    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", str(Path(ckpt) / "params"))

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import matplotlib

    matplotlib.use("Agg")
    import jax
    import matplotlib.pyplot as plt
    import openpi.training.config as _config

    from grpo.eval_indices import pick_eval_indices
    from grpo.openpi_grpo_train import (
        build_datasets,
        prepare_group_observation,
        unnormalize_actions,
    )
    from grpo.trainable_state import build_train_state, make_sampler
    from rewards.action_space import actions_to_xyz
    from rewards.flat_actions import unflatten_actions

    config = _config.get_config(OPENPI_CONFIG_NAME)
    state, _ = build_train_state(config, seed=seed, peak_lr=1e-6)  # frozen BC
    sampler = make_sampler(config)
    frames, data_config = build_datasets(config, dataset_root=dataset_root)
    ad, ah = config.model.action_dim, config.model.action_horizon

    indices = pick_eval_indices(len(frames), max(num_clips, 1), seed=seed)
    outdir = Path(CACHE_DIR) / "cosmos_stage" / tag
    (outdir / "imgs").mkdir(parents=True, exist_ok=True)
    (outdir / "cands").mkdir(parents=True, exist_ok=True)
    colors = ["red", "green", "blue", "orange", "purple", "brown"]

    def traj(flat, speed):
        return actions_to_xyz(
            unflatten_actions(flat), action_format="accel_curvature",
            dt=0.1, initial_speed=speed, initial_yaw=0.0,
        )

    rng = jax.random.key(seed)
    manifest = []
    for k, idx in enumerate(indices):
        item = frames.model_item(idx)
        r = frames.row(idx)
        speed = float(r["observation.state"][0]) if np.asarray(r["observation.state"]).size else 5.0
        obs = prepare_group_observation(item, group_size)
        rng, srng = jax.random.split(rng)
        noise = jax.random.normal(srng, (group_size, ah, ad))
        norm = np.asarray(sampler(state, srng, obs, noise), np.float32)  # (G, ah, ad)
        phys = unnormalize_actions(norm, data_config)  # (G, 128)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.imshow(np.asarray(r["observation.images.front"]).astype(np.uint8))
        a1.axis("off")
        a1.set_title(f"clip {idx} — '{str(r['prompt'])[:32]}'  v={speed:.1f}")
        for g in range(group_size):
            xyz = traj(phys[g], speed)
            a2.plot(xyz[:, 0], xyz[:, 1], color=colors[g % len(colors)], lw=2.5, label=f"path {g + 1}")
        a2.set_aspect("equal", adjustable="datalim")
        a2.legend(loc="best")
        a2.grid(True, alpha=0.3)
        a2.set_title("candidate future paths (top-down)")
        a2.set_xlabel("x forward (m)")
        a2.set_ylabel("y left (m)")
        fig.tight_layout()
        stem = f"clip_{k:05d}_idx{idx}"
        fig.savefig(outdir / "imgs" / f"{stem}.png", dpi=90)
        plt.close(fig)
        np.savez(
            outdir / "cands" / f"{stem}.npz",
            norm=norm, index=int(idx), speed=float(speed), prompt=str(r["prompt"]),
        )
        manifest.append({"k": k, "index": int(idx), "stem": stem, "prompt": str(r["prompt"])[:60]})
        if k % 50 == 0:
            print(f"[stageA] {k}/{len(indices)}", flush=True)
            cache_volume.commit()

    (outdir / "manifest.json").write_text(json.dumps(manifest))
    cache_volume.commit()
    print(f"[stageA] wrote {len(manifest)} candidate sets -> {outdir}", flush=True)
    return {"tag": tag, "n": len(manifest), "group_size": group_size, "outdir": str(outdir)}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 2,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def render_swapped(tag: str = "cosmos_g2") -> dict:
    """Debias prep: re-render each Stage-A clip with the BEV path order SWAPPED.

    Reads the SAME candidates from cosmos_stage/<tag>/cands; writes a sibling stage
    cosmos_stage/<tag>_swap/ (copied manifest + imgs/ where path1=cand1, path2=cand0).
    Then ``judge_candidates(tag=<tag>_swap)`` runs unchanged, and the orig+swap verdicts
    are combined for swap-consistency (drop position-locked, keep consistent).
    """
    import json
    import os

    import numpy as np

    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", str(Path(ckpt) / "params"))
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openpi.training.config as _config

    from grpo.openpi_grpo_train import build_datasets, unnormalize_actions
    from rewards.action_space import actions_to_xyz
    from rewards.flat_actions import unflatten_actions

    config = _config.get_config(OPENPI_CONFIG_NAME)
    frames, data_config = build_datasets(config, dataset_root=dataset_root)
    colors = ["red", "green", "blue", "orange", "purple", "brown"]

    src = Path(CACHE_DIR) / "cosmos_stage" / tag
    dst = Path(CACHE_DIR) / "cosmos_stage" / f"{tag}_swap"
    (dst / "imgs").mkdir(parents=True, exist_ok=True)
    (dst / "manifest.json").write_text((src / "manifest.json").read_text())  # same manifest

    def traj(flat, speed):
        return actions_to_xyz(
            unflatten_actions(flat), action_format="accel_curvature",
            dt=0.1, initial_speed=speed, initial_yaw=0.0,
        )

    manifest = json.loads((src / "manifest.json").read_text())
    n = 0
    for e in manifest:
        npz = np.load(src / "cands" / f"{e['stem']}.npz", allow_pickle=True)
        norm = np.asarray(npz["norm"], np.float32)  # (G, ah, ad)
        phys = unnormalize_actions(norm, data_config)  # (G, 128)
        idx = int(npz["index"])
        speed = float(npz["speed"])
        r = frames.row(idx)
        order = list(range(phys.shape[0]))[::-1]  # SWAP: [1, 0] for g=2

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
        a1.imshow(np.asarray(r["observation.images.front"]).astype(np.uint8))
        a1.axis("off")
        a1.set_title(f"clip {idx} — '{str(r['prompt'])[:32]}'  v={speed:.1f}")
        for slot, g in enumerate(order):  # slot 0 -> path1(red), slot 1 -> path2(green)
            xyz = traj(phys[g], speed)
            a2.plot(xyz[:, 0], xyz[:, 1], color=colors[slot % len(colors)], lw=2.5, label=f"path {slot + 1}")
        a2.set_aspect("equal", adjustable="datalim")
        a2.legend(loc="best")
        a2.grid(True, alpha=0.3)
        a2.set_title("candidate future paths (top-down)")
        a2.set_xlabel("x forward (m)")
        a2.set_ylabel("y left (m)")
        fig.tight_layout()
        fig.savefig(dst / "imgs" / f"{e['stem']}.png", dpi=90)
        plt.close(fig)
        n += 1
        if n % 100 == 0:
            print(f"[swap] {n}/{len(manifest)}", flush=True)
            cache_volume.commit()

    cache_volume.commit()
    print(f"[swap] wrote {n} swapped renders -> {dst} (order={order})", flush=True)
    return {"tag_swap": f"{tag}_swap", "n": n, "order": order}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def compare_videos(
    num_samples: int = 10,
    seed: int = 42,
    grpo_ckpt_dir: str = f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/grpo-50ep/50",
    use_holdout: bool = True,
) -> dict:
    """Open-loop compare BC (pretrained) vs GRPO'd checkpoint on N held-out clips.

    Same clips + same inference noise for both models. Reads frames via the parquet
    FrameSource (bypasses lerobot), runs policy.infer on each, reports ADE/FDE/MSE vs GT.
    """
    import numpy as np

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    bc_ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(bc_ckpt)
    os.environ["PI05_BC_CHECKPOINT_PARAMS"] = str(Path(bc_ckpt) / "params")

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import openpi.policies.policy_config as policy_config
    import openpi.training.config as _config

    from grpo.eval_indices import pick_eval_holdout_indices, pick_eval_indices
    from grpo.eval_metrics import eval_candidate_vs_references
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout
    from grpo.parquet_frames import FrameSource

    config = _config.get_config(OPENPI_CONFIG_NAME)
    data_config = config.data.create(config.assets_dirs, config.model)
    frames = FrameSource(dataset_root, data_config)
    ah, ad = config.model.action_horizon, config.model.action_dim

    _ensure_checkpoint_layout(bc_ckpt, OPENPI_DIR)
    _ensure_checkpoint_layout(grpo_ckpt_dir, OPENPI_DIR)
    print(f"[compare] loading BC {bc_ckpt}", flush=True)
    bc_policy = policy_config.create_trained_policy(config, bc_ckpt)
    print(f"[compare] loading GRPO {grpo_ckpt_dir}", flush=True)
    grpo_policy = policy_config.create_trained_policy(config, grpo_ckpt_dir)

    picker = pick_eval_holdout_indices if use_holdout else pick_eval_indices
    indices = picker(len(frames), num_samples, seed=seed)

    def _infer(policy, obs, noise):
        out = policy.infer(obs, noise=noise)
        return np.asarray(out["actions"], dtype=np.float32).reshape(-1)[:128]

    rows = []
    for i, idx in enumerate(indices):
        r = frames.row(idx)
        obs = {
            "observation/image": r["observation.images.front"],
            "observation/state": r["observation.state"],
            "prompt": r["prompt"],
        }
        gt = np.asarray(r["action"], np.float32).reshape(-1)[:128]
        speed = float(r["observation.state"][0]) if r["observation.state"].size else 0.0
        rng = np.random.default_rng(seed + idx)
        noise = rng.standard_normal((1, ah, ad), dtype=np.float32)

        bc_flat = _infer(bc_policy, obs, noise)
        grpo_flat = _infer(grpo_policy, obs, noise)
        bc_m = eval_candidate_vs_references(bc_flat, gt_flat=gt, initial_speed=speed)
        grpo_m = eval_candidate_vs_references(grpo_flat, gt_flat=gt, initial_speed=speed)
        rows.append({
            "i": i, "index": int(idx), "prompt": r["prompt"][:40],
            "bc_ade_m": bc_m.get("gt_ade_m"), "grpo_ade_m": grpo_m.get("gt_ade_m"),
            "bc_fde_m": bc_m.get("gt_fde_m"), "grpo_fde_m": grpo_m.get("gt_fde_m"),
            "bc_mse": bc_m.get("gt_action_mse"), "grpo_mse": grpo_m.get("gt_action_mse"),
            "delta_ade_m": (grpo_m.get("gt_ade_m") or 0) - (bc_m.get("gt_ade_m") or 0),
        })
        print(json.dumps(rows[-1]), flush=True)

    def _mean(k):
        vals = [r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        return float(np.mean(vals)) if vals else None

    wins = sum(1 for r in rows if r["delta_ade_m"] < 0)
    summary = {
        "n": len(rows),
        "bc_ade_m_mean": _mean("bc_ade_m"), "grpo_ade_m_mean": _mean("grpo_ade_m"),
        "bc_fde_m_mean": _mean("bc_fde_m"), "grpo_fde_m_mean": _mean("grpo_fde_m"),
        "mean_delta_ade_m": _mean("delta_ade_m"),
        "grpo_ade_win_rate": wins / len(rows) if rows else None,
        "verdict": "grpo_better_ade" if (_mean("delta_ade_m") or 0) < 0 else "bc_better_ade",
    }
    report = {"summary": summary, "indices": [int(x) for x in indices],
              "grpo_ckpt_dir": grpo_ckpt_dir, "rows": rows}
    out_path = Path(CACHE_DIR) / "grpo_eval" / f"compare_videos_{num_samples}_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    cache_volume.commit()
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    return report


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=48 * 1024,
)
def probe_episodes() -> dict:
    """CPU: episode/clip structure (frames per episode) + scan meta for camera intrinsics."""
    import glob as _glob
    import numpy as np
    import pyarrow.parquet as pq

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    out = {"dataset_root": dataset_root}

    files = sorted(_glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True))
    cols = set(pq.ParquetFile(files[0]).schema_arrow.names)
    out["columns"] = sorted(cols)
    ep_col = next((c for c in ("episode_index", "episode_id") if c in cols), None)
    out["episode_col"] = ep_col
    if ep_col:
        eps = []
        for f in files:
            eps.append(np.asarray(pq.read_table(f, columns=[ep_col])[ep_col].to_pylist()))
        ep = np.concatenate(eps)
        uniq, counts = np.unique(ep, return_counts=True)
        out["n_episodes"] = int(uniq.size)
        out["total_frames"] = int(ep.size)
        out["frames_per_ep_min_med_max"] = [int(counts.min()), int(np.median(counts)), int(counts.max())]
        # episodes with a decent number of frames, for video clips
        good = uniq[counts >= 20][:15]
        out["sample_episode_ids_ge20"] = [int(x) for x in good]
        out["sample_episode_sizes"] = [int(counts[list(uniq).index(x)]) for x in good]

    # Scan meta JSONs for any camera calibration / intrinsics.
    meta_files = _glob.glob(f"{dataset_root}/meta/**/*.json", recursive=True)
    hits = {}
    for mf in meta_files:
        try:
            txt = Path(mf).read_text()[:20000].lower()
        except Exception:  # noqa: BLE001
            continue
        found = [kw for kw in ("intrinsic", "focal", "fx", "fy", "camera_matrix", "calibration", "fov", "distortion") if kw in txt]
        if found:
            hits[os.path.basename(mf)] = found
    out["meta_files"] = [os.path.basename(m) for m in meta_files]
    out["intrinsics_hits"] = hits
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=48 * 1024,
)
def smoke_data_cpu() -> dict:
    """CPU: validate the full parquet -> transform -> Observation path (no model, no GPU)."""
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    os.environ.setdefault("PI05_BC_CHECKPOINT_PARAMS", ckpt)

    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import numpy as np
    import openpi.training.config as _config

    from grpo.openpi_grpo_train import build_datasets, prepare_group_observation, unnormalize_actions

    config = _config.get_config(OPENPI_CONFIG_NAME)
    frames, data_config = build_datasets(config, dataset_root=dataset_root)

    out = {"describe": frames.describe(), "len": len(frames)}
    raw = frames.row(0)
    out["row0"] = {
        "image": list(np.asarray(raw["observation.images.front"]).shape),
        "state": list(np.asarray(raw["observation.state"]).shape),
        "action": list(np.asarray(raw["action"]).shape),
        "prompt": raw["prompt"][:80],
    }
    item = frames.model_item(0)
    out["model_item_keys"] = list(item.keys())
    obs = prepare_group_observation(item, 3)  # tile to group size 3
    out["obs"] = {
        "state": list(obs.state.shape),
        "images": {k: list(v.shape) for k, v in obs.images.items()},
        "tokenized_prompt": None if obs.tokenized_prompt is None else list(obs.tokenized_prompt.shape),
    }
    # Sanity: un-normalize a fake (3,4,32) chunk -> (3,128).
    fake = np.zeros((3, config.model.action_horizon, config.model.action_dim), np.float32)
    out["unnorm_shape"] = list(unnormalize_actions(fake, data_config).shape)
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=48 * 1024,
)
def diagnose_dataset() -> dict:
    """CPU: dump the dataset's real schema + how a frame is built, so we stop guessing.

    Reports meta feature/image/video keys, the underlying hf_dataset columns + a sample
    row's keys, what lerobot's __getitem__ returns (or the error), and the source of
    __getitem__ / _get_query_indices.
    """
    import inspect

    import glob

    os.environ["HF_HUB_DISABLE_XET"] = "1"  # avoid hf_xet 'request_headers' download bug
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    out = {"dataset_root": dataset_root}

    # Read the parquet directly — bypass lerobot entirely (no download, no episode index).
    parquets = sorted(glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True))
    out["n_parquet_files"] = len(parquets)
    out["first_parquet"] = parquets[0] if parquets else None
    if not parquets:
        out["meta_info"] = sorted(glob.glob(f"{dataset_root}/**/*.json", recursive=True))[:20]
        print(json.dumps(out, indent=2, default=str))
        return out

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquets[0])
    out["num_rows_file0"] = pf.metadata.num_rows
    out["columns"] = [pf.schema_arrow.field(i).name for i in range(len(pf.schema_arrow))]
    out["column_types"] = {f.name: str(f.type) for f in pf.schema_arrow}

    row = pf.read_row_group(0).slice(0, 1).to_pylist()[0]

    def _describe(v):
        import numpy as _np
        if isinstance(v, dict):
            return {"dict_keys": list(v.keys()),
                    "bytes_len": len(v["bytes"]) if v.get("bytes") else None,
                    "path": v.get("path")}
        if isinstance(v, (list, tuple)):
            arr = _np.asarray(v)
            return {"list_len": len(v), "shape": list(arr.shape), "dtype": str(arr.dtype)}
        if isinstance(v, (bytes, bytearray)):
            return {"bytes_len": len(v)}
        return {"type": type(v).__name__, "value": str(v)[:80]}

    out["row0"] = {k: _describe(v) for k, v in row.items()}
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=48 * 1024,
)
def probe_params(repo_id: str = HF_BC_CHECKPOINT_REPO) -> dict:
    """Ground truth: restore raw params from the checkpoint ROOT vs ROOT/params and print
    action_in_proj shapes (no model, no weight-loader slicing). Settles 32-vs-128."""
    import glob as _glob

    os.environ["HF_HUB_DISABLE_XET"] = "1"
    ckpt = ensure_bc_checkpoint(CACHE_DIR, repo_id=repo_id)
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout

    _ensure_checkpoint_layout(ckpt, OPENPI_DIR)
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    import numpy as np
    import flax.traverse_util as traverse_util
    from openpi.models import model as _model

    out = {"ckpt": ckpt}
    out["root_contents"] = sorted(os.path.basename(p) for p in _glob.glob(f"{ckpt}/*"))
    out["params_contents"] = sorted(os.path.basename(p) for p in _glob.glob(f"{ckpt}/params/*"))
    for label, path in (("ROOT", ckpt), ("ROOT/params", f"{ckpt}/params")):
        try:
            p = _model.restore_params(path, restore_type=np.ndarray)
            flat = traverse_util.flatten_dict(p)
            hit = {"/".join(map(str, k)): list(np.shape(v))
                   for k, v in flat.items() if "action_in_proj" in "/".join(map(str, k))}
            out[label] = {"n_leaves": len(flat), "action_in_proj": hit}
        except Exception as e:  # noqa: BLE001
            out[label] = {"error": f"{type(e).__name__}: {e}"}
    print(json.dumps(out, indent=2, default=str))
    return out


@app.function(
    image=image,
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=48 * 1024,
)
def diagnose_load(repo_id: str = HF_BC_CHECKPOINT_REPO) -> dict:
    """CPU: diff the model's expected param tree vs the on-disk checkpoint shapes.

    Pinpoints exactly which tensors mismatch / are missing, so we know whether the
    fault is geometry (action_dim) or the weight-loader / LoRA layout.
    """
    ckpt = ensure_bc_checkpoint(CACHE_DIR, repo_id=repo_id)
    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        ensure_driving_norm_stats,
        patch_openpi,
        prepend_openpi_venv,
    )

    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout

    # Bake the BC checkpoint params path BEFORE patching (else config -> 32-dim pi05_base).
    os.environ["PI05_BC_CHECKPOINT_PARAMS"] = f"{ckpt}/params"
    _ensure_checkpoint_layout(ckpt, OPENPI_DIR)
    patch_openpi(OPENPI_DIR)
    prepend_openpi_venv(OPENPI_DIR)
    ensure_driving_norm_stats(OPENPI_DIR, CACHE_DIR, ckpt)

    import jax
    import flax.traverse_util as traverse_util
    import openpi.training.config as _config

    config = _config.get_config(OPENPI_CONFIG_NAME)
    import flax.nnx as nnx

    model = nnx.eval_shape(config.model.create, jax.random.key(0))
    _graphdef, _state = nnx.split(model)
    expected = _state.to_pure_dict()
    exp_flat = {"/".join(map(str, k)): tuple(getattr(v, "shape", ()))
                for k, v in traverse_util.flatten_dict(expected).items()}

    loaded = config.weight_loader.load(expected)
    got_flat = {"/".join(map(str, k)): tuple(getattr(v, "shape", ()))
                for k, v in traverse_util.flatten_dict(loaded).items()}

    mismatches = {k: {"expected": exp_flat[k], "got": got_flat.get(k)}
                  for k in exp_flat if k in got_flat and exp_flat[k] != got_flat[k]}
    missing = [k for k in exp_flat if k not in got_flat]
    extra = [k for k in got_flat if k not in exp_flat]
    report = {
        "repo_id": repo_id,
        "n_expected": len(exp_flat),
        "n_got": len(got_flat),
        "action_in_proj_expected": {k: v for k, v in exp_flat.items() if "action_in_proj" in k},
        "action_in_proj_got": {k: v for k, v in got_flat.items() if "action_in_proj" in k},
        "n_mismatches": len(mismatches),
        "mismatches": dict(list(mismatches.items())[:20]),
        "n_missing": len(missing),
        "missing_sample": missing[:20],
        "n_extra": len(extra),
        "extra_sample": extra[:20],
    }
    print(json.dumps(report, indent=2))
    return report


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def smoke_grpo_step(group_size: int = 8, dataset_index: int = 0) -> dict:
    """Load Mark BC checkpoint, sample G actions, composite reward, flow losses."""
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    return _run_grpo_step_script(
        ckpt_dir=ckpt,
        group_size=group_size,
        dataset_index=dataset_index,
        num_train_steps=0,
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface"),
        modal.Secret.from_name("wandb"),
    ],
    memory=80 * 1024,
)
def train_grpo(
    num_steps: int = 500,
    group_size: int = 8,
    exp_name: str = "grpo-posttrain",
) -> dict:
    """GRPO post-training loop. Implements ranking; calls your objective in grpo/objective.py."""
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    return _run_grpo_step_script(
        ckpt_dir=ckpt,
        group_size=group_size,
        dataset_index=0,
        num_train_steps=num_steps,
        exp_name=exp_name,
    )


def _prepare_dataset_and_assets(ckpt_dir: str) -> str:
    """Patch openpi, ensure norm stats + local dataset; return dataset root."""
    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        ensure_driving_norm_stats,
        link_dataset_to_hf_cache,
        patch_openpi,
    )

    # CRITICAL: bake Mark's BC checkpoint path into the config BEFORE patching, else
    # patch_openpi falls back to gs://.../pi05_base/params (the 32-dim base model) and
    # training silently runs on the base instead of the BC checkpoint.
    os.environ["PI05_BC_CHECKPOINT_PARAMS"] = f"{ckpt_dir}/params"
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout

    _ensure_checkpoint_layout(ckpt_dir, OPENPI_DIR)  # create params/ symlink dir
    patch_openpi(OPENPI_DIR)
    stats_path = ensure_driving_norm_stats(OPENPI_DIR, CACHE_DIR, ckpt_dir)
    train_local = f"{CACHE_DIR}/hf/lerobot/{HF_BC_DATASET_REPO}"
    if not os.path.isdir(train_local):
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=HF_BC_DATASET_REPO, repo_type="dataset", local_dir=train_local)
    link_dataset_to_hf_cache(CACHE_DIR, HF_BC_DATASET_REPO)

    # Replace bootstrap (identity) stats with REAL stats from the parquet so the BC
    # policy (and GRPO samples) are correctly scaled. Key is the symlink target file.
    try:
        from grpo.parquet_frames import FrameSource

        real = FrameSource(train_local).compute_norm_stats()
        target = os.path.realpath(stats_path)
        Path(target).write_text(json.dumps(real, indent=2))
        print(f"[norm] wrote real norm stats -> {target}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[norm] real-stats computation failed ({e}); keeping bootstrap", flush=True)
    return train_local


def _run_grpo_train_script(
    *,
    ckpt_dir: str,
    dataset_root: str,
    num_steps: int,
    group_size: int,
    save_interval: int,
    seed: int,
    kl_coef: float,
    peak_lr: float,
    exp_name: str,
    out_dir: str,
    resume_from: str | None = None,
) -> dict:
    """Subprocess the real training loop inside the openpi venv."""
    runner = Path("/app/pi_05_drives/grpo/openpi_grpo_train.py")
    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", str(runner),
        "--ckpt-dir", ckpt_dir,
        "--dataset-root", dataset_root,
        "--out-dir", out_dir,
        "--num-steps", str(num_steps),
        "--group-size", str(group_size),
        "--save-interval", str(save_interval),
        "--seed", str(seed),
        "--kl-coef", str(kl_coef),
        "--peak-lr", str(peak_lr),
        "--exp-name", exp_name,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    env = _setup_openpi_env(ckpt_dir)
    env["LEROBOT_DATASET_ROOT"] = dataset_root
    # Stream live to the Modal logs (don't buffer) so long runs are monitorable.
    proc = subprocess.run(cmd, cwd=OPENPI_DIR, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"grpo train runner failed rc={proc.returncode}")

    cache_volume.commit()
    report_path = Path(CACHE_DIR) / "grpo_reports" / f"{exp_name}_train.json"
    if report_path.exists():
        with report_path.open() as f:
            return json.load(f)
    return {"status": "ok", "note": "streamed to logs"}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def smoke_grpo_train(group_size: int = 4) -> dict:
    """Cheap end-to-end check: build TrainState, 2 real gradient steps, no save."""
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    return _run_grpo_train_script(
        ckpt_dir=ckpt, dataset_root=dataset_root, num_steps=2, group_size=group_size,
        save_interval=0, seed=42, kl_coef=1.0, peak_lr=1e-6, exp_name="grpo-train-smoke",
        out_dir=f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/grpo-train-smoke",
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    memory=80 * 1024,
)
def train_grpo_real(
    num_steps: int = 500,
    group_size: int = 8,
    save_interval: int = 100,
    seed: int = 42,
    kl_coef: float = 1.0,
    peak_lr: float = 1e-6,
    exp_name: str = "grpo-posttrain",
    resume_from: str | None = None,
) -> dict:
    """Real Flow-GRPO post-training: gradient + optax + orbax checkpoint.

    Produces /cache/checkpoints/pi05_driving/<exp_name>/<step> loadable by compare_fifty:
      modal run grpo/modal_posttrain.py::compare_fifty --num-samples 50 \\
        --grpo-ckpt-dir /cache/checkpoints/pi05_driving/<exp_name>/<num_steps>

    To resume from a prior checkpoint on a FRESH container (sidesteps the per-process
    GPU memory leak), pass --resume-from with a step dir, e.g.:
      modal run --detach grpo/modal_posttrain.py::train_grpo_real --num-steps 2500 \\
        --resume-from /cache/checkpoints/pi05_driving/grpo-5k-v2/2500 --exp-name grpo-5k-v2
    Checkpoints are named by GLOBAL step (resume 2500 + 2500 more -> saves at 5000).
    """
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_root = _prepare_dataset_and_assets(ckpt)
    return _run_grpo_train_script(
        ckpt_dir=ckpt, dataset_root=dataset_root, num_steps=num_steps, group_size=group_size,
        save_interval=save_interval, seed=seed, kl_coef=kl_coef, peak_lr=peak_lr,
        exp_name=exp_name,
        out_dir=f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/{exp_name}",
        resume_from=resume_from,
    )


def _run_grpo_step_script(
    *,
    ckpt_dir: str,
    group_size: int,
    dataset_index: int,
    num_train_steps: int,
    exp_name: str = "grpo-smoke",
    dataset_root: str | None = None,
) -> dict:
    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        ensure_driving_norm_stats,
        link_dataset_to_hf_cache,
        patch_openpi,
    )

    patch_openpi(OPENPI_DIR)
    ensure_driving_norm_stats(OPENPI_DIR, CACHE_DIR, ckpt_dir)
    if dataset_root is None:
        train_local = f"{CACHE_DIR}/hf/lerobot/{HF_BC_DATASET_REPO}"
        if not os.path.isdir(train_local):
            from huggingface_hub import snapshot_download

            snapshot_download(
                repo_id=HF_BC_DATASET_REPO,
                repo_type="dataset",
                local_dir=train_local,
            )
        dataset_root = train_local
    repo_id = HF_BC_DATASET_REPO
    link_dataset_to_hf_cache(CACHE_DIR, repo_id)

    runner = Path("/app/pi_05_drives/grpo/openpi_grpo_runner.py")
    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python",
        "-u",
        str(runner),
        "--ckpt-dir",
        ckpt_dir,
        "--group-size",
        str(group_size),
        "--dataset-index",
        str(dataset_index),
        "--dataset-root",
        dataset_root,
        "--num-steps",
        str(num_train_steps),
        "--exp-name",
        exp_name,
    ]
    env = _setup_openpi_env(ckpt_dir)
    env["LEROBOT_DATASET_ROOT"] = dataset_root
    proc = subprocess.run(cmd, cwd=OPENPI_DIR, env=env, text=True, capture_output=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"grpo runner failed rc={proc.returncode}")

    cache_volume.commit()
    report_path = Path(CACHE_DIR) / "grpo_reports" / f"{exp_name}_last.json"
    if report_path.exists():
        with report_path.open() as f:
            return json.load(f)
    return {"status": "ok", "stdout_tail": proc.stdout[-2000:]}


@app.function(
    image=image,
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_eval_dataset(force: bool = False) -> str:
    """Download BC LeRobot dataset (train repo; eval repo if it exists on HF)."""
    local, _repo_id, _ = ensure_bc_dataset(CACHE_DIR, force=force)
    cache_volume.commit()
    return local


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 6,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def compare_fifty(
    num_samples: int = 50,
    seed: int = 42,
    grpo_ckpt_dir: str | None = None,
    grpo_exp_name: str = "grpo-posttrain",
    save_full_rows: bool = False,
) -> dict:
    """Open-loop compare BC (HF) vs GRPO post-trained on N eval samples."""
    import json
    from pathlib import Path

    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        link_dataset_to_hf_cache,
        patch_openpi,
        prepend_openpi_venv,
    )
    from grpo.openpi_eval import compare_checkpoints, find_latest_grpo_checkpoint

    patch_openpi(OPENPI_DIR)
    prepend_openpi_venv(OPENPI_DIR)
    os.environ["HF_HOME"] = f"{CACHE_DIR}/hf"
    # The BC checkpoint + dataset are already cached on the volume. HF has been throttling
    # us (429 / connection-reset) after all the big model pulls + uploads, so go fully OFFLINE
    # — use the local cache, make zero HF API calls.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, use_holdout = ensure_bc_dataset(CACHE_DIR)
    link_dataset_to_hf_cache(CACHE_DIR, repo_id)

    bc_ckpt = CKPT_LOCAL
    grpo_ckpt = grpo_ckpt_dir or find_latest_grpo_checkpoint(CACHE_DIR, grpo_exp_name)
    if not grpo_ckpt:
        print(
            f"WARNING: No GRPO checkpoint at /cache/checkpoints/{OPENPI_CONFIG_NAME}/{grpo_exp_name}. "
            "Running BC-only baseline. Pass --grpo-ckpt-dir after training."
        )
    out_path = Path(CACHE_DIR) / "grpo_eval" / f"compare_{num_samples}_seed{seed}.json"
    report = compare_checkpoints(
        openpi_dir=OPENPI_DIR,
        bc_ckpt_dir=bc_ckpt,
        grpo_ckpt_dir=grpo_ckpt,
        num_samples=num_samples,
        seed=seed,
        repo_id=repo_id,
        dataset_root=dataset_local,
        output_path=out_path,
        save_per_sample_rows=save_full_rows,
        use_holdout_split=use_holdout,
    )

    if not save_full_rows:
        slim = {
            "verdict_hint": report["verdict_hint"],
            "bc_summary": report["bc"]["summary"],
            "output_path": str(out_path),
            "grpo_ckpt_dir": grpo_ckpt,
            "dataset_repo": repo_id,
            "use_holdout_split": use_holdout,
        }
        if report.get("comparison") is not None:
            slim["comparison"] = report["comparison"]
        if report.get("grpo") is not None:
            slim["grpo_summary"] = report["grpo"]["summary"]
        report = slim

    cache_volume.commit()
    return report


@app.local_entrypoint()
def main(
    cmd: str = "smoke",
    group_size: int = 8,
    num_steps: int = 100,
    num_samples: int = 50,
    grpo_ckpt_dir: str | None = None,
):
    if cmd == "download":
        print(download_bc_checkpoint.remote())
        print(download_eval_dataset.remote())
    elif cmd == "train":
        print(train_grpo.remote(num_steps=num_steps, group_size=group_size))
    elif cmd == "compare":
        print(compare_fifty.remote(num_samples=num_samples, grpo_ckpt_dir=grpo_ckpt_dir))
    else:
        print(smoke_grpo_step.remote(group_size=group_size))
