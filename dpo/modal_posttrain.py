"""
DPO post-training on Mark's BC checkpoint — Alpamayo/AR1 picks best vs worst trajectory.

  modal run dpo/modal_posttrain.py::download_bc_checkpoint
  modal run dpo/modal_posttrain.py::smoke_dpo_step --group-size 8
  modal run --detach dpo/modal_posttrain.py::train_dpo --num-clips 50

AR1 labels (optional, improves preference quality):
  modal run grpo/modal_ar1_labels.py::smoke_one_clip
  → /cache/labels/ar1_labels.jsonl
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_PKG_ROOT = "/app/pi_05_drives"
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import modal

from dpo.constants import (
    HF_BC_CHECKPOINT_REPO,
    HF_BC_DATASET_REPO,
    HF_BC_EVAL_DATASET_REPO,
    OPENPI_CONFIG_NAME,
)
from grpo.modal_posttrain import (
    CACHE_DIR,
    CKPT_LOCAL,
    OPENPI_DIR,
    ensure_bc_checkpoint,
    ensure_bc_dataset,
)

APP_NAME = "pi05-dpo-posttrain"
_PKG_DIR = Path(__file__).resolve().parents[1]  # repo root (pi_05_drives)
LABELS_PATH = f"{CACHE_DIR}/labels/ar1_labels.jsonl"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    .pip_install("uv", "huggingface_hub", "numpy")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
    )
    .env({"PYTHONPATH": "/app/pi_05_drives:/opt/openpi/src"})
    .add_local_dir(
        _PKG_DIR, remote_path="/app/pi_05_drives", copy=True,
        ignore=[
            "checkpoints", "downloaded_ckpt", "caddy_lerobot", "caddy_zero_shot_full",
            "caddy_zero_shot_out", "inference_batch_val", "inference_output",
            "viz", "viz_images", "viz_local", ".git", "__pycache__",
            "*.mp4", "*.png", "*.pyc", ".DS_Store",
        ],
    )
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}
app = modal.App(APP_NAME)


def _setup_openpi_env(ckpt_params_dir: str) -> dict[str, str]:
    from openpi_patches.patch_openpi import prepend_openpi_venv

    prepend_openpi_venv(OPENPI_DIR)
    params_dir = ckpt_params_dir if ckpt_params_dir.rstrip("/").endswith("params") else f"{ckpt_params_dir}/params"
    return {
        **os.environ,
        "PI05_BC_CHECKPOINT_PARAMS": params_dir,
        "HF_HOME": f"{CACHE_DIR}/hf",
        "OPENPI_DIR": OPENPI_DIR,
        "AR1_LABELS_PATH": LABELS_PATH,
        "HF_HUB_DISABLE_XET": "1",
        "XLA_PYTHON_CLIENT_ALLOCATOR": "platform",  # on-demand alloc; frees GPU promptly (fixes progressive OOM)
        "PYTHONUNBUFFERED": "1",
    }


@app.function(
    image=image,
    timeout=60 * 60 * 4,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_bc_checkpoint(force: bool = False) -> str:
    path = ensure_bc_checkpoint(CACHE_DIR, force=force)
    cache_volume.commit()
    return path


@app.function(
    image=image,
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def download_bc_dataset(force: bool = False) -> str:
    local, _repo, _ = ensure_bc_dataset(CACHE_DIR, force=force)
    cache_volume.commit()
    return local


def _prepare_openpi(ckpt_dir: str, repo_id: str) -> None:
    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        ensure_driving_norm_stats,
        link_dataset_to_hf_cache,
        patch_openpi,
        prepend_openpi_venv,
    )

    # Bake Mark's BC checkpoint path BEFORE patching (else config falls back to the
    # 32-dim gs://.../pi05_base base model). See grpo/_prepare_dataset_and_assets.
    os.environ["PI05_BC_CHECKPOINT_PARAMS"] = f"{ckpt_dir}/params"
    from grpo.openpi_grpo_runner import _ensure_checkpoint_layout

    _ensure_checkpoint_layout(ckpt_dir, OPENPI_DIR)
    patch_openpi(OPENPI_DIR)
    prepend_openpi_venv(OPENPI_DIR)
    link_dataset_to_hf_cache(CACHE_DIR, repo_id)
    ensure_driving_norm_stats(OPENPI_DIR, CACHE_DIR, ckpt_dir)


def _run_dpo_script(
    *,
    ckpt_dir: str,
    group_size: int,
    dataset_index: int,
    exp_name: str,
) -> dict:
    """Single-clip smoke via subprocess."""
    _prepare_openpi(ckpt_dir, HF_BC_DATASET_REPO)

    runner = Path("/app/pi_05_drives/dpo/openpi_dpo_runner.py")
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
        "--num-steps",
        "0",
        "--exp-name",
        exp_name,
        "--labels-path",
        LABELS_PATH,
    ]
    env = _setup_openpi_env(ckpt_dir)
    proc = subprocess.run(cmd, cwd=OPENPI_DIR, env=env, text=True, capture_output=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"dpo runner failed rc={proc.returncode}")

    report_path = Path(CACHE_DIR) / "dpo_reports" / f"{exp_name}_last.json"
    if report_path.exists():
        with report_path.open() as f:
            return json.load(f)
    return {"status": "ok", "stdout_tail": proc.stdout[-2000:]}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def smoke_dpo_step(group_size: int = 8, dataset_index: int = 0) -> dict:
    """Sample G trajectories; AR1 reference picks winner/loser; DPO loss."""
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    ensure_bc_dataset(CACHE_DIR)
    return _run_dpo_script(
        ckpt_dir=ckpt,
        group_size=group_size,
        dataset_index=dataset_index,
        exp_name="dpo-smoke",
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
def train_dpo(
    num_clips: int = 50,
    seed: int = 42,
    group_size: int = 8,
    exp_name: str = "dpo-posttrain",
) -> dict:
    """DPO post-train on N clips: sample G, AR1 picks best/worst, DPO loss per clip."""
    from pathlib import Path

    from dpo.openpi_dpo_train import train_on_clips

    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, use_holdout = ensure_bc_dataset(CACHE_DIR)
    _prepare_openpi(ckpt, HF_BC_DATASET_REPO)

    out_path = Path(CACHE_DIR) / "dpo_reports" / f"{exp_name}_clips{num_clips}_seed{seed}.json"
    report = train_on_clips(
        openpi_dir=OPENPI_DIR,
        ckpt_dir=ckpt,
        dataset_root=dataset_local,
        repo_id=repo_id,
        num_clips=num_clips,
        seed=seed,
        group_size=group_size,
        labels_path=LABELS_PATH if Path(LABELS_PATH).exists() else None,
        use_holdout_split=use_holdout,
        exp_name=exp_name,
        output_path=out_path,
    )
    cache_volume.commit()
    return {
        "summary": report["summary"],
        "num_clips": report["num_clips"],
        "indices": report["indices"],
        "output_path": str(out_path),
        "dataset_repo": repo_id,
        "dataset_root": dataset_local,
        "use_holdout_split": use_holdout,
    }


def _run_dpo_train_script(
    *,
    ckpt_dir: str,
    dataset_root: str,
    num_clips: int,
    group_size: int,
    save_interval: int,
    seed: int,
    beta: float,
    peak_lr: float,
    exp_name: str,
    out_dir: str,
    resume_from: str | None = None,
) -> dict:
    """Subprocess the real DPO training loop inside the openpi venv."""
    runner = Path("/app/pi_05_drives/dpo/dpo_train_real.py")
    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", str(runner),
        "--ckpt-dir", ckpt_dir,
        "--dataset-root", dataset_root,
        "--out-dir", out_dir,
        "--num-clips", str(num_clips),
        "--group-size", str(group_size),
        "--save-interval", str(save_interval),
        "--seed", str(seed),
        "--beta", str(beta),
        "--peak-lr", str(peak_lr),
        "--exp-name", exp_name,
    ]
    if resume_from:
        cmd += ["--resume-from", resume_from]
    if Path(LABELS_PATH).exists():
        cmd.extend(["--labels-path", LABELS_PATH])
    env = _setup_openpi_env(ckpt_dir)
    env["LEROBOT_DATASET_ROOT"] = dataset_root
    # Stream live to the Modal logs (don't buffer) so long runs are monitorable.
    proc = subprocess.run(cmd, cwd=OPENPI_DIR, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"dpo train runner failed rc={proc.returncode}")

    cache_volume.commit()
    report_path = Path(CACHE_DIR) / "dpo_reports" / f"{exp_name}_train.json"
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
def smoke_dpo_train(group_size: int = 4) -> dict:
    """Cheap end-to-end check: build TrainState, 2 real DPO gradient steps, no save."""
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, _ = ensure_bc_dataset(CACHE_DIR)
    _prepare_openpi(ckpt, repo_id)
    return _run_dpo_train_script(
        ckpt_dir=ckpt, dataset_root=dataset_local, num_clips=2, group_size=group_size,
        save_interval=0, seed=42, beta=100.0, peak_lr=1e-6, exp_name="dpo-train-smoke",
        out_dir=f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/dpo-train-smoke",
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    memory=80 * 1024,
)
def train_dpo_real(
    num_clips: int = 50,
    group_size: int = 8,
    save_interval: int = 50,
    seed: int = 42,
    beta: float = 100.0,
    peak_lr: float = 1e-6,
    exp_name: str = "dpo-posttrain",
    resume_from: str | None = None,
) -> dict:
    """Real DPO post-training: Alpamayo preference + gradient + optax + orbax checkpoint.

    Produces /cache/checkpoints/pi05_driving/<exp_name>/<num_clips> loadable by eval_fifty.

    To resume from a prior checkpoint on a FRESH container (sidesteps the per-process
    GPU memory leak that OOMs DPO ~step 1100), pass --resume-from with a step dir, e.g.:
      modal run --detach dpo/modal_posttrain.py::train_dpo_real --num-clips 1000 \\
        --resume-from /cache/checkpoints/pi05_driving/dpo-5k-v2/1000 --exp-name dpo-5k-v2
    Checkpoints are named by GLOBAL step (resume 1000 + 1000 more -> saves at 2000).
    """
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, _ = ensure_bc_dataset(CACHE_DIR)
    _prepare_openpi(ckpt, repo_id)
    return _run_dpo_train_script(
        ckpt_dir=ckpt, dataset_root=dataset_local, num_clips=num_clips, group_size=group_size,
        save_interval=save_interval, seed=seed, beta=beta, peak_lr=peak_lr,
        exp_name=exp_name,
        out_dir=f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/{exp_name}",
        resume_from=resume_from,
    )


def _run_dpo_cosmos_script(
    *,
    ckpt_dir: str,
    dataset_root: str,
    labels_path: str,
    stage_dir: str,
    num_steps: int,
    save_interval: int,
    seed: int,
    beta: float,
    peak_lr: float,
    exp_name: str,
    out_dir: str,
    anchor_lambda: float = 0.0,
) -> dict:
    """Subprocess the Cosmos-DPO (Stage C) training loop inside the openpi venv."""
    runner = Path("/app/pi_05_drives/dpo/dpo_train_cosmos.py")
    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", str(runner),
        "--ckpt-dir", ckpt_dir,
        "--dataset-root", dataset_root,
        "--labels-path", labels_path,
        "--stage-dir", stage_dir,
        "--out-dir", out_dir,
        "--num-steps", str(num_steps),
        "--save-interval", str(save_interval),
        "--seed", str(seed),
        "--beta", str(beta),
        "--peak-lr", str(peak_lr),
        "--anchor-lambda", str(anchor_lambda),
        "--exp-name", exp_name,
    ]
    env = _setup_openpi_env(ckpt_dir)
    env["LEROBOT_DATASET_ROOT"] = dataset_root
    proc = subprocess.run(cmd, cwd=OPENPI_DIR, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"cosmos dpo runner failed rc={proc.returncode}")
    cache_volume.commit()
    report_path = Path(CACHE_DIR) / "dpo_reports" / f"{exp_name}_train.json"
    if report_path.exists():
        with report_path.open() as f:
            return json.load(f)
    return {"status": "ok", "note": "streamed to logs"}


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def train_dpo_cosmos(
    tag: str = "cosmos_g2",
    num_steps: int = 5000,
    save_interval: int = 500,
    seed: int = 42,
    beta: float = 100.0,
    peak_lr: float = 1e-6,
    exp_name: str = "dpo-cosmos-5k",
    labels_name: str = "",
    anchor_lambda: float = 0.0,
) -> dict:
    """Stage C: DPO ``num_steps`` on Cosmos-judged pairs from /cache/labels/cosmos3_<tag>.jsonl.

    Requires Stage A (gen_judge_candidates --tag <tag>) + Stage B
    (dpo/modal_cosmos_judge.py::judge_candidates --tag <tag>) to have run first.
    ``labels_name`` overrides the labels file (e.g. the debiased cosmos3_<tag>_db.jsonl)
    while still reading candidate npz from cosmos_stage/<tag>/cands.
    """
    ckpt = ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, _ = ensure_bc_dataset(CACHE_DIR)
    _prepare_openpi(ckpt, repo_id)
    labels_path = f"{CACHE_DIR}/labels/{labels_name}" if labels_name else f"{CACHE_DIR}/labels/cosmos3_{tag}.jsonl"
    return _run_dpo_cosmos_script(
        ckpt_dir=ckpt, dataset_root=dataset_local,
        labels_path=labels_path,
        stage_dir=f"{CACHE_DIR}/cosmos_stage/{tag}",
        num_steps=num_steps, save_interval=save_interval, seed=seed, beta=beta, peak_lr=peak_lr,
        exp_name=exp_name, anchor_lambda=anchor_lambda,
        out_dir=f"{CACHE_DIR}/checkpoints/{OPENPI_CONFIG_NAME}/{exp_name}",
    )


@app.function(
    image=image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=80 * 1024,
)
def eval_fifty(
    num_samples: int = 50,
    seed: int = 42,
    group_size: int = 8,
    save_full_rows: bool = False,
) -> dict:
    """AR1 preference + DPO loss on N holdout samples (BC policy, one load)."""
    from pathlib import Path

    sys.path.insert(0, "/app/pi_05_drives")
    from openpi_patches.patch_openpi import (
        ensure_driving_norm_stats,
        link_dataset_to_hf_cache,
        patch_openpi,
        prepend_openpi_venv,
    )
    from dpo.openpi_dpo_eval import eval_fifty as run_eval_fifty

    patch_openpi(OPENPI_DIR)
    prepend_openpi_venv(OPENPI_DIR)
    ensure_bc_checkpoint(CACHE_DIR)
    dataset_local, repo_id, use_holdout = ensure_bc_dataset(CACHE_DIR)
    link_dataset_to_hf_cache(CACHE_DIR, repo_id)
    ensure_driving_norm_stats(OPENPI_DIR, CACHE_DIR, CKPT_LOCAL)

    out_path = Path(CACHE_DIR) / "dpo_eval" / f"eval_{num_samples}_seed{seed}.json"
    report = run_eval_fifty(
        openpi_dir=OPENPI_DIR,
        ckpt_dir=CKPT_LOCAL,
        num_samples=num_samples,
        seed=seed,
        group_size=group_size,
        dataset_root=dataset_local,
        labels_path=LABELS_PATH if Path(LABELS_PATH).exists() else None,
        use_holdout_split=use_holdout,
        output_path=out_path,
    )

    if not save_full_rows:
        report = {
            "summary": report["summary"],
            "num_samples": report["num_samples"],
            "indices": report["indices"],
            "dataset_repo": repo_id,
            "use_holdout_split": use_holdout,
            "output_path": str(out_path),
            "ckpt_dir": CKPT_LOCAL,
        }

    cache_volume.commit()
    return report


@app.local_entrypoint()
def main(
    cmd: str = "train",
    group_size: int = 8,
    num_clips: int = 50,
    seed: int = 42,
):
    if cmd == "download":
        print(download_bc_checkpoint.remote())
        print(download_bc_dataset.remote())
    elif cmd == "smoke":
        print(smoke_dpo_step.remote(group_size=group_size))
    else:
        print(train_dpo.remote(num_clips=num_clips, seed=seed, group_size=group_size))
