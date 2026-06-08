"""Apply pi-drive openpi patches (driving config + lerobot fixes)."""

from __future__ import annotations

import glob
import os
import re
import shutil
from pathlib import Path

_PATCHES_DIR = Path(__file__).resolve().parent


def prepend_openpi_venv(openpi_dir: str | Path) -> str:
    """Put openpi's uv venv (lerobot, jax, flax, …) on sys.path and PYTHONPATH."""
    import sys

    openpi_dir = Path(openpi_dir)
    matches = glob.glob(str(openpi_dir / ".venv/lib/python*/site-packages"))
    if not matches:
        raise RuntimeError(
            f"No site-packages under {openpi_dir}/.venv — run `uv sync` in openpi first"
        )
    site_packages = matches[0]
    src = str(openpi_dir / "src")
    client_src = str(openpi_dir / "packages/openpi-client/src")
    # site-packages first (lerobot, openpi_client, jax, …), then openpi src.
    for p in (src, client_src, site_packages):
        if p not in sys.path:
            sys.path.insert(0, p)
    prev = os.environ.get("PYTHONPATH", "")
    parts = [site_packages]
    if Path(client_src).is_dir():
        parts.append(client_src)
    parts.append(src)
    if prev:
        parts.append(prev)
    os.environ["PYTHONPATH"] = ":".join(parts)
    return site_packages


def patch_openpi(openpi_dir: str | Path) -> None:
    """Patch a cloned openpi repo for Mark's pi05_driving BC / GRPO post-training."""
    openpi_dir = Path(openpi_dir)
    src = _PATCHES_DIR

    # 1. driving_policy.py
    dst_policy = openpi_dir / "src/openpi/policies/driving_policy.py"
    dst_policy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "driving_policy.py", dst_policy)

    # 2. gemma.py — gemma_2b_lora_driving variant
    gemma_path = openpi_dir / "src/openpi/models/gemma.py"
    content = gemma_path.read_text()
    if "gemma_2b_lora_driving" not in content:
        content = content.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]',
        )
        content = content.replace(
            '    if variant == "gemma_300m_lora":',
            '''    if variant == "gemma_2b_lora_driving":
        return Config(
            width=2048,
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},
        )
    if variant == "gemma_300m_lora":''',
        )
        gemma_path.write_text(content)

    # 3. config.py — pi05_driving TrainConfig
    config_path = openpi_dir / "src/openpi/training/config.py"
    content = config_path.read_text()
    if "pi05_driving" not in content:
        content = content.replace(
            "import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy",
        )
        driving_data_config = '''
@dataclasses.dataclass(frozen=True)
class LeRobotDrivingDataConfig(DataConfigFactory):
    """Data config for driving with pi0.5 (PhysicalAI-AV BC)."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[driving_policy.DrivingInputs(model_type=model_config.model_type)],
            outputs=[driving_policy.DrivingOutputs()],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )

'''
        content = content.replace(
            "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
            driving_data_config + "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
        )
        ckpt_path = os.environ.get(
            "PI05_BC_CHECKPOINT_PARAMS",
            "gs://openpi-assets/checkpoints/pi05_base/params",
        )
        driving_train_config = f'''
    #
    # PhysicalAI-AV driving BC (Mark checkpoint — override via PI05_BC_CHECKPOINT_PARAMS).
    #
    TrainConfig(
        name="pi05_driving",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=128,
            action_horizon=1,
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
        ),
        data=LeRobotDrivingDataConfig(
            repo_id="markmusic/pi05-physical-av-bc",
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "{ckpt_path}"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            action_dim=128,
            action_horizon=1,
            paligemma_variant="gemma_2b_lora_driving",
            action_expert_variant="gemma_300m",
        ).get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=750,
            peak_lr=3e-5,
            decay_steps=15_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(
            b1=0.9,
            b2=0.999,
            clip_gradient_norm=1.0,
        ),
        num_train_steps=15_000,
        batch_size=96,
        fsdp_devices=1,
        save_interval=500,
        log_interval=50,
        checkpoint_base_dir="/cache/checkpoints",
    ),
'''
        content = content.replace(
            "    *polaris_config.get_polaris_configs(),\n]",
            "    *polaris_config.get_polaris_configs()," + driving_train_config + "]",
        )
        config_path.write_text(content)

    _patch_lerobot_utils(openpi_dir)
    _patch_lerobot_dataset(openpi_dir)
    _patch_lerobot_tolerance(openpi_dir)
    _patch_lerobot_timestamps(openpi_dir)
    print(f"openpi patched at {openpi_dir}")


def _patch_lerobot_utils(openpi_dir: Path) -> None:
    for path in glob.glob(str(openpi_dir / ".venv/lib/python*/site-packages/lerobot/common/datasets/utils.py")):
        content = Path(path).read_text()
        if "PATCHED" not in content:
            content = content.replace(
                "raise ForwardCompatibilityError(repo_id, min(upper_versions))",
                "pass  # PATCHED: accept our dataset version",
            )
            Path(path).write_text(content)
        if "HF_TRANSFORM_PATCHED" in content:
            continue
        content = Path(path).read_text()
        content = content.replace(
            "items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]",
            "items_dict[key] = [x if isinstance(x, (str, dict)) else torch.tensor(x) for x in items_dict[key]]  # HF_TRANSFORM_PATCHED",
        )
        Path(path).write_text(content)


def _patch_lerobot_tolerance(openpi_dir: Path) -> None:
    """Relax timestamp checks (Mark's AV dataset has single-frame episodes)."""
    for path in glob.glob(
        str(openpi_dir / ".venv/lib/python*/site-packages/lerobot/common/datasets/lerobot_dataset.py")
    ):
        content = Path(path).read_text()
        if "TOLERANCE_PATCHED" in content:
            continue
        for old, new in (
            ("tolerance_s: float = 1e-4,", "tolerance_s: float = 1.0,  # TOLERANCE_PATCHED"),
            ("tolerance_s: float = 1e-4", "tolerance_s: float = 1.0  # TOLERANCE_PATCHED"),
            ("tolerance_s: float = 0.0001,", "tolerance_s: float = 1.0,  # TOLERANCE_PATCHED"),
            ("tolerance_s: float = 0.0001", "tolerance_s: float = 1.0  # TOLERANCE_PATCHED"),
        ):
            content = content.replace(old, new)
        Path(path).write_text(content)


def _patch_lerobot_timestamps(openpi_dir: Path) -> None:
    """Skip timestamp sync validation (single-frame episodes at ts=0)."""
    for path in glob.glob(
        str(openpi_dir / ".venv/lib/python*/site-packages/lerobot/common/datasets/utils.py")
    ):
        content = Path(path).read_text()
        if "TIMESTAMP_SYNC_PATCHED" in content:
            continue
        needle = "def check_timestamps_sync("
        if needle not in content:
            continue
        content = content.replace(
            needle,
            "def check_timestamps_sync(*_args, **_kwargs):  # TIMESTAMP_SYNC_PATCHED\n    return\n\n"
            "def _check_timestamps_sync_unused(",
            1,
        )
        Path(path).write_text(content)


def _patch_lerobot_dataset(openpi_dir: Path) -> None:
    for path in glob.glob(
        str(openpi_dir / ".venv/lib/python*/site-packages/lerobot/common/datasets/lerobot_dataset.py")
    ):
        ds_content = Path(path).read_text()
        if "DOWNLOAD_PATCHED" in ds_content:
            continue
        ds_content = ds_content.replace(
            "def download_episodes(self",
            """def download_episodes(self, download_videos=True):  # DOWNLOAD_PATCHED
        import os as _os
        _data_dir = _os.path.join(str(self.root), "data")
        _has_data = _os.path.exists(_data_dir) and len(_os.listdir(_data_dir)) > 0
        if _has_data:
            print(f"  DOWNLOAD_PATCHED: local data at {self.root}, skipping HF download")
            return
        return self._original_download_episodes(download_videos)

    def _original_download_episodes(self""",
        )
        ds_content = ds_content.replace(
            "assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())",
            "pass  # DOWNLOAD_PATCHED: images embedded in parquet",
        )
        Path(path).write_text(ds_content)


def link_dataset_to_hf_cache(cache_dir: str, repo_id: str) -> None:
    """Symlink local LeRobot dataset into HF hub cache."""
    local_dataset = Path(cache_dir) / "hf/lerobot" / repo_id
    if not local_dataset.exists():
        print(f"No local dataset at {local_dataset}")
        return
    hub_dir = Path(cache_dir) / "hf/hub" / f"datasets--{repo_id.replace('/', '--')}"
    snapshot_dir = hub_dir / "snapshots/local"
    if snapshot_dir.exists():
        return
    hub_dir.mkdir(parents=True, exist_ok=True)
    (hub_dir / "refs").mkdir(exist_ok=True)
    (hub_dir / "refs/main").write_text("local")
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(local_dataset, snapshot_dir)
    print(f"Linked {local_dataset} → HF cache")


def _compute_norm_stats_subset(
    openpi_dir: Path,
    cache_dir: Path,
    *,
    config_name: str,
    max_frames: int = 5000,
) -> Path | None:
    """Run openpi compute_norm_stats on a subset (checkpoint HF repo has no assets/)."""
    import subprocess

    asset_id = "markmusic/pi05-physical-av-bc"
    out_file = openpi_dir / "assets" / config_name / asset_id / "norm_stats.json"
    if out_file.is_file():
        return out_file

    env = {
        **os.environ,
        "HF_HOME": str(cache_dir / "hf"),
        "OPENPI_DIR": str(openpi_dir),
    }
    print(f"Computing norm stats from dataset ({max_frames} frames) → {out_file.parent}")
    proc = subprocess.run(
        [
            str(openpi_dir / ".venv/bin/python"),
            "scripts/compute_norm_stats.py",
            "--config-name",
            config_name,
            "--max-frames",
            str(max_frames),
        ],
        cwd=str(openpi_dir),
        env=env,
        text=True,
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout[-1500:])
    if proc.returncode != 0:
        print(proc.stderr[-1500:] if proc.stderr else "")
        return None
    return out_file if out_file.is_file() else None


def _write_bootstrap_norm_stats(path: Path, *, action_dim: int, state_dim: int) -> None:
    """Identity-ish stats so openpi can load the policy (replace with real stats when available)."""
    import json

    def block(dim: int) -> dict:
        z = [0.0] * dim
        o = [1.0] * dim
        return {"mean": z, "std": o, "q01": [-1.0] * dim, "q99": [1.0] * dim}

    payload = {"norm_stats": {"action": block(action_dim), "state": block(state_dim)}}
    path.write_text(json.dumps(payload, indent=2))


def ensure_driving_norm_stats(
    openpi_dir: str | Path,
    cache_dir: str | Path,
    ckpt_dir: str | Path,
    *,
    asset_id: str = "markmusic/pi05-physical-av-bc",
    config_name: str = "pi05_driving",
) -> Path:
    """Ensure norm_stats.json exists for pi05_driving (HF download or bootstrap)."""
    openpi_dir = Path(openpi_dir)
    cache_dir = Path(cache_dir)
    ckpt_dir = Path(ckpt_dir)

    def _stats_path(base: Path) -> Path:
        return base / "assets" / config_name / asset_id / "norm_stats.json"

    search = [
        _stats_path(cache_dir),
        _stats_path(openpi_dir),
        ckpt_dir / "assets" / asset_id / "norm_stats.json",
    ]
    path = None
    for candidate in search:
        if candidate.is_file():
            path = candidate
            break

    if path is None:
        # Mark's HF checkpoint is orbax-only (no assets/norm_stats.json on the hub).
        path = _compute_norm_stats_subset(openpi_dir, cache_dir, config_name=config_name)

    if path is None:
        found = cache_dir / "assets" / config_name / asset_id
        found.mkdir(parents=True, exist_ok=True)
        path = found / "norm_stats.json"
        print(
            f"WARNING: compute_norm_stats failed; using bootstrap norm stats at {path}. "
            "Re-run after dataset is on the volume for real stats."
        )
        _write_bootstrap_norm_stats(path, action_dim=128, state_dim=2)

    targets = [
        openpi_dir / "assets" / config_name / asset_id,
        ckpt_dir / "assets" / asset_id,
        cache_dir / "assets" / config_name / asset_id,
    ]
    for tgt in targets:
        tgt.mkdir(parents=True, exist_ok=True)
        link = tgt / "norm_stats.json"
        if link.exists() or link.is_symlink():
            continue
        link.symlink_to(path.resolve())
    print(f"Norm stats ready at {path}")
    return path
