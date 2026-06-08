"""Resolve Mark's BC LeRobot dataset on Modal volume — no HF calls for missing eval repo."""

from __future__ import annotations

from pathlib import Path

from grpo.constants import HF_BC_DATASET_REPO, HF_BC_EVAL_DATASET_REPO


def lerobot_root(cache_dir: str | Path, repo_id: str) -> Path:
    return Path(cache_dir) / "hf" / "lerobot" / repo_id


def is_valid_lerobot_dataset(path: str | Path) -> bool:
    """True if a complete LeRobot dataset on disk (not a stub / failed eval download)."""
    p = Path(path)
    if not (p / "meta" / "info.json").is_file():
        return False
    if not (p / "meta" / "episodes_stats.jsonl").is_file():
        return False
    data_dir = p / "data"
    return data_dir.is_dir() and any(data_dir.iterdir())


def resolve_bc_dataset(
    cache_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[str, str, bool]:
    """Return ``(local_path, repo_id, use_holdout_split)``.

    Public HuggingFace has **no** ``pi05-physical-av-bc-eval``. We only use that
    repo id when a *valid* copy exists on the volume; otherwise train repo + 15% holdout.
    Never probes the HF API for the eval repo.
    """
    from huggingface_hub import snapshot_download

    cache_dir = Path(cache_dir)
    train_repo = HF_BC_DATASET_REPO
    eval_repo = HF_BC_EVAL_DATASET_REPO
    train_local = lerobot_root(cache_dir, train_repo)
    eval_local = lerobot_root(cache_dir, eval_repo)

    if eval_local.exists() and not is_valid_lerobot_dataset(eval_local):
        print(f"Ignoring incomplete eval cache at {eval_local} (no meta/info.json)")

    if is_valid_lerobot_dataset(eval_local) and not force:
        print(f"Using cached eval dataset at {eval_local}")
        return str(eval_local), eval_repo, False

    if is_valid_lerobot_dataset(train_local) and not force:
        print(f"Using cached train dataset at {train_local} (15% holdout for clip splits)")
        return str(train_local), train_repo, True

    repo_id = train_repo
    local = train_local
    print(
        f"Downloading {repo_id} (public HF has no {eval_repo}; use holdout for eval splits)"
    )
    local.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=str(local))
    return str(local), repo_id, True
