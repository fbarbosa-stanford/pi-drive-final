"""π0.5 behavior cloning training on Modal (8× H100).

Trains π0.5 on PhysicalAI-AV ground truth driving data using the
pi05_driving config in openpi. LoRA on VLM + full fine-tune on action expert.

Usage:
    # Pre-flight: validate HF push works
    modal run pi05/modal_train_bc.py::validate_hf

    # Fresh download dataset from HF (needed once, gets all parquets)
    modal run --detach pi05/modal_train_bc.py::fresh_download

    # Consolidate 10K parquets into 10 chunk files (needed once)
    modal run --detach pi05/modal_train_bc.py::consolidate_dataset

    # Train (detached so laptop can close)
    modal run --detach pi05/modal_train_bc.py::train_bc --num-steps 5000

    # During training — on-demand checkpoint save:
    modal run pi05/modal_train_bc.py::trigger_save

    # During training — save checkpoint + push to HuggingFace:
    modal run pi05/modal_train_bc.py::trigger_push_hf

    # After training — upload a specific checkpoint:
    modal run pi05/modal_train_bc.py::upload_checkpoint --step 5000

    # Run diagnostic to test dataset loading:
    modal run --detach pi05/modal_train_bc.py::diagnose_dataset

Dataset issues solved (2026-05-30):
    1. HF API rate limits (429): LeRobot's snapshot_download makes per-file API calls
       to check ETags. With 10K files, hits 1000 req/5min rate limit. Fixed by patching
       download_episodes to skip when local data exists on the volume.

    2. No images/ directory: Images are embedded directly in parquet files (not stored
       as separate PNGs). The LeRobotDataset __init__ assertion checks for separate
       image files — patched to skip this check.

    3. 10K individual parquets too slow: Each episode stored as 1 parquet file (~300KB).
       Loading 10K files on Modal FUSE is very slow. Fixed by consolidating into 10
       chunk-level files (~300MB each). Must commit volume after each chunk or progress
       is lost on timeout.

    4. HF-generated file-*.parquet conflicts: Git-cloning from HF includes auto-generated
       file-000.parquet files that are incompatible with our episode format. Cleaned up
       during consolidation.

    5. Dataset not on volume: Original dataset was built on a different container and
       only pushed to HF. Used git clone (bypasses per-file rate limits) to download
       fresh copy to the volume.
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-train-bc"
CACHE_DIR = "/cache"
OPENPI_DIR = "/opt/openpi"

# ---------------------------------------------------------------------------
# Image: openpi (JAX/Flax) + our driving config patches
# ---------------------------------------------------------------------------

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "git-lfs", "build-essential", "clang")
    .pip_install("uv")
    .run_commands(
        f"GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git {OPENPI_DIR}",
        f"cd {OPENPI_DIR} && uv sync",
    )
    .pip_install("huggingface_hub", "wandb", "pyarrow", "pandas")
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/hf",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        }
    )
)

# ---------------------------------------------------------------------------
# Volumes & App
# ---------------------------------------------------------------------------

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}

app = modal.App(APP_NAME)

TRIGGER_DIR = f"{CACHE_DIR}/triggers"
HF_CHECKPOINT_REPO = "markmusic/pi05-driving-bc-v2-checkpoint"

build_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pandas", "pyarrow", "numpy")
)


# ---------------------------------------------------------------------------
# Build LeRobot datasets on CPU (run before training to avoid GPU waste)
# ---------------------------------------------------------------------------


@app.function(
    image=build_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 2,
    memory=32 * 1024,
)
def _build_chunk(
    chunk_idx: int,
    start: int,
    end: int,
    split_name: str,
    scale: str,
    repo_id: str,
    tasks_json: str,
) -> dict:
    """Build one parquet chunk of the LeRobot dataset. Called in parallel.

    Reads its own data slice from parquet + images on the volume — no data
    passed through RPC (which caused OOM on the coordinator last time).
    """
    import json
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = time.time()
    task_to_idx = json.loads(tasks_json)

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    local_path = f"{CACHE_DIR}/hf/lerobot/{repo_id}"

    df = pd.read_parquet(f"{output_dir}/samples.parquet")
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    chunk_df = split_df.iloc[start:end]
    del df, split_df

    chunk_dir = f"{local_path}/data/chunk-{chunk_idx:03d}"
    os.makedirs(chunk_dir, exist_ok=True)

    print(f"[Chunk {chunk_idx}] Reading {len(chunk_df)} images...")

    def read_image(img_path):
        with open(f"{output_dir}/{img_path}", "rb") as f:
            return f.read()

    with ThreadPoolExecutor(max_workers=32) as executor:
        image_bytes_list = list(executor.map(read_image, chunk_df["image_path"]))

    print(f"[Chunk {chunk_idx}] Images read, building Arrow table...")

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    image_array = pa.array(
        [{"bytes": b, "path": None} for b in image_bytes_list],
        type=image_type,
    )
    del image_bytes_list

    states = list(zip(
        chunk_df["speed"].astype(float).tolist(),
        chunk_df["heading_rate"].astype(float).tolist(),
    ))
    actions_flat = [
        np.array(a, dtype=np.float32).flatten().tolist()
        for a in chunk_df["actions"]
    ]
    task_indices = chunk_df["nav_prompt"].map(task_to_idx).tolist()

    n = end - start
    table = pa.table({
        "observation.images.front": image_array,
        "observation.state": pa.array(states, type=pa.list_(pa.float32())),
        "action": pa.array(actions_flat, type=pa.list_(pa.float32())),
        "episode_index": pa.array(range(start, end), type=pa.int64()),
        "frame_index": pa.array([0] * n, type=pa.int64()),
        "index": pa.array(range(start, end), type=pa.int64()),
        "timestamp": pa.array([0.0] * n, type=pa.float64()),
        "task_index": pa.array(task_indices, type=pa.int64()),
    })

    pq.write_table(table, f"{chunk_dir}/episode_000000.parquet")
    del table, image_array, actions_flat
    cache_volume.commit()

    elapsed = time.time() - t0
    print(f"[Chunk {chunk_idx}] {n} samples in {elapsed:.0f}s ({n/elapsed:.0f} samples/s)")
    return {"chunk_idx": chunk_idx, "n_samples": n, "elapsed": elapsed}


@app.function(
    image=build_image,
    volumes=VOLUMES,
    timeout=60 * 60 * 2,
    memory=8 * 1024,
)
def build_datasets(scale: str = "xlarge"):
    """Build LeRobot datasets from extracted data on CPU (fast, parallel, no GPU).

    Each worker reads its own slice from the parquet — coordinator only passes
    lightweight args (indices, paths, task map).

    Usage:
        modal run --detach pi05/modal_train_bc.py::build_datasets --scale xlarge
    """
    import json
    import os
    import shutil
    import time

    import pandas as pd

    TRAIN_REPO = "markmusic/pi05-physical-av-bc"
    EVAL_REPO = "markmusic/pi05-physical-av-bc-eval"

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    samples_path = f"{output_dir}/samples.parquet"
    train_path = f"{CACHE_DIR}/hf/lerobot/{TRAIN_REPO}"
    eval_path = f"{CACHE_DIR}/hf/lerobot/{EVAL_REPO}"

    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"No extracted data at {samples_path}")

    if os.path.exists(f"{train_path}/.built") and os.path.exists(f"{eval_path}/.built"):
        print("LeRobot datasets already built")
        return

    df = pd.read_parquet(samples_path, columns=["split", "nav_prompt"])
    print(f"Loaded {len(df)} total samples")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

    tasks = sorted(df["nav_prompt"].unique().tolist())
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    tasks_json = json.dumps(task_to_idx)
    print(f"Tasks ({len(tasks)}): {tasks}")

    CHUNK_SIZE = 20_000

    for split_name, repo_id, local_path in [
        ("train", TRAIN_REPO, train_path),
        ("eval", EVAL_REPO, eval_path),
    ]:
        marker = f"{local_path}/.built"
        if os.path.exists(marker):
            print(f"{split_name} already built")
            continue

        split_df = df[df["split"] == split_name].reset_index(drop=True)
        n_split = len(split_df)
        print(f"\nBuilding {split_name}: {n_split} samples -> {repo_id}")

        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        os.makedirs(local_path, exist_ok=True)

        t0 = time.time()
        n_chunks = (n_split + CHUNK_SIZE - 1) // CHUNK_SIZE

        chunk_args = []
        for ci in range(n_chunks):
            s = ci * CHUNK_SIZE
            e = min(s + CHUNK_SIZE, n_split)
            chunk_args.append((ci, s, e, split_name, scale, repo_id, tasks_json))

        print(f"  Launching {n_chunks} parallel chunk builders...")
        results = list(_build_chunk.starmap(chunk_args))

        total_samples = sum(r["n_samples"] for r in results)
        print(f"  All chunks done — {total_samples} samples")

        cache_volume.reload()
        meta_dir = f"{local_path}/meta"
        os.makedirs(meta_dir, exist_ok=True)

        info = {
            "codebase_version": "v2.1",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "robot_type": "cart_fsd",
            "fps": 10,
            "total_episodes": n_split,
            "total_frames": n_split,
            "total_tasks": len(tasks),
            "total_chunks": n_chunks,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"[0:{n_split}]"},
            "features": {
                "observation.images.front": {
                    "dtype": "image",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["speed", "heading_rate"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [128],
                    "names": None,
                },
            },
        }
        with open(f"{meta_dir}/info.json", "w") as f:
            json.dump(info, f, indent=2)

        with open(f"{meta_dir}/tasks.jsonl", "w") as f:
            for task in tasks:
                f.write(json.dumps({"task_index": task_to_idx[task], "task": task}) + "\n")

        split_df_full = pd.read_parquet(samples_path, columns=["split", "nav_prompt"])
        split_df_full = split_df_full[split_df_full["split"] == split_name].reset_index(drop=True)
        with open(f"{meta_dir}/episodes.jsonl", "w") as f:
            for i in range(n_split):
                t_idx = task_to_idx[split_df_full.iloc[i]["nav_prompt"]]
                f.write(json.dumps({"episode_index": i, "length": 1, "task_index": t_idx}) + "\n")
        del split_df_full

        with open(marker, "w") as f:
            f.write(f"{n_split} samples")
        with open(f"{local_path}/.consolidated", "w") as f:
            f.write("done")

        cache_volume.commit()
        elapsed = time.time() - t0
        print(f"Built {split_name}: {n_split} samples in {elapsed:.0f}s")

    print("\nBUILD COMPLETE")


# ---------------------------------------------------------------------------
# Pre-flight: validate HF push works before spending GPU money
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=300,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
)
def validate_hf():
    """Test HF token + repo access. Run this before training."""
    import os
    import tempfile

    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()
    print(f"HF token valid: logged in as {user['name']}")

    api.create_repo(
        HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True
    )
    print(f"Repo {HF_CHECKPOINT_REPO} exists and is writable")

    test_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, dir="/tmp"
    )
    test_file.write("preflight check")
    test_file.close()
    api.upload_file(
        path_or_fileobj=test_file.name,
        path_in_repo=".preflight_test",
        repo_id=HF_CHECKPOINT_REPO,
        repo_type="model",
        commit_message="preflight validation",
    )
    os.unlink(test_file.name)
    api.delete_file(
        ".preflight_test",
        repo_id=HF_CHECKPOINT_REPO,
        repo_type="model",
        commit_message="remove preflight test",
    )
    print("HF push validated — write + delete succeeded")
    return True


# ---------------------------------------------------------------------------
# Trigger functions: on-demand checkpoint save / HF push
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60,
    volumes=VOLUMES,
)
def trigger_save():
    """Write a trigger file that tells the training loop to save a checkpoint NOW."""
    import os
    import pathlib

    pathlib.Path(TRIGGER_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{TRIGGER_DIR}/save_now").touch()
    cache_volume.commit()
    print("Trigger written: training will save a checkpoint at the next step check")
    return "save_now trigger queued"


@app.function(
    image=train_image,
    timeout=60,
    volumes=VOLUMES,
)
def trigger_push_hf():
    """Write a trigger that saves a checkpoint AND pushes it to HuggingFace."""
    import os
    import pathlib

    pathlib.Path(TRIGGER_DIR).mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"{TRIGGER_DIR}/save_now").touch()
    pathlib.Path(f"{TRIGGER_DIR}/push_hf").touch()
    cache_volume.commit()
    print("Trigger written: training will save + push checkpoint to HF")
    return "push_hf trigger queued"


# ---------------------------------------------------------------------------
# Consolidate dataset: merge 10K tiny parquets → 10 larger ones
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60 * 60,
    volumes=VOLUMES,
    memory=32 * 1024,
)
def consolidate_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Merge per-episode parquet files into chunk-level files for fast loading.

    Commits after each chunk so progress survives timeouts.
    """
    import glob
    import os
    import shutil
    import tempfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    data_dir = os.path.join(dataset_dir, "data")
    marker = os.path.join(dataset_dir, ".consolidated")

    if os.path.exists(marker):
        print("Dataset already consolidated")
        return

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"No data directory at {data_dir}")

    chunk_dirs = sorted(glob.glob(os.path.join(data_dir, "chunk-*")))
    print(f"Consolidating {len(chunk_dirs)} chunks...")

    for chunk_dir in chunk_dirs:
        episode_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
        if len(episode_files) <= 1:
            print(f"  {os.path.basename(chunk_dir)}: already consolidated")
            continue

        print(f"  {os.path.basename(chunk_dir)}: merging {len(episode_files)} files...", end=" ", flush=True)

        # Read and merge on local /tmp (fast), then copy back to volume
        with tempfile.TemporaryDirectory() as tmpdir:
            tables = []
            for f in episode_files:
                try:
                    tables.append(pq.read_table(f))
                except Exception as e:
                    print(f"\n    WARNING: skipping corrupt file {os.path.basename(f)}: {e}")

            merged = pa.concat_tables(tables)
            tmp_out = os.path.join(tmpdir, "episode_000000.parquet")
            pq.write_table(merged, tmp_out)

            # Delete originals, copy merged file
            for f in episode_files:
                os.remove(f)
            shutil.copy2(tmp_out, os.path.join(chunk_dir, "episode_000000.parquet"))

        print(f"{len(merged)} rows", flush=True)
        cache_volume.commit()

    # Remove HuggingFace auto-generated file-*.parquet files that conflict
    for chunk_dir in chunk_dirs:
        for extra in glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
            print(f"  Removing HF-generated {os.path.basename(chunk_dir)}/{os.path.basename(extra)}")
            os.remove(extra)

    with open(marker, "w") as f:
        f.write("done")
    cache_volume.commit()
    print("Consolidation complete")


# ---------------------------------------------------------------------------
# Fresh download: git clone dataset from HF to get all files incl. images
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    timeout=60 * 60 * 2,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def fresh_download(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Git-clone dataset from HuggingFace to get all files including images.

    Uses git-lfs batch API which avoids per-file rate limits.
    Downloads to /cache/hf/lerobot/<repo_id>-fresh, then replaces the original.
    """
    import os
    import shutil
    import subprocess

    fresh_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}-fresh"
    final_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"

    # Get HF token for private repo
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    if not hf_token:
        raise RuntimeError("No HF token found — set HUGGING_FACE_HUB_TOKEN secret")

    clone_url = f"https://user:{hf_token}@huggingface.co/datasets/{repo_id}"

    # Remove stale fresh dir from any previous attempt
    if os.path.exists(fresh_dir):
        print(f"Removing previous fresh download at {fresh_dir}")
        shutil.rmtree(fresh_dir)

    print(f"Git-cloning {repo_id} to {fresh_dir}...")
    print("This downloads all files including 10K images via git-lfs batch API")
    result = subprocess.run(
        ["git", "lfs", "install"],
        cwd="/tmp",
        capture_output=True,
        text=True,
    )
    print(f"git lfs install: {result.stdout.strip()}")

    result = subprocess.run(
        ["git", "clone", "--depth=1", clone_url, fresh_dir],
        text=True,
        timeout=60 * 90,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Git clone failed (rc={result.returncode})")

    # Verify we got what we need
    for subdir in ["data", "meta"]:
        path = os.path.join(fresh_dir, subdir)
        if os.path.exists(path):
            n_files = sum(1 for _ in os.scandir(path))
            print(f"  {subdir}/: {n_files} entries")
        else:
            print(f"  WARNING: no {subdir}/ directory!")

    images_dir = os.path.join(fresh_dir, "images")
    if os.path.exists(images_dir):
        n_imgs = 0
        for root, dirs, files in os.walk(images_dir):
            n_imgs += len(files)
        print(f"  images/: {n_imgs} files")
    else:
        print("  WARNING: no images/ directory!")

    # Check data chunk structure
    data_dir = os.path.join(fresh_dir, "data")
    if os.path.exists(data_dir):
        for chunk in sorted(os.listdir(data_dir)):
            chunk_path = os.path.join(data_dir, chunk)
            if os.path.isdir(chunk_path):
                files = os.listdir(chunk_path)
                print(f"  data/{chunk}: {len(files)} files")

    # Rename old dir (preserve it, don't delete per user instructions)
    backup_dir = f"{final_dir}-old"
    if os.path.exists(final_dir):
        if os.path.exists(backup_dir):
            print(f"Removing previous backup at {backup_dir}")
            shutil.rmtree(backup_dir)
        print(f"Moving old dataset to {backup_dir}")
        os.rename(final_dir, backup_dir)

    # Move fresh download into place
    os.rename(fresh_dir, final_dir)
    print(f"Fresh dataset installed at {final_dir}")

    # Remove git metadata to save space (not needed for training)
    git_dir = os.path.join(final_dir, ".git")
    if os.path.exists(git_dir):
        shutil.rmtree(git_dir)
        print("Removed .git directory")

    cache_volume.commit()
    print("Done! Dataset ready with all images.")
    return 0


# ---------------------------------------------------------------------------
# Diagnostic: quickly test dataset loading to find hangs
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="H100",
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=32 * 1024,
)
def diagnose_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Test dataset loading step-by-step to find where it hangs."""
    import subprocess
    import time

    _patch_openpi()
    _consolidate_local_dataset(repo_id)
    _link_dataset_to_hf_cache(repo_id)

    script = f'''
import time, sys, os, glob
sys.stdout.reconfigure(line_buffering=True)

print("[1/8] Importing modules...")
t0 = time.time()
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
print(f"  Imports done in {{time.time()-t0:.1f}}s")

print("[2/8] Checking local dataset on volume...")
local_path = "/cache/hf/lerobot/markmusic/pi05-physical-av-bc"
if os.path.exists(local_path):
    dirs = os.listdir(local_path)
    print(f"  Local dataset found at {{local_path}}")
    print(f"  Contents: {{dirs[:20]}}")
    meta_path = os.path.join(local_path, "meta")
    if os.path.exists(meta_path):
        meta_files = os.listdir(meta_path)
        print(f"  meta/: {{meta_files}}")
        info_path = os.path.join(meta_path, "info.json")
        if os.path.exists(info_path):
            import json
            with open(info_path) as f:
                info = json.load(f)
            print(f"  info.json: version={{info.get('codebase_version')}}, total_episodes={{info.get('total_episodes')}}")
    data_path = os.path.join(local_path, "data")
    if os.path.exists(data_path):
        chunks = os.listdir(data_path)
        print(f"  data/: {{chunks[:10]}}")
        for chunk in chunks[:2]:
            chunk_path = os.path.join(data_path, chunk)
            if os.path.isdir(chunk_path):
                files = os.listdir(chunk_path)
                print(f"    {{chunk}}/: {{len(files)}} files, first={{files[:3]}}")
    img_path = os.path.join(local_path, "images")
    if os.path.exists(img_path):
        n_imgs = sum(1 for _ in glob.iglob(os.path.join(img_path, "**/*.png"), recursive=True))
        print(f"  images/: {{n_imgs}} PNG files")
else:
    print(f"  No local dataset at {{local_path}}")

print("[3/8] Checking HF hub cache symlink...")
hub_dir = "/cache/hf/hub/datasets--markmusic--pi05-physical-av-bc"
snapshot_dir = os.path.join(hub_dir, "snapshots/local")
refs_file = os.path.join(hub_dir, "refs/main")
if os.path.exists(snapshot_dir):
    is_link = os.path.islink(snapshot_dir)
    target = os.readlink(snapshot_dir) if is_link else "NOT A SYMLINK"
    print(f"  Snapshot dir exists: symlink={{is_link}}, target={{target}}")
    if os.path.exists(refs_file):
        with open(refs_file) as f:
            print(f"  refs/main: {{f.read().strip()}}")
    contents = os.listdir(snapshot_dir) if os.path.isdir(snapshot_dir) else []
    print(f"  Contents: {{contents[:15]}}")
else:
    print(f"  NO symlink at {{snapshot_dir}} — dataset will try to download from HF!")

print("[4/8] Loading config...")
t0 = time.time()
config = _config.get_config("pi05_driving")
data_config = config.data.create(config.assets_dirs, config.model)
print(f"  Config loaded in {{time.time()-t0:.1f}}s")
print(f"  repo_id: {{data_config.repo_id}}")
print(f"  action_sequence_keys: {{data_config.action_sequence_keys}}")
print(f"  action_horizon: {{config.model.action_horizon}}")

print("[5/8] Loading dataset metadata (should use cached symlink)...")
t0 = time.time()
dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id)
print(f"  Metadata loaded in {{time.time()-t0:.1f}}s")
print(f"  fps: {{dataset_meta.fps}}")
print(f"  tasks: {{dataset_meta.tasks}}")

print("[6/8] Creating LeRobotDataset (should use cached symlink)...")
delta_timestamps = {{
    key: [t / dataset_meta.fps for t in range(config.model.action_horizon)]
    for key in data_config.action_sequence_keys
}}
print(f"  delta_timestamps: {{delta_timestamps}}")
t0 = time.time()
dataset = lerobot_dataset.LeRobotDataset(
    data_config.repo_id,
    delta_timestamps=delta_timestamps,
)
print(f"  Dataset created in {{time.time()-t0:.1f}}s")
print(f"  Length: {{len(dataset)}}")

print("[7/8] Loading first sample...")
t0 = time.time()
sample = dataset[0]
print(f"  Sample loaded in {{time.time()-t0:.1f}}s")
for k, v in sample.items():
    import numpy as np
    arr = np.asarray(v) if not isinstance(v, str) else v
    if isinstance(arr, np.ndarray):
        print(f"    {{k}}: shape={{arr.shape}}, dtype={{arr.dtype}}")
    else:
        print(f"    {{k}}: {{repr(arr)[:80]}}")

print("[8/8] Testing FULL transform pipeline (RepackTransform + DrivingInputs)...")
t0 = time.time()
config = _config.get_config("pi05_driving")
data_config = config.data.create(config.assets_dirs, config.model)
from openpi.transforms import flatten_dict
flat = flatten_dict(sample)
print(f"  flatten_dict keys: {{list(flat.keys())[:20]}}")

# Apply transforms manually (no multiprocessing) to test the full pipeline
from openpi.transforms import PromptFromLeRobotTask
transformed_ds = _data_loader.TransformedDataset(dataset, [PromptFromLeRobotTask(dataset_meta.tasks)])
item_with_prompt = transformed_ds[0]
print(f"  After PromptFromLeRobotTask: prompt={{item_with_prompt.get('prompt', 'MISSING')}}")

# Apply repack + data transforms
all_transforms = [
    *data_config.repack_transforms.inputs,
    *data_config.data_transforms.inputs,
]
result = dict(item_with_prompt)
for tfm in all_transforms:
    result = tfm(result)
    print(f"  After {{type(tfm).__name__}}: keys={{list(result.keys())[:10]}}")

print(f"  Full pipeline test in {{time.time()-t0:.1f}}s")
for k, v in result.items():
    if hasattr(v, 'shape'):
        print(f"    {{k}}: shape={{v.shape}}, dtype={{v.dtype}}")
    elif isinstance(v, dict):
        for k2, v2 in v.items():
            if hasattr(v2, 'shape'):
                print(f"    {{k}}[{{k2}}]: shape={{v2.shape}}, dtype={{v2.dtype}}")
    else:
        print(f"    {{k}}: {{type(v).__name__}}={{repr(v)[:60]}}")

print("DIAGNOSTIC COMPLETE — full pipeline works")
'''
    import tempfile
    script_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir="/tmp"
    )
    script_file.write(script)
    script_file.close()

    result = subprocess.run(
        [f"{OPENPI_DIR}/.venv/bin/python", "-u", script_file.name],
        cwd=OPENPI_DIR,
        text=True,
        env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
    )
    return result.returncode


# ---------------------------------------------------------------------------
# Compute normalization stats (must run before training)
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="any",
    timeout=60 * 60 * 3,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def compute_norm_stats():
    import os
    import shutil
    import subprocess

    TRAIN_REPO = "markmusic/pi05-physical-av-bc"
    EVAL_REPO = "markmusic/pi05-physical-av-bc-eval"

    _patch_openpi()

    # Check if dataset needs rebuild (missing or too few rows)
    train_path = f"{CACHE_DIR}/hf/lerobot/{TRAIN_REPO}"
    batch_dir = f"{CACHE_DIR}/extracted/xlarge/batches"
    built_marker = f"{train_path}/.built"
    if os.path.exists(batch_dir) and not os.path.exists(built_marker):
        import pyarrow.parquet as pq
        data_dir = os.path.join(train_path, "data")
        total_rows = 0
        if os.path.exists(data_dir):
            for cd in os.listdir(data_dir):
                cd_path = os.path.join(data_dir, cd)
                if not os.path.isdir(cd_path):
                    continue
                for pf in os.listdir(cd_path):
                    if pf.endswith(".parquet"):
                        try:
                            total_rows += pq.read_metadata(os.path.join(cd_path, pf)).num_rows
                        except Exception:
                            pass
        if total_rows < 10000:
            print(f"Dataset missing or incomplete ({total_rows} rows). Building from batch parquets...")
            for marker in [".built", ".repaired_v2", ".consolidated"]:
                for repo in [TRAIN_REPO, EVAL_REPO]:
                    p = os.path.join(f"{CACHE_DIR}/hf/lerobot/{repo}", marker)
                    if os.path.exists(p):
                        os.remove(p)
            _build_lerobot_from_extracted("xlarge", train_repo=TRAIN_REPO, eval_repo=EVAL_REPO)
            cache_volume.commit()

    _repair_metadata(repo_id=TRAIN_REPO)
    _repair_metadata(repo_id=EVAL_REPO)
    _link_dataset_to_hf_cache(repo_id=TRAIN_REPO)

    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", "-m", "scripts.compute_norm_stats",
        "--config-name=pi05_driving",
    ]

    print(f"=== Computing norm stats ===")
    print(f"Command: {' '.join(cmd)}")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONUNBUFFERED": "1"}
    result = subprocess.run(cmd, cwd=OPENPI_DIR, text=True, env=env)

    if result.returncode != 0:
        raise RuntimeError(f"Norm stats computation failed (rc={result.returncode})")

    assets_dir = f"{OPENPI_DIR}/assets/pi05_driving/{TRAIN_REPO}"
    print(f"Norm stats saved to {assets_dir}")
    for root, dirs, files in os.walk(f"{OPENPI_DIR}/assets"):
        for f in files:
            print(f"  {os.path.join(root, f)}")

    cache_assets = f"{CACHE_DIR}/norm_stats_v2/pi05_driving/{TRAIN_REPO}"
    if os.path.exists(cache_assets):
        shutil.rmtree(cache_assets)
    os.makedirs(os.path.dirname(cache_assets), exist_ok=True)
    shutil.copytree(assets_dir, cache_assets)
    cache_volume.commit()

    print("Norm stats computed and cached to norm_stats_v2/")
    return 0


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    gpu="H100:8",
    timeout=60 * 60 * 24,
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("wandb"),
        modal.Secret.from_name("huggingface"),
    ],
    memory=128 * 1024,
)
def train_bc(
    num_steps: int | None = None,
    exp_name: str = "bc-coldstart-v2",
    resume: bool = False,
    batch_size: int = 96,
    skip_hf_validation: bool = False,
    scale: str = "xlarge",
):
    import os
    import pathlib
    import shutil
    import subprocess
    import sys
    import threading
    import time

    TRAIN_REPO = "markmusic/pi05-physical-av-bc"
    EVAL_REPO = "markmusic/pi05-physical-av-bc-eval"

    # --- Pre-flight: validate HF push works ---
    if not skip_hf_validation:
        from huggingface_hub import HfApi

        api = HfApi()
        user = api.whoami()
        print(f"Pre-flight: HF token valid (user={user['name']})")
        api.create_repo(
            HF_CHECKPOINT_REPO, repo_type="model", private=True, exist_ok=True
        )
        print(f"Pre-flight: HF repo {HF_CHECKPOINT_REPO} accessible")

    # Patch openpi with our driving config
    _patch_openpi()

    # Build LeRobot datasets from extracted data (train + eval)
    _build_lerobot_from_extracted(scale, train_repo=TRAIN_REPO, eval_repo=EVAL_REPO)

    # Consolidate dataset parquets if needed (many files → few files)
    _consolidate_local_dataset()

    # Repair metadata if corrupted by previous HF download
    _repair_metadata(repo_id=TRAIN_REPO)
    _repair_metadata(repo_id=EVAL_REPO)

    # Symlink local datasets into HF hub cache so lerobot skips downloads
    _link_dataset_to_hf_cache(repo_id=TRAIN_REPO)
    _link_dataset_to_hf_cache(repo_id=EVAL_REPO)

    # Restore cached norm stats if available
    cache_assets = f"{CACHE_DIR}/norm_stats_v2/pi05_driving/{TRAIN_REPO}"
    assets_dir = f"{OPENPI_DIR}/assets/pi05_driving/{TRAIN_REPO}"
    if os.path.exists(cache_assets) and not os.path.exists(assets_dir):
        os.makedirs(os.path.dirname(assets_dir), exist_ok=True)
        shutil.copytree(cache_assets, assets_dir)
        print(f"Restored norm stats from cache")

    # Compute norm stats if still missing
    if not os.path.exists(assets_dir):
        print("Norm stats missing — computing now...")
        stats_cmd = [
            f"{OPENPI_DIR}/.venv/bin/python", "-u", "-m", "scripts.compute_norm_stats",
            "--config-name=pi05_driving",
        ]
        stats_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        print(f"Running: {' '.join(stats_cmd)}")
        stats_result = subprocess.run(
            stats_cmd, cwd=OPENPI_DIR, text=True, env=stats_env,
        )
        if stats_result.returncode != 0:
            raise RuntimeError(f"Failed to compute norm stats (rc={stats_result.returncode})")
        os.makedirs(os.path.dirname(cache_assets), exist_ok=True)
        if os.path.exists(cache_assets):
            shutil.rmtree(cache_assets)
        shutil.copytree(assets_dir, cache_assets)
        cache_volume.commit()
        print("Norm stats computed and cached")

    # Symlink train norm stats for eval dataset (same normalization)
    eval_assets = f"{OPENPI_DIR}/assets/pi05_driving/{EVAL_REPO}"
    if os.path.exists(assets_dir) and not os.path.exists(eval_assets):
        os.makedirs(os.path.dirname(eval_assets), exist_ok=True)
        os.symlink(assets_dir, eval_assets)
        print(f"Symlinked eval norm stats → train norm stats")

    checkpoint_dir = f"{CACHE_DIR}/checkpoints"
    ckpt_exp_dir = f"{checkpoint_dir}/pi05_driving/{exp_name}"
    if not resume and os.path.exists(ckpt_exp_dir):
        for item in os.listdir(ckpt_exp_dir):
            item_path = os.path.join(ckpt_exp_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
        print(f"Cleaned checkpoint dir: {ckpt_exp_dir}")
    os.makedirs(ckpt_exp_dir, exist_ok=True)
    cache_volume.commit()

    cmd = [
        f"{OPENPI_DIR}/.venv/bin/python", "-u", "-m", "scripts.train",
        "pi05_driving",
        "--exp-name", exp_name,
    ]

    if num_steps is not None:
        cmd.extend(["--num-train-steps", str(num_steps)])

    if batch_size != 96:
        cmd.extend(["--batch-size", str(batch_size)])

    # Always use --resume: if no checkpoint exists, openpi treats it as fresh start.
    # Avoid --overwrite which does rmtree + mkdir that fails on Modal FUSE volumes.
    cmd.append("--resume")

    env = {
        **os.environ,
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
        "WANDB_PROJECT": "pi05-driving",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    print(f"=== Training π0.5 driving BC ===")
    print(f"Command: {' '.join(cmd)}")
    print(f"GPUs: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Checkpoint dir: {checkpoint_dir}")

    # --- Background thread: watches for push_hf trigger and uploads ---
    _stop_watcher = threading.Event()

    def _hf_upload_watcher():
        from huggingface_hub import HfApi

        hf_api = HfApi()
        ckpt_base = f"{checkpoint_dir}/pi05_driving/{exp_name}"
        while not _stop_watcher.is_set():
            _stop_watcher.wait(30)
            if _stop_watcher.is_set():
                break
            try:
                cache_volume.reload()
                trigger = pathlib.Path(f"{TRIGGER_DIR}/push_hf")
                if trigger.exists():
                    trigger.unlink(missing_ok=True)
                    cache_volume.commit()
                    if not os.path.exists(ckpt_base):
                        print("[hf-watcher] No checkpoints yet, skipping upload")
                        continue
                    steps = sorted(
                        [int(d) for d in os.listdir(ckpt_base) if d.isdigit()],
                        reverse=True,
                    )
                    if not steps:
                        print("[hf-watcher] No checkpoint steps found")
                        continue
                    latest = steps[0]
                    params_dir = os.path.join(ckpt_base, str(latest), "params")
                    if not os.path.exists(params_dir):
                        params_dir = os.path.join(ckpt_base, str(latest))
                    print(f"[hf-watcher] Uploading step {latest} to {HF_CHECKPOINT_REPO}")
                    hf_api.create_repo(
                        HF_CHECKPOINT_REPO,
                        repo_type="model",
                        private=True,
                        exist_ok=True,
                    )
                    hf_api.upload_folder(
                        folder_path=params_dir,
                        repo_id=HF_CHECKPOINT_REPO,
                        repo_type="model",
                        commit_message=f"Checkpoint at step {latest}",
                    )
                    print(f"[hf-watcher] Uploaded step {latest} to HF")
            except Exception as e:
                print(f"[hf-watcher] Error: {e}")

    watcher_thread = threading.Thread(target=_hf_upload_watcher, daemon=True)
    watcher_thread.start()
    print("Background HF upload watcher started (polls every 30s)")

    result = subprocess.run(
        cmd,
        cwd=OPENPI_DIR,
        env=env,
        text=True,
    )

    _stop_watcher.set()
    cache_volume.commit()

    if result.returncode != 0:
        print(f"Training failed with return code {result.returncode}")
        return result.returncode

    # Auto-upload final checkpoint to HF
    ckpt_base = f"{checkpoint_dir}/pi05_driving/{exp_name}"
    if os.path.exists(ckpt_base):
        from huggingface_hub import HfApi

        hf_api = HfApi()
        steps = sorted(
            [int(d) for d in os.listdir(ckpt_base) if d.isdigit()], reverse=True
        )
        if steps:
            latest = steps[0]
            params_dir = os.path.join(ckpt_base, str(latest), "params")
            if not os.path.exists(params_dir):
                params_dir = os.path.join(ckpt_base, str(latest))
            print(f"Uploading final checkpoint (step {latest}) to {HF_CHECKPOINT_REPO}")
            hf_api.create_repo(
                HF_CHECKPOINT_REPO,
                repo_type="model",
                private=True,
                exist_ok=True,
            )
            hf_api.upload_folder(
                folder_path=params_dir,
                repo_id=HF_CHECKPOINT_REPO,
                repo_type="model",
                commit_message=f"Final checkpoint at step {latest}",
            )
            print(f"Final checkpoint uploaded to {HF_CHECKPOINT_REPO}")

    print("Training complete!")
    return 0


# ---------------------------------------------------------------------------
# Upload checkpoint to HuggingFace
# ---------------------------------------------------------------------------


@app.function(
    image=train_image,
    volumes=VOLUMES,
    timeout=60 * 60,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=16 * 1024,
)
def upload_checkpoint(
    step: int | None = None,
    exp_name: str = "bc-coldstart",
    repo_id: str = "markmusic/pi05-driving-bc-v2-checkpoint",
):
    import os

    from huggingface_hub import HfApi

    ckpt_base = f"{CACHE_DIR}/checkpoints/pi05_driving/{exp_name}"

    if not os.path.exists(ckpt_base):
        print(f"No checkpoint dir at {ckpt_base}")
        # List what exists
        for root, dirs, files in os.walk(f"{CACHE_DIR}/checkpoints"):
            for d in dirs:
                print(f"  {os.path.join(root, d)}")
        return

    # Find the checkpoint step
    if step is None:
        steps = sorted(
            [int(d) for d in os.listdir(ckpt_base) if d.isdigit()],
            reverse=True,
        )
        if not steps:
            print("No checkpoint steps found")
            return
        step = steps[0]
        print(f"Using latest checkpoint at step {step}")

    params_dir = os.path.join(ckpt_base, str(step), "params")
    if not os.path.exists(params_dir):
        print(f"No params dir at {params_dir}")
        return

    print(f"Uploading checkpoint step {step} to {repo_id}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=params_dir,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Checkpoint at step {step}",
    )
    print(f"Uploaded to https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Convert LeRobot v3.0 dataset to v2.1 for openpi compatibility
# ---------------------------------------------------------------------------


def _convert_dataset_v3_to_v21(repo_id: str):
    """Download dataset from HF and convert v3.0 format to v2.1.

    openpi bundles lerobot 0.1.0 (Python 3.11) which expects v2.0/v2.1 format.
    Our dataset was built with lerobot 0.5.x which produces v3.0.
    Key differences: v2.1 uses tasks.jsonl, v3.0 uses tasks.parquet.
    """
    import json
    import os
    import subprocess

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"

    # Download from HF if not already cached
    if not os.path.exists(dataset_dir):
        print(f"Downloading dataset {repo_id} from HuggingFace...")
        subprocess.run(
            ["huggingface-cli", "download", repo_id,
             "--repo-type", "dataset",
             "--local-dir", dataset_dir],
            check=True,
        )

    meta_dir = os.path.join(dataset_dir, "meta")
    info_path = os.path.join(meta_dir, "info.json")
    tasks_parquet = os.path.join(meta_dir, "tasks.parquet")
    tasks_jsonl = os.path.join(meta_dir, "tasks.jsonl")

    if not os.path.exists(info_path):
        print(f"No info.json found at {info_path}, skipping conversion")
        return

    with open(info_path) as f:
        info = json.load(f)

    if info.get("codebase_version") != "v3.0":
        print(f"Dataset already in {info.get('codebase_version')} format")
        return

    print(f"Converting dataset from v3.0 to v2.1...")

    # Convert tasks.parquet → tasks.jsonl
    if os.path.exists(tasks_parquet) and not os.path.exists(tasks_jsonl):
        import pyarrow.parquet as pq
        table = pq.read_table(tasks_parquet)
        with open(tasks_jsonl, "w") as f:
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                f.write(json.dumps(row) + "\n")
        print(f"  Created tasks.jsonl with {table.num_rows} tasks")

    # Convert episodes parquet → episodes.jsonl
    episodes_parquet = os.path.join(meta_dir, "episodes", "chunk-000", "file-000.parquet")
    episodes_jsonl = os.path.join(meta_dir, "episodes.jsonl")
    if os.path.exists(episodes_parquet) and not os.path.exists(episodes_jsonl):
        import pyarrow.parquet as pq
        table = pq.read_table(episodes_parquet)
        with open(episodes_jsonl, "w") as f:
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                f.write(json.dumps(row) + "\n")
        print(f"  Created episodes.jsonl with {table.num_rows} episodes")

    # Update info.json to v2.1
    info["codebase_version"] = "v2.1"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Updated info.json codebase_version to v2.1")

    cache_volume.commit()
    print("Dataset conversion complete")


# ---------------------------------------------------------------------------
# Build LeRobot datasets from extracted data (no HF push)
# ---------------------------------------------------------------------------


def _build_lerobot_from_extracted(
    scale: str,
    train_repo: str = "markmusic/pi05-physical-av-bc",
    eval_repo: str = "markmusic/pi05-physical-av-bc-eval",
):
    """Build train + eval LeRobot datasets from extracted batch parquets.

    Reads batch parquets with embedded image bytes (written by extract_parallel).
    No individual image file reads — pure sequential parquet I/O.
    """
    import json
    import os
    import shutil
    import time

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = f"{CACHE_DIR}/extracted/{scale}"
    batch_dir = f"{output_dir}/batches"
    train_path = f"{CACHE_DIR}/hf/lerobot/{train_repo}"
    eval_path = f"{CACHE_DIR}/hf/lerobot/{eval_repo}"

    if os.path.exists(f"{train_path}/.built") and os.path.exists(f"{eval_path}/.built"):
        print(f"LeRobot datasets already built (train={train_path}, eval={eval_path})")
        return

    if not os.path.exists(batch_dir):
        raise FileNotFoundError(
            f"No batch parquets at {batch_dir}. Run extraction with embedded images first."
        )

    batch_files = sorted(
        f for f in os.listdir(batch_dir)
        if f.startswith("batch_") and f.endswith(".parquet")
    )
    print(f"Found {len(batch_files)} batch parquets in {batch_dir}")

    print("Loading all batch parquets...")
    t0 = time.time()
    dfs = []
    for bf in batch_files:
        dfs.append(pd.read_parquet(f"{batch_dir}/{bf}"))
        if len(dfs) % 50 == 0:
            print(f"  Loaded {len(dfs)}/{len(batch_files)} batches...")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"Loaded {len(df)} total samples in {time.time() - t0:.0f}s")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

    has_images = "image_bytes" in df.columns
    if not has_images:
        raise ValueError("Batch parquets missing 'image_bytes' column. Re-run extraction.")

    tasks = sorted(df["nav_prompt"].unique().tolist())
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    print(f"Tasks ({len(tasks)}): {tasks}")

    CHUNK_SIZE = 20_000
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    for split_name, repo_id, local_path in [
        ("train", train_repo, train_path),
        ("eval", eval_repo, eval_path),
    ]:
        marker = f"{local_path}/.built"
        if os.path.exists(marker):
            print(f"{split_name} already built")
            continue

        split_df = df[df["split"] == split_name].reset_index(drop=True)
        print(f"\nBuilding {split_name}: {len(split_df)} samples -> {repo_id}")

        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        t_split = time.time()
        n_chunks = (len(split_df) + CHUNK_SIZE - 1) // CHUNK_SIZE

        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, len(split_df))
            chunk_df = split_df.iloc[start:end]

            chunk_dir = f"{local_path}/data/chunk-{chunk_idx:03d}"
            os.makedirs(chunk_dir, exist_ok=True)

            image_array = pa.array(
                [{"bytes": b, "path": None} for b in chunk_df["image_bytes"]],
                type=image_type,
            )

            states = list(zip(
                chunk_df["speed"].astype(float).tolist(),
                chunk_df["heading_rate"].astype(float).tolist(),
            ))
            actions_flat = []
            for a in chunk_df["actions"]:
                flat = []
                for item in a:
                    if isinstance(item, (list, tuple, np.ndarray)):
                        flat.extend(float(v) for v in item)
                    else:
                        flat.append(float(item))
                actions_flat.append(flat)
            task_indices = chunk_df["nav_prompt"].map(task_to_idx).tolist()

            table = pa.table({
                "observation.images.front": image_array,
                "observation.state": pa.array(states, type=pa.list_(pa.float32())),
                "action": pa.array(actions_flat, type=pa.list_(pa.float32())),
                "episode_index": pa.array(range(start, end), type=pa.int64()),
                "frame_index": pa.array([0] * len(chunk_df), type=pa.int64()),
                "index": pa.array(range(start, end), type=pa.int64()),
                "timestamp": pa.array([0.0] * len(chunk_df), type=pa.float64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
            })

            pq.write_table(table, f"{chunk_dir}/episode_000000.parquet")
            del table, image_array, actions_flat

            elapsed = time.time() - t_split
            rate = end / elapsed if elapsed > 0 else 0
            print(f"  Chunk {chunk_idx}/{n_chunks}: {len(chunk_df)} samples ({rate:.0f} samples/s)")
            cache_volume.commit()

        meta_dir = f"{local_path}/meta"
        os.makedirs(meta_dir, exist_ok=True)

        info = {
            "codebase_version": "v2.1",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "robot_type": "cart_fsd",
            "fps": 10,
            "total_episodes": len(split_df),
            "total_frames": len(split_df),
            "total_tasks": len(tasks),
            "total_chunks": n_chunks,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"[0:{len(split_df)}]"},
            "features": {
                "observation.images.front": {
                    "dtype": "image",
                    "shape": [480, 640, 3],
                    "names": ["height", "width", "channel"],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [2],
                    "names": ["speed", "heading_rate"],
                },
                "action": {
                    "dtype": "float32",
                    "shape": [128],
                    "names": None,
                },
            },
        }

        with open(f"{meta_dir}/info.json", "w") as f:
            json.dump(info, f, indent=2)

        with open(f"{meta_dir}/tasks.jsonl", "w") as f:
            for task in tasks:
                f.write(json.dumps({"task_index": task_to_idx[task], "task": task}) + "\n")

        with open(f"{meta_dir}/episodes.jsonl", "w") as f:
            for i in range(len(split_df)):
                t_idx = task_to_idx[split_df.iloc[i]["nav_prompt"]]
                f.write(json.dumps({"episode_index": i, "length": 1, "task_index": t_idx}) + "\n")

        with open(marker, "w") as f:
            f.write(f"{len(split_df)} samples")
        with open(f"{local_path}/.consolidated", "w") as f:
            f.write("done")

        cache_volume.commit()
        elapsed = time.time() - t_split
        print(f"Built {split_name}: {len(split_df)} samples in {elapsed:.0f}s ({len(split_df)/elapsed:.1f} samples/s)")

    print("\nBUILD COMPLETE")


# ---------------------------------------------------------------------------
# Repair metadata if corrupted by HF download
# ---------------------------------------------------------------------------


def _repair_metadata(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Verify and repair metadata + episode_index if corrupted by HF download.

    Ensures episode_index in parquets is sequential (0..N-1) and metadata matches.
    """
    import json
    import os
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    meta_dir = os.path.join(dataset_dir, "meta")
    data_dir = os.path.join(dataset_dir, "data")

    if not os.path.exists(data_dir) or not os.path.exists(meta_dir):
        return

    info_path = os.path.join(meta_dir, "info.json")
    episodes_path = os.path.join(meta_dir, "episodes.jsonl")
    tasks_path = os.path.join(meta_dir, "tasks.jsonl")
    repair_marker = os.path.join(dataset_dir, ".repaired_v2")

    if not os.path.exists(info_path):
        return

    if os.path.exists(repair_marker):
        with open(info_path) as f:
            info = json.load(f)
        print(f"Metadata already repaired for {repo_id}: {info.get('total_episodes')} episodes")
        return

    # Collect all parquet files with an action column
    chunk_dirs = sorted(
        d for d in os.listdir(data_dir)
        if d.startswith("chunk-") and os.path.isdir(os.path.join(data_dir, d))
    )
    pq_files = []
    total_rows = 0
    max_ep_idx = -1
    for cd in chunk_dirs:
        for pf in sorted(os.listdir(os.path.join(data_dir, cd))):
            if not pf.endswith(".parquet"):
                continue
            fpath = os.path.join(data_dir, cd, pf)
            try:
                schema = pq.read_schema(fpath)
            except Exception:
                continue
            if "action" not in schema.names:
                continue
            t = pq.read_table(fpath, columns=["episode_index"])
            n = t.num_rows
            if n > 0:
                ep_vals = t.column("episode_index").to_pylist()
                max_ep_idx = max(max_ep_idx, max(ep_vals))
            total_rows += n
            pq_files.append((cd, pf, fpath, n))

    # Remove empty chunk directories
    for cd in chunk_dirs:
        cd_path = os.path.join(data_dir, cd)
        if os.path.isdir(cd_path) and not any(
            f.endswith(".parquet") for f in os.listdir(cd_path)
        ):
            import shutil as _shutil
            _shutil.rmtree(cd_path)
            print(f"  Removed empty chunk dir: {cd}")

    with open(info_path) as f:
        info = json.load(f)

    metadata_ok = (
        info.get("total_episodes") == total_rows
        and max_ep_idx == total_rows - 1
    )

    if metadata_ok:
        with open(repair_marker, "w") as f:
            f.write("ok")
        cache_volume.commit()
        print(f"Metadata OK for {repo_id}: {total_rows} episodes, max_idx={max_ep_idx}")
        return

    print(f"REPAIRING {repo_id}: info says {info.get('total_episodes')} episodes, "
          f"data has {total_rows} rows, max episode_index={max_ep_idx}")
    t0 = time.time()

    # Rewrite parquets with sequential episode_index (0..N-1)
    global_idx = 0
    task_indices = []
    CHUNK_SIZE = 20_000
    new_chunks = {}

    for cd, pf, fpath, n in pq_files:
        if n == 0:
            continue
        table = pq.read_table(fpath)
        new_ep_idx = pa.array(range(global_idx, global_idx + n), type=pa.int64())
        new_index = pa.array(range(global_idx, global_idx + n), type=pa.int64())
        table = table.set_column(
            table.column_names.index("episode_index"), "episode_index", new_ep_idx
        )
        table = table.set_column(
            table.column_names.index("index"), "index", new_index
        )
        if "task_index" in table.column_names:
            task_indices.extend(table.column("task_index").to_pylist())

        target_chunk = f"chunk-{global_idx // CHUNK_SIZE:03d}"
        if target_chunk not in new_chunks:
            new_chunks[target_chunk] = []
        new_chunks[target_chunk].append(table)
        global_idx += n

    # Write consolidated chunks
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        for chunk_name, tables in sorted(new_chunks.items()):
            merged = pa.concat_tables(tables, promote_options="permissive")
            chunk_dir = os.path.join(tmpdir, chunk_name)
            os.makedirs(chunk_dir, exist_ok=True)
            pq.write_table(merged, os.path.join(chunk_dir, "episode_000000.parquet"))

        # Replace data dir contents
        for cd in chunk_dirs:
            shutil.rmtree(os.path.join(data_dir, cd), ignore_errors=True)
        for chunk_name in new_chunks:
            src = os.path.join(tmpdir, chunk_name)
            dst = os.path.join(data_dir, chunk_name)
            shutil.copytree(src, dst)

    n_chunks = len(new_chunks)

    # Write corrected metadata
    with open(episodes_path, "w") as f:
        for i in range(total_rows):
            t_idx = task_indices[i] if i < len(task_indices) else 0
            f.write(json.dumps({
                "episode_index": i, "length": 1, "task_index": t_idx
            }) + "\n")

    info["total_episodes"] = total_rows
    info["total_frames"] = total_rows
    info["total_chunks"] = n_chunks
    info["chunks_size"] = CHUNK_SIZE
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    with open(repair_marker, "w") as f:
        f.write("ok")

    norm_cache = f"{CACHE_DIR}/norm_stats_v2/pi05_driving/{repo_id}"
    if os.path.exists(norm_cache):
        shutil.rmtree(norm_cache)
        print(f"Cleared stale norm stats cache")

    cache_volume.commit()
    elapsed = time.time() - t0
    print(f"Repaired {repo_id}: {total_rows} episodes, {n_chunks} chunks, "
          f"episode_index 0-{total_rows-1} ({elapsed:.0f}s)")


# ---------------------------------------------------------------------------
# Patch openpi with driving config
# ---------------------------------------------------------------------------


def _link_dataset_to_hf_cache(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Link our local dataset into the HF hub cache so lerobot skips downloads."""
    import os
    import shutil

    local_dataset = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    if not os.path.exists(local_dataset):
        print(f"No local dataset at {local_dataset}, will download from HF")
        return

    hub_dir = f"{CACHE_DIR}/hf/hub/datasets--{repo_id.replace('/', '--')}"
    snapshot_dir = f"{hub_dir}/snapshots/local"

    if os.path.islink(snapshot_dir):
        os.unlink(snapshot_dir)
    elif os.path.exists(snapshot_dir):
        shutil.rmtree(snapshot_dir)

    os.makedirs(f"{hub_dir}/refs", exist_ok=True)
    os.makedirs(os.path.dirname(snapshot_dir), exist_ok=True)

    with open(f"{hub_dir}/refs/main", "w") as f:
        f.write("local")

    try:
        os.symlink(local_dataset, snapshot_dir)
        print(f"Symlinked local dataset → HF hub cache")
    except OSError:
        os.makedirs(snapshot_dir, exist_ok=True)
        for item in os.listdir(local_dataset):
            src = os.path.join(local_dataset, item)
            dst = os.path.join(snapshot_dir, item)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        print(f"Copied local dataset → HF hub cache (symlink not supported)")

    cache_volume.commit()
    print(f"Linked local dataset → HF hub cache (skips HF downloads)")


def _consolidate_local_dataset(repo_id: str = "markmusic/pi05-physical-av-bc"):
    """Merge per-episode parquet files into chunk-level files for fast loading."""
    import glob
    import os
    import shutil
    import tempfile

    dataset_dir = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
    data_dir = os.path.join(dataset_dir, "data")
    marker = os.path.join(dataset_dir, ".consolidated")

    if not os.path.exists(data_dir):
        print(f"No local dataset at {data_dir}, skipping consolidation")
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    chunk_dirs = sorted(glob.glob(os.path.join(data_dir, "chunk-*")))

    if os.path.exists(marker):
        # Already consolidated — just clean up any leftover HF files
        needs_cleanup = False
        for chunk_dir in chunk_dirs:
            extras = glob.glob(os.path.join(chunk_dir, "file-*.parquet"))
            episode_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
            if extras or len(episode_files) > 1:
                needs_cleanup = True
                break
        if not needs_cleanup:
            print("Dataset already consolidated")
            return
        print("Cleaning up leftover files...")
        # Remove extra episode files and HF-generated files, keep only episode_000000.parquet
        cleaned = 0
        for chunk_dir in chunk_dirs:
            main_file = os.path.join(chunk_dir, "episode_000000.parquet")
            if not os.path.exists(main_file):
                continue
            for extra in glob.glob(os.path.join(chunk_dir, "episode_*.parquet")):
                if extra != main_file:
                    os.remove(extra)
                    cleaned += 1
            for extra in glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
                os.remove(extra)
                cleaned += 1
        if cleaned:
            print(f"  Removed {cleaned} leftover files")
            cache_volume.commit()
        else:
            print("  No leftover files found")
        return

    print(f"Consolidating {len(chunk_dirs)} chunks (one-time operation)...")

    for chunk_dir in chunk_dirs:
        episode_files = sorted(glob.glob(os.path.join(chunk_dir, "episode_*.parquet")))
        if len(episode_files) <= 1:
            print(f"  {os.path.basename(chunk_dir)}: already consolidated")
            continue

        print(f"  {os.path.basename(chunk_dir)}: merging {len(episode_files)} files...", end=" ", flush=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            tables = []
            for f in episode_files:
                try:
                    tables.append(pq.read_table(f))
                except Exception as e:
                    print(f"\n    WARNING: skipping corrupt file {os.path.basename(f)}: {e}")

            merged = pa.concat_tables(tables, promote_options="permissive")
            tmp_out = os.path.join(tmpdir, "episode_000000.parquet")
            pq.write_table(merged, tmp_out)

            for f in episode_files:
                os.remove(f)
            shutil.copy2(tmp_out, os.path.join(chunk_dir, "episode_000000.parquet"))

        print(f"{len(merged)} rows", flush=True)
        cache_volume.commit()

    # Remove HuggingFace auto-generated file-*.parquet files that conflict
    for chunk_dir in chunk_dirs:
        for extra in glob.glob(os.path.join(chunk_dir, "file-*.parquet")):
            print(f"  Removing HF-generated {os.path.basename(chunk_dir)}/{os.path.basename(extra)}")
            os.remove(extra)

    with open(marker, "w") as f:
        f.write("done")
    cache_volume.commit()
    print("Consolidation complete")


def _patch_openpi():
    """Copy our driving config patches into the openpi repo."""
    import shutil

    # 1. Copy driving_policy.py
    driving_policy_src = "/opt/driving_policy.py"
    driving_policy_dst = f"{OPENPI_DIR}/src/openpi/policies/driving_policy.py"

    # Write driving_policy.py inline since we can't mount from the host
    with open(driving_policy_dst, "w") as f:
        f.write('''"""Data transforms for π0.5 driving policy (Cart FSD)."""

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
            actions = actions[np.newaxis, :]  # (128,) -> (1, 128)
        return {"actions": actions}
''')

    # 2. Patch gemma.py to add gemma_2b_lora_driving variant
    gemma_path = f"{OPENPI_DIR}/src/openpi/models/gemma.py"
    with open(gemma_path, "r") as f:
        content = f.read()

    if "gemma_2b_lora_driving" not in content:
        # Add variant to Literal type
        content = content.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]',
        )
        # Add config block before gemma_300m_lora
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
        with open(gemma_path, "w") as f:
            f.write(content)

    # 3. Patch config.py to add driving config
    config_path = f"{OPENPI_DIR}/src/openpi/training/config.py"
    with open(config_path, "r") as f:
        content = f.read()

    if "pi05_driving" not in content:
        # Add import
        content = content.replace(
            "import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\nimport openpi.policies.droid_policy as droid_policy",
        )

        # Add LeRobotDrivingDataConfig class before TrainConfig
        driving_data_config = '''
@dataclasses.dataclass(frozen=True)
class LeRobotDrivingDataConfig(DataConfigFactory):
    """Data config for Cart FSD driving with pi0.5."""

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

        # Add training config entry before closing bracket
        driving_train_config = '''
    #
    # Cart FSD driving config.
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
            "gs://openpi-assets/checkpoints/pi05_base/params"
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
        # Insert before the closing bracket of _CONFIGS
        import re
        content = content.replace(
            "    *polaris_config.get_polaris_configs(),\n]",
            "    *polaris_config.get_polaris_configs()," + driving_train_config + "]",
        )

        with open(config_path, "w") as f:
            f.write(content)

    # 4. Patch lerobot to accept our dataset format
    import glob
    import os
    for lerobot_utils in glob.glob(
        f"{OPENPI_DIR}/.venv/lib/python*/site-packages/lerobot/common/datasets/utils.py"
    ):
        with open(lerobot_utils, "r") as f:
            content = f.read()
        if "PATCHED" not in content:
            content = content.replace(
                "raise ForwardCompatibilityError(repo_id, min(upper_versions))",
                "pass  # PATCHED: accept our dataset version",
            )
            # Patch hf_transform_to_torch to decode image struct dicts to PIL
            # Our parquets store images as pa.struct(bytes, path). Without HF Image()
            # features metadata, these load as plain dicts. flatten_dict() in
            # RepackTransform recurses into dicts, breaking key lookup.
            # Fix: decode image bytes→PIL so the existing PIL→tensor path handles them.
            content = content.replace(
                "items_dict[key] = [x if isinstance(x, str) else torch.tensor(x) for x in items_dict[key]]",
                "items_dict[key] = [_decode_image_dict(x) if isinstance(x, dict) and 'bytes' in x else (x if isinstance(x, str) else torch.tensor(x)) for x in items_dict[key]]  # IMAGE_DICT_PATCHED",
            )
            # Add the helper function near the top of the file
            decode_fn = (
                "\nimport io as _io  # IMAGE_DICT_HELPER\n"
                "def _decode_image_dict(d):\n"
                "    return PILImage.open(_io.BytesIO(d['bytes']))\n\n"
            )
            if "IMAGE_DICT_HELPER" not in content:
                content = content.replace(
                    "import torch",
                    "import torch" + decode_fn,
                )
            with open(lerobot_utils, "w") as f:
                f.write(content)
            print(f"  Patched version check + image dict handling in {lerobot_utils}")

    # 4b. Patch lerobot to skip downloading data when it already exists locally
    #     AND skip the file-path assertion (images are embedded in parquet, no separate files).
    for lerobot_ds in glob.glob(
        f"{OPENPI_DIR}/.venv/lib/python*/site-packages/lerobot/common/datasets/lerobot_dataset.py"
    ):
        with open(lerobot_ds, "r") as f:
            ds_content = f.read()
        if "DOWNLOAD_PATCHED" not in ds_content:
            # Patch 1: skip download when local data exists
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
            # Patch 2: skip file-path assertion (images embedded in parquet, no separate files)
            ds_content = ds_content.replace(
                "assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())",
                "pass  # DOWNLOAD_PATCHED: images embedded in parquet, no separate files to check",
            )
            # Patch 3: skip timestamp sync check (single-frame episodes, no inter-frame timing)
            ds_content = ds_content.replace(
                "check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)",
                "pass  # TIMESTAMP_PATCHED: single-frame driving episodes, no sync to validate",
            )
            with open(lerobot_ds, "w") as f:
                f.write(ds_content)
            print(f"  Patched lerobot_dataset.py to skip download + timestamp check")

        # Patch 4 (separate guard): fix EmptyDatasetError — data_dir doesn't recurse
        # into chunk subdirs, so use explicit data_files glob instead
        if "GLOB_PATCHED" not in ds_content:
            ds_content = ds_content.replace(
                'hf_dataset = load_dataset("parquet", data_dir=path, split="train")',
                'import glob as _glob  # GLOB_PATCHED\n'
                '            _pq_files = sorted(_glob.glob(f"{path}/**/*.parquet", recursive=True))\n'
                '            if _pq_files:\n'
                '                hf_dataset = load_dataset("parquet", data_files=_pq_files, split="train")\n'
                '            else:\n'
                '                hf_dataset = load_dataset("parquet", data_dir=path, split="train")',
            )
            with open(lerobot_ds, "w") as f:
                f.write(ds_content)
            print(f"  Patched lerobot_dataset.py with glob fix for chunk subdirectories")

    # 4c. Patch load_metadata to handle missing episodes_stats gracefully.
    #     Our v2 dataset doesn't have episodes_stats.jsonl. Without this patch,
    #     load_metadata() raises FileNotFoundError → pull_from_repo downloads v1 metadata
    #     from HF → overwrites our v2 episodes.jsonl (176K → 10K episodes).
    for lerobot_ds in glob.glob(
        f"{OPENPI_DIR}/.venv/lib/python*/site-packages/lerobot/common/datasets/lerobot_dataset.py"
    ):
        with open(lerobot_ds, "r") as f:
            ds_content = f.read()
        if "META_PATCHED" not in ds_content:
            ds_content = ds_content.replace(
                'self.episodes_stats = load_episodes_stats(self.root)',
                'try:  # META_PATCHED\n'
                '                self.episodes_stats = load_episodes_stats(self.root)\n'
                '            except (FileNotFoundError, Exception):\n'
                '                self.episodes_stats = {}',
            )
            ds_content = ds_content.replace(
                'self.stats = aggregate_stats(list(self.episodes_stats.values()))',
                'self.stats = aggregate_stats(list(self.episodes_stats.values())) if self.episodes_stats else {}',
            )
            with open(lerobot_ds, "w") as f:
                f.write(ds_content)
            print(f"  Patched lerobot_dataset.py to handle missing episodes_stats")

    # 5. Patch scripts/train.py to handle shape mismatches when loading
    #    base checkpoint with different action_dim (32→2).
    train_script = f"{OPENPI_DIR}/scripts/train.py"
    with open(train_script, "r") as f:
        content = f.read()
    if "SHAPE_PATCHED" not in content:
        old_validate = "at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)"
        new_validate = """# SHAPE_PATCHED: filter shape-mismatched params before validation
    import logging as _log
    def _filter_shapes(expected, loaded):
        import jax
        e_flat, e_struct = jax.tree.flatten(expected)
        l_flat, _ = jax.tree.flatten(loaded)
        fixed = []
        for e, l in zip(e_flat, l_flat):
            if hasattr(e, 'shape') and hasattr(l, 'shape') and e.shape != l.shape:
                _log.warning(f"Shape mismatch: expected {e.shape}, got {l.shape} — using init weights")
                fixed.append(e)
            else:
                fixed.append(l)
        return jax.tree.unflatten(e_struct, fixed)
    loaded_params = _filter_shapes(params_shape, loaded_params)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)"""
        if old_validate in content:
            content = content.replace(old_validate, new_validate)
            with open(train_script, "w") as f:
                f.write(content)
            print("  Patched scripts/train.py for shape mismatch handling")
        else:
            print("  WARNING: Could not find validation call in train.py")
            for i, line in enumerate(content.split('\n')):
                if 'check_pytree_equality' in line:
                    print(f"    Line {i+1}: {line.strip()}")

    # 5b. Patch init_wandb to create checkpoint dir if wiped by --overwrite
    with open(train_script, "r") as f:
        content = f.read()
    if "WANDB_DIR_PATCHED" not in content:
        content = content.replace(
            "    ckpt_dir = config.checkpoint_dir\n    if not ckpt_dir.exists():",
            "    ckpt_dir = config.checkpoint_dir\n    ckpt_dir.mkdir(parents=True, exist_ok=True)  # WANDB_DIR_PATCHED\n    if not ckpt_dir.exists():",
        )
        with open(train_script, "w") as f:
            f.write(content)
        print("  Patched scripts/train.py for wandb checkpoint dir creation")

    # 6. Add LR logging + eval loss to scripts/train.py
    with open(train_script, "r") as f:
        content = f.read()
    if "LR_EVAL_PATCHED" not in content:
        # Patch A: Add eval_step_fn before main()
        content = content.replace(
            "def main(config: _config.TrainConfig):",
            '''def eval_step_fn(
    config: _config.TrainConfig,
    rng,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    observation, actions = batch
    return jnp.mean(model.compute_loss(rng, observation, actions, train=False))


def main(config: _config.TrainConfig):  # LR_EVAL_PATCHED''',
        )

        # Patch B: Add LR schedule, peval_step, and proper eval from held-out dataset
        content = content.replace(
            "    start_step = int(train_state.step)",
            '''    lr_schedule_fn = config.lr_schedule.create()

    peval_step = jax.jit(
        functools.partial(eval_step_fn, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    # Build eval data loader from held-out eval dataset (single-process, 5 batches only)
    _eval_data_factory = dataclasses.replace(
        config.data,
        repo_id="markmusic/pi05-physical-av-bc-eval",
        assets=_config.AssetsConfig(asset_id="markmusic/pi05-physical-av-bc"),
    )
    _eval_loader = _data_loader.create_data_loader(
        dataclasses.replace(config, data=_eval_data_factory, num_workers=0),
        sharding=data_sharding,
        shuffle=False,
        skip_norm_stats=False,
        num_batches=5,
    )
    _eval_batches = list(_eval_loader)
    del _eval_loader
    logging.info(f"Cached {len(_eval_batches)} held-out eval batches")

    start_step = int(train_state.step)''',
        )

        # Patch C: Add LR to logged metrics
        content = content.replace(
            "            wandb.log(reduced_info, step=step)",
            '''            reduced_info["learning_rate"] = float(lr_schedule_fn(step))
            wandb.log(reduced_info, step=step)''',
        )

        # Patch D: Add eval loss at checkpoint intervals
        content = content.replace(
            '''        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)''',
            '''        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            if _eval_batches:
                _el = []
                for _eb in _eval_batches:
                    with sharding.set_mesh(mesh):
                        _el.append(peval_step(train_rng, train_state, _eb))
                _eval_loss = float(np.mean([float(jax.device_get(x)) for x in _el]))
                pbar.write(f"Step {step}: eval_loss={_eval_loss:.4f}")
                wandb.log({"eval_loss": _eval_loss}, step=step)
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)''',
        )

        with open(train_script, "w") as f:
            f.write(content)
        print("  Patched scripts/train.py for LR + eval loss logging")

    # 7. Add trigger-file checkpoint save (on-demand save via /cache/triggers/save_now)
    with open(train_script, "r") as f:
        content = f.read()
    if "TRIGGER_PATCHED" not in content:
        content = content.replace(
            "    infos = []",
            '''    import pathlib as _pathlib  # TRIGGER_PATCHED
    _trigger_dir = _pathlib.Path("/cache/triggers")

    infos = []''',
            1,  # only replace FIRST occurrence (the 4-space one before the for-loop)
        )

        content = content.replace(
            "        batch = next(data_iter)",
            '''        # Check for on-demand save trigger every 10 steps
        if step % 10 == 0 and _trigger_dir.exists():
            _save_trigger = _trigger_dir / "save_now"
            if _save_trigger.exists():
                try:
                    _save_trigger.unlink(missing_ok=True)
                except OSError:
                    pass
                pbar.write(f"[trigger] Manual checkpoint save at step {step}")
                if _eval_batches:
                    _el = []
                    for _eb in _eval_batches:
                        with sharding.set_mesh(mesh):
                            _el.append(peval_step(train_rng, train_state, _eb))
                    _eval_loss = float(np.mean([float(jax.device_get(x)) for x in _el]))
                    pbar.write(f"[trigger] eval_loss={_eval_loss:.4f}")
                    wandb.log({"eval_loss": _eval_loss}, step=step)
                _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
                pbar.write(f"[trigger] Checkpoint saved at step {step}")

        batch = next(data_iter)''',
        )

        with open(train_script, "w") as f:
            f.write(content)
        print("  Patched scripts/train.py for trigger-file checkpoint saves")

    # Compile-check the patched train.py to catch syntax/indentation errors early
    with open(train_script, "r") as f:
        source = f.read()
    try:
        compile(source, train_script, "exec")
        print("  Compile-check passed for scripts/train.py")
    except SyntaxError as e:
        lines = source.split("\n")
        ctx_start = max(0, e.lineno - 5)
        ctx_end = min(len(lines), e.lineno + 5)
        ctx = "\n".join(f"  {i+1:4d}: {lines[i]}" for i in range(ctx_start, ctx_end))
        raise SyntaxError(f"Patched train.py has syntax error at line {e.lineno}:\n{ctx}") from e

    print("openpi patched with driving config")
