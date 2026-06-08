"""CPU-only data prep for π0.5 driving BC training.

No GPU, no openpi, no JAX. Just reads batch parquets from the Modal volume.

Usage:
    # Rebuild LeRobot datasets from batch parquets (run first if corrupted)
    modal run --detach pi05/modal_compute_stats.py::rebuild_dataset

    # Compute norm stats (run after rebuild)
    modal run --detach pi05/modal_compute_stats.py::compute_norm_stats

    # Do both in sequence
    modal run --detach pi05/modal_compute_stats.py::prep_all
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-compute-stats"
CACHE_DIR = "/cache"

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "numpy", "pyarrow", "pandas", "tqdm"
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
app = modal.App(APP_NAME)

TRAIN_REPO = "markmusic/pi05-physical-av-bc"
EVAL_REPO = "markmusic/pi05-physical-av-bc-eval"
CHUNK_SIZE = 20_000


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60,
    memory=32 * 1024,
)
def compute_norm_stats():
    import json
    import os
    import shutil
    import time

    import numpy as np
    import pandas as pd

    batch_dir = f"{CACHE_DIR}/extracted/xlarge/batches"
    if not os.path.exists(batch_dir):
        raise FileNotFoundError(f"No batch parquets at {batch_dir}")

    batch_files = sorted(
        f for f in os.listdir(batch_dir)
        if f.startswith("batch_") and f.endswith(".parquet")
    )
    print(f"Found {len(batch_files)} batch parquets")

    # Running stats (matches openpi's normalize.RunningStats)
    NUM_BINS = 5000

    class RunningStats:
        def __init__(self):
            self.count = 0
            self.mean = None
            self.mean_sq = None
            self.mn = None
            self.mx = None
            self.histograms = None
            self.bin_edges = None

        def update(self, batch: np.ndarray):
            batch = batch.reshape(-1, batch.shape[-1])
            n, d = batch.shape
            if self.count == 0:
                self.mean = np.mean(batch, axis=0)
                self.mean_sq = np.mean(batch**2, axis=0)
                self.mn = np.min(batch, axis=0)
                self.mx = np.max(batch, axis=0)
                self.histograms = [np.zeros(NUM_BINS) for _ in range(d)]
                self.bin_edges = [
                    np.linspace(self.mn[i] - 1e-10, self.mx[i] + 1e-10, NUM_BINS + 1)
                    for i in range(d)
                ]
            else:
                new_mx = np.max(batch, axis=0)
                new_mn = np.min(batch, axis=0)
                changed = np.any(new_mx > self.mx) or np.any(new_mn < self.mn)
                self.mx = np.maximum(self.mx, new_mx)
                self.mn = np.minimum(self.mn, new_mn)
                if changed:
                    self._adjust_histograms()
            self.count += n
            bm = np.mean(batch, axis=0)
            bms = np.mean(batch**2, axis=0)
            self.mean += (bm - self.mean) * (n / self.count)
            self.mean_sq += (bms - self.mean_sq) * (n / self.count)
            self._update_histograms(batch)

        def _adjust_histograms(self):
            for i in range(len(self.histograms)):
                old_edges = self.bin_edges[i]
                new_edges = np.linspace(self.mn[i], self.mx[i], NUM_BINS + 1)
                new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self.histograms[i])
                self.histograms[i] = new_hist
                self.bin_edges[i] = new_edges

        def _update_histograms(self, batch):
            for i in range(batch.shape[1]):
                hist, _ = np.histogram(batch[:, i], bins=self.bin_edges[i])
                self.histograms[i] += hist

        def get_stats(self):
            var = self.mean_sq - self.mean**2
            std = np.sqrt(np.maximum(0, var))
            q01_vals, q99_vals = [], []
            for hist, edges in zip(self.histograms, self.bin_edges):
                cumsum = np.cumsum(hist)
                idx01 = np.searchsorted(cumsum, 0.01 * self.count)
                idx99 = np.searchsorted(cumsum, 0.99 * self.count)
                q01_vals.append(float(edges[idx01]))
                q99_vals.append(float(edges[idx99]))
            return {
                "mean": [float(x) for x in self.mean],
                "std": [float(x) for x in std],
                "q01": q01_vals,
                "q99": q99_vals,
            }

    state_stats = RunningStats()
    action_stats = RunningStats()

    t0 = time.time()
    total_samples = 0
    for i, bf in enumerate(batch_files):
        df = pd.read_parquet(
            f"{batch_dir}/{bf}",
            columns=["speed", "heading_rate", "actions"],
        )

        states = np.stack([
            df["speed"].astype(np.float32).values,
            df["heading_rate"].astype(np.float32).values,
        ], axis=1)
        state_stats.update(states)

        actions_list = []
        for a in df["actions"]:
            flat = []
            for item in a:
                if isinstance(item, (list, tuple, np.ndarray)):
                    flat.extend(float(v) for v in item)
                else:
                    flat.append(float(item))
            actions_list.append(flat)
        actions = np.array(actions_list, dtype=np.float32)
        action_stats.update(actions)

        total_samples += len(df)
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(batch_files)} batches, {total_samples} samples, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nProcessed {total_samples} samples from {len(batch_files)} batches in {elapsed:.0f}s")

    s = state_stats.get_stats()
    a = action_stats.get_stats()

    print(f"\nState stats (dim=2):")
    print(f"  mean: {s['mean']}")
    print(f"  std:  {s['std']}")
    print(f"  q01:  {s['q01']}")
    print(f"  q99:  {s['q99']}")
    print(f"\nAction stats (dim={len(a['mean'])}):")
    print(f"  mean range: [{min(a['mean']):.4f}, {max(a['mean']):.4f}]")
    print(f"  std range:  [{min(a['std']):.4f}, {max(a['std']):.4f}]")
    print(f"  q01 range:  [{min(a['q01']):.4f}, {max(a['q01']):.4f}]")
    print(f"  q99 range:  [{min(a['q99']):.4f}, {max(a['q99']):.4f}]")

    # Save in openpi's exact format (matches normalize.py _NormStatsDict)
    norm_stats_json = {
        "norm_stats": {
            "state": s,
            "actions": a,
        }
    }

    # Save to volume cache
    cache_path = f"{CACHE_DIR}/norm_stats_v2/pi05_driving/{TRAIN_REPO}"
    os.makedirs(cache_path, exist_ok=True)
    with open(f"{cache_path}/norm_stats.json", "w") as f:
        json.dump(norm_stats_json, f, indent=2)

    cache_volume.commit()
    print(f"\nSaved to {cache_path}/norm_stats.json")
    print("DONE — norm stats cached. Run train_bc next.")
    return 0


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60 * 2,
    memory=64 * 1024,
)
def rebuild_dataset():
    """Rebuild train + eval LeRobot datasets from batch parquets. CPU-only."""
    import json
    import os
    import shutil
    import time

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    batch_dir = f"{CACHE_DIR}/extracted/xlarge/batches"
    train_path = f"{CACHE_DIR}/hf/lerobot/{TRAIN_REPO}"
    eval_path = f"{CACHE_DIR}/hf/lerobot/{EVAL_REPO}"

    if not os.path.exists(batch_dir):
        raise FileNotFoundError(f"No batch parquets at {batch_dir}")

    batch_files = sorted(
        f for f in os.listdir(batch_dir)
        if f.startswith("batch_") and f.endswith(".parquet")
    )
    print(f"Found {len(batch_files)} batch parquets in {batch_dir}")

    # Load all batches
    print("Loading all batch parquets...")
    t0 = time.time()
    dfs = []
    for i, bf in enumerate(batch_files):
        dfs.append(pd.read_parquet(f"{batch_dir}/{bf}"))
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(batch_files)} batches...")
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"Loaded {len(df)} total samples in {time.time() - t0:.0f}s")
    print(f"Columns: {list(df.columns)}")
    print(f"Split distribution: {df['split'].value_counts().to_dict()}")

    if "image_bytes" not in df.columns:
        raise ValueError("Batch parquets missing 'image_bytes' column")

    tasks = sorted(df["nav_prompt"].unique().tolist())
    task_to_idx = {t: i for i, t in enumerate(tasks)}
    print(f"Tasks ({len(tasks)}): {tasks}")

    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

    for split_name, repo_id, local_path in [
        ("train", TRAIN_REPO, train_path),
        ("eval", EVAL_REPO, eval_path),
    ]:
        marker = f"{local_path}/.built"
        if os.path.exists(marker):
            # Check if actually has data
            data_dir = f"{local_path}/data"
            total = 0
            if os.path.exists(data_dir):
                for cd in os.listdir(data_dir):
                    cdp = os.path.join(data_dir, cd)
                    if os.path.isdir(cdp):
                        for pf in os.listdir(cdp):
                            if pf.endswith(".parquet"):
                                try:
                                    total += pq.read_metadata(os.path.join(cdp, pf)).num_rows
                                except Exception:
                                    pass
            expected = len(df[df["split"] == split_name])
            if total == expected:
                print(f"{split_name} already built ({total} rows)")
                continue
            print(f"{split_name} marker exists but only {total} rows — rebuilding")

        split_df = df[df["split"] == split_name].reset_index(drop=True)
        n = len(split_df)
        print(f"\nBuilding {split_name}: {n} samples -> {repo_id}")

        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        t_split = time.time()
        n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE

        for ci in range(n_chunks):
            start = ci * CHUNK_SIZE
            end = min(start + CHUNK_SIZE, n)
            chunk_df = split_df.iloc[start:end]

            chunk_dir = f"{local_path}/data/chunk-{ci:03d}"
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

            nc = len(chunk_df)
            table = pa.table({
                "observation.images.front": image_array,
                "observation.state": pa.array(states, type=pa.list_(pa.float32())),
                "action": pa.array(actions_flat, type=pa.list_(pa.float32())),
                "episode_index": pa.array(range(start, end), type=pa.int64()),
                "frame_index": pa.array([0] * nc, type=pa.int64()),
                "index": pa.array(range(start, end), type=pa.int64()),
                "timestamp": pa.array([0.0] * nc, type=pa.float64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
            })

            pq.write_table(table, f"{chunk_dir}/episode_000000.parquet")
            del table, image_array, actions_flat

            elapsed = time.time() - t_split
            print(f"  Chunk {ci+1}/{n_chunks}: {nc} samples ({end}/{n}, {elapsed:.0f}s)")
            cache_volume.commit()

        # Write metadata
        meta_dir = f"{local_path}/meta"
        os.makedirs(meta_dir, exist_ok=True)

        info = {
            "codebase_version": "v2.1",
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "robot_type": "cart_fsd",
            "fps": 10,
            "total_episodes": n,
            "total_frames": n,
            "total_tasks": len(tasks),
            "total_chunks": n_chunks,
            "chunks_size": CHUNK_SIZE,
            "splits": {"train": f"[0:{n}]"},
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
            for i in range(n):
                t_idx = task_to_idx[split_df.iloc[i]["nav_prompt"]]
                f.write(json.dumps({"episode_index": i, "length": 1, "task_index": t_idx}) + "\n")

        with open(marker, "w") as f:
            f.write(f"{n} samples")
        with open(f"{local_path}/.consolidated", "w") as f:
            f.write("done")
        with open(f"{local_path}/.repaired_v2", "w") as f:
            f.write("ok")

        cache_volume.commit()
        elapsed = time.time() - t_split
        print(f"Built {split_name}: {n} samples in {elapsed:.0f}s")

    # Set up HF hub cache symlinks so LeRobot finds local data
    # ALWAYS re-create to avoid stale copies from previous runs
    for repo_id in [TRAIN_REPO, EVAL_REPO]:
        local_dataset = f"{CACHE_DIR}/hf/lerobot/{repo_id}"
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
            print(f"Symlinked {repo_id} → HF hub cache")
        except OSError:
            shutil.copytree(local_dataset, snapshot_dir)
            print(f"Copied {repo_id} → HF hub cache (symlink unsupported)")

        cache_volume.commit()

    print("\nDATASET REBUILD COMPLETE")
    return 0


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    timeout=60 * 60 * 3,
    memory=64 * 1024,
)
def prep_all():
    """Rebuild dataset + compute norm stats in one shot. CPU-only."""
    rebuild_dataset.remote()
    compute_norm_stats.remote()
    print("\nALL PREP DONE. Ready for: modal run --detach pi05/modal_train_bc.py::train_bc")
