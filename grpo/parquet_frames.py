"""Read the LeRobot parquet directly, bypassing lerobot's frame loader.

The ``pi05-physical-av-bc`` dataset has non-contiguous ``episode_index`` and a broken
``hf_xet`` download path, which make ``LeRobotDataset.__getitem__`` unusable (episode-index
overflow, failed image fetch). For GRPO/DPO we only need image + state + 128-action +
prompt per row — all in the parquet — so we read rows directly and apply openpi's own
data transforms, getting the exact same model-space item as ``transform_dataset`` would.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np

_IMAGE_KEY = "observation.images.front"
_STATE_KEY = "observation.state"
_ACTION_KEY = "action"


def decode_image(img) -> np.ndarray:
    """HF parquet stores images as struct{bytes,path}, raw PNG bytes, PIL, or array."""
    if isinstance(img, np.ndarray):
        return img
    if hasattr(img, "convert"):  # PIL.Image
        return np.asarray(img.convert("RGB"))
    if isinstance(img, dict):
        raw = img.get("bytes")
        if raw is not None:
            from io import BytesIO

            from PIL import Image

            return np.asarray(Image.open(BytesIO(raw)).convert("RGB"))
        path = img.get("path")
        if path:
            from PIL import Image

            return np.asarray(Image.open(path).convert("RGB"))
    if isinstance(img, (bytes, bytearray)):
        from io import BytesIO

        from PIL import Image

        return np.asarray(Image.open(BytesIO(img)).convert("RGB"))
    return np.asarray(img)


def _load_tasks(dataset_root: str | Path) -> dict[int, str]:
    """lerobot v3 task-index -> task string map (meta/tasks.jsonl or meta/tasks.parquet)."""
    root = Path(dataset_root)
    tasks: dict[int, str] = {}
    jl = root / "meta" / "tasks.jsonl"
    if jl.exists():
        for line in jl.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                tasks[int(d.get("task_index", len(tasks)))] = d.get("task", "")
        return tasks
    pq_tasks = root / "meta" / "tasks.parquet"
    if pq_tasks.exists():
        import pyarrow.parquet as pq

        for i, d in enumerate(pq.read_table(pq_tasks).to_pylist()):
            tasks[int(d.get("task_index", i))] = d.get("task", "")
    return tasks


class FrameSource:
    """Random-access view over the dataset's parquet rows + openpi transform application."""

    def __init__(self, dataset_root: str | Path, data_config=None):
        import pyarrow.parquet as pq

        candidates = sorted(glob.glob(f"{dataset_root}/data/**/*.parquet", recursive=True))
        if not candidates:
            raise RuntimeError(f"No parquet files under {dataset_root}/data")
        # Skip corrupt/partial parquet (e.g. left by an interrupted download) instead of
        # crashing the whole pipeline.
        self.files, self.counts, self.skipped = [], [], []
        for f in candidates:
            try:
                self.counts.append(pq.ParquetFile(f).metadata.num_rows)
                self.files.append(f)
            except Exception as e:  # noqa: BLE001
                self.skipped.append(os.path.basename(f))
                print(f"[FrameSource] skipping unreadable parquet {os.path.basename(f)}: {e}", flush=True)
        if not self.files:
            raise RuntimeError(f"No readable parquet files under {dataset_root}/data")
        if self.skipped:
            print(f"[FrameSource] skipped {len(self.skipped)} corrupt file(s): {self.skipped}", flush=True)
        self.starts = np.cumsum([0] + self.counts)
        self.total = int(self.starts[-1])

        cols = set(pq.ParquetFile(self.files[0]).schema_arrow.names)
        self.image_col = _IMAGE_KEY if _IMAGE_KEY in cols else next((c for c in cols if "image" in c.lower()), None)
        self.state_col = _STATE_KEY if _STATE_KEY in cols else next((c for c in cols if "state" in c.lower()), None)
        self.action_col = (
            _ACTION_KEY if _ACTION_KEY in cols
            else next((c for c in cols if "action" in c.lower() and "index" not in c.lower()), None)
        )
        self.task_col = next((c for c in ("task", "prompt", "nav_prompt", "task_index") if c in cols), None)
        self._read_cols = [c for c in (self.image_col, self.state_col, self.action_col, self.task_col) if c]
        self.tasks = _load_tasks(dataset_root)

        self.data_config = data_config
        self._cache_fi: int | None = None
        self._cache_tbl = None
        self._transforms = None

    def __len__(self) -> int:
        return self.total

    def describe(self) -> dict:
        return {
            "n_files": len(self.files),
            "total_rows": self.total,
            "image_col": self.image_col,
            "state_col": self.state_col,
            "action_col": self.action_col,
            "task_col": self.task_col,
            "n_tasks": len(self.tasks),
        }

    def _table(self, fi: int):
        import pyarrow.parquet as pq

        if fi != self._cache_fi:
            self._cache_tbl = pq.read_table(self.files[fi], columns=self._read_cols)
            self._cache_fi = fi
        return self._cache_tbl

    def _raw_row(self, idx: int) -> dict:
        idx = int(idx) % self.total
        fi = int(np.searchsorted(self.starts, idx, side="right") - 1)
        local = idx - int(self.starts[fi])
        return self._table(fi).slice(local, 1).to_pylist()[0]

    def _prompt(self, raw: dict) -> str:
        if self.task_col is None:
            return "drive"
        v = raw.get(self.task_col)
        if isinstance(v, str):
            return v or "drive"
        if isinstance(v, (int, np.integer)):
            return self.tasks.get(int(v), "drive")
        return "drive"

    def row(self, idx: int) -> dict:
        """Decoded raw row: ``observation.images.front`` (HWC array), ``observation.state``
        (np), ``action`` (128 np), ``prompt`` (str). Keys match the repack transform sources."""
        r = self._raw_row(idx)
        return {
            "observation.images.front": decode_image(r.get(self.image_col)),
            "observation.state": np.asarray(r.get(self.state_col), dtype=np.float32).reshape(-1),
            "action": np.asarray(r.get(self.action_col), dtype=np.float32).reshape(-1),
            "prompt": self._prompt(r),
        }

    def compute_norm_stats(self, max_frames: int = 5000) -> dict:
        """Mean/std/q01/q99 of action (128) and state from the parquet, in openpi format.

        Keys are ``actions`` and ``state`` (matching the post-repack data keys). Fast:
        reads only the action+state columns, no image decode.
        """
        import pyarrow.parquet as pq

        acts, sts, n = [], [], 0
        for f in self.files:
            cols = [c for c in (self.action_col, self.state_col) if c]
            t = pq.read_table(f, columns=cols)
            acts.append(np.asarray(t[self.action_col].to_pylist(), dtype=np.float32))
            if self.state_col:
                sts.append(np.asarray(t[self.state_col].to_pylist(), dtype=np.float32))
            n += acts[-1].shape[0]
            if n >= max_frames:
                break
        A = np.concatenate(acts)[:max_frames]
        S = np.concatenate(sts)[:max_frames] if sts else np.zeros((len(A), 2), np.float32)

        def _block(X):
            return {
                "mean": X.mean(0).tolist(),
                "std": (X.std(0) + 1e-6).tolist(),
                "q01": np.quantile(X, 0.01, axis=0).tolist(),
                "q99": np.quantile(X, 0.99, axis=0).tolist(),
            }

        return {"norm_stats": {"actions": _block(A), "state": _block(S)}}

    def _build_transforms(self):
        from openpi import transforms as _transforms

        dc = self.data_config
        self._transforms = [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            _transforms.Normalize(dc.norm_stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]

    def model_item(self, idx: int) -> dict:
        """Apply the openpi transform pipeline to a row -> model-space dict
        (normalized state, tokenized prompt, resized image, normalized (4,32) action)."""
        if self._transforms is None:
            self._build_transforms()
        d = dict(self.row(idx))
        for t in self._transforms:
            d = t(d)
        return d
