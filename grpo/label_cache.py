"""Cached AR1 / GT labels for offline GRPO (Stage 1 → Stage 2 handoff)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class LabelRecord:
    """One training sample at (clip_id, t0) with labels for ranking."""

    clip_id: str
    t0_us: int
    coc_text: str = ""
    # Expert (AR1) trajectory — either actions or decoded xyz
    expert_actions: np.ndarray | None = None
    expert_xyz: np.ndarray | None = None
    gt_xyz: np.ndarray | None = None
    ego_history_xyz: np.ndarray | None = None
    ego_history_rot: np.ndarray | None = None
    initial_speed: float = 0.0
    initial_yaw: float = 0.0
    task: str = "drive forward safely"

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("expert_actions", "expert_xyz", "gt_xyz", "ego_history_xyz", "ego_history_rot"):
            val = d[key]
            d[key] = val.tolist() if val is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> LabelRecord:
        def _arr(key: str):
            v = d.get(key)
            return None if v is None else np.asarray(v, dtype=np.float32)

        return cls(
            clip_id=str(d["clip_id"]),
            t0_us=int(d["t0_us"]),
            coc_text=str(d.get("coc_text", "")),
            expert_actions=_arr("expert_actions"),
            expert_xyz=_arr("expert_xyz"),
            gt_xyz=_arr("gt_xyz"),
            ego_history_xyz=_arr("ego_history_xyz"),
            ego_history_rot=_arr("ego_history_rot"),
            initial_speed=float(d.get("initial_speed", 0.0)),
            initial_yaw=float(d.get("initial_yaw", 0.0)),
            task=str(d.get("task", "drive forward safely")),
        )


class LabelCache:
    """JSONL-backed store of LabelRecord entries."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[LabelRecord]:
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        records: list[LabelRecord] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(LabelRecord.from_dict(json.loads(line)))
        return records

    def save(self, records: list[LabelRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec.to_dict()) + "\n")

    @staticmethod
    def from_parquet_stub(parquet_path: str | Path) -> list[LabelRecord]:
        """Placeholder for Stage 1 pseudo-label parquet → LabelRecord conversion."""
        raise NotImplementedError(
            "Wire this to modal_pseudolabel.py output once AR1 labels land. "
            f"Expected parquet at: {parquet_path}"
        )
