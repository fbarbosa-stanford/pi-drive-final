"""Convert between openpi flat (128,) and (64, 2) accel/curvature chunks."""

from __future__ import annotations

import numpy as np

ACTION_HORIZON = 64
ACTION_DIM_PER_STEP = 2
FLAT_ACTION_DIM = ACTION_HORIZON * ACTION_DIM_PER_STEP


def unflatten_actions(flat: np.ndarray) -> np.ndarray:
    """(128,) or (1, 128) -> (64, 2)."""
    x = np.asarray(flat, dtype=np.float32).reshape(-1)
    if x.size != FLAT_ACTION_DIM:
        raise ValueError(f"expected {FLAT_ACTION_DIM} dims, got {x.size}")
    return x.reshape(ACTION_HORIZON, ACTION_DIM_PER_STEP)


def flatten_actions(chunk: np.ndarray) -> np.ndarray:
    """(64, 2) -> (128,)."""
    x = np.asarray(chunk, dtype=np.float32)
    if x.shape != (ACTION_HORIZON, ACTION_DIM_PER_STEP):
        raise ValueError(f"expected ({ACTION_HORIZON}, {ACTION_DIM_PER_STEP}), got {x.shape}")
    return x.reshape(FLAT_ACTION_DIM)
