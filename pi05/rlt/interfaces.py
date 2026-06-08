"""ABC seams for the pieces that are *deferred* until the friend's 10 Hz π0.5
Thor inference code (and the cart sensor/actuation glue) lands.

The learner core depends only on these abstractions, never on π0.5 or hardware,
so it is fully testable today against the kinematic unicycle env. The
``README`` seam contract spells out exactly what a concrete ``VLAWrapper`` must
provide (the ``prefix_out`` embedding hook → RL token + the native reference
chunk). Each ABC is intentionally tiny: raw tensors / plain values in and out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .transition import DoneType


class VLAWrapper(ABC):
    """Frozen π0.5 + RL-token extractor. ``encode`` maps a raw observation to
    the opaque RL token ``z_rl`` and π0.5's native (accel, curvature) reference
    chunk ``a_ref``. Deferred: wraps the friend's π0.5 forward + the embedding
    hook at openpi ``pi0.py:209`` (``prefix_out``)."""

    @property
    @abstractmethod
    def token_dim(self) -> int: ...

    @property
    @abstractmethod
    def chunk_len(self) -> int: ...

    @abstractmethod
    def encode(self, obs) -> tuple[Tensor, Tensor]:
        """obs -> (z_rl[token_dim], a_ref[chunk_len, action_dim])."""


class EgomotionStream(ABC):
    """High-quality streamed egomotion readout -> proprioception ``s_p``."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def read(self) -> Tensor:
        """Latest proprio vector, shape (dim,)."""


class RewardInterface(ABC):
    """Sparse/terminal/binary outcome. For RLT this is a human success/fail
    label at episode end (mid-episode returns 0/NONE)."""

    @abstractmethod
    def evaluate(self, obs) -> tuple[float, DoneType]:
        """obs -> (reward, done_type)."""


class SafetyLayer(ABC):
    """Hard backstop applied to every action before it actuates."""

    @abstractmethod
    def clamp(self, a: Tensor, state) -> tuple[Tensor, bool]:
        """a -> (safe_a, was_modified)."""
