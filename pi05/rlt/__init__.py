"""RLT (RL-Token) learner core for Cart FSD π0.5 Stage 2.

Framework-agnostic, fully-testable actor-critic machinery that locally edits
π0.5's native (accel, curvature) reference chunk under a BC anchor, with clean
ABC seams where the deferred π0.5 / hardware pieces drop in later. See
``README.md`` for the math and the seam contract.
"""

from __future__ import annotations

from .buffer import ReplayBuffer
from .config import RLTConfig
from .interfaces import (
    EgomotionStream,
    RewardInterface,
    SafetyLayer,
    VLAWrapper,
)
from .learner import AsyncLearner
from .losses import critic_loss, critic_td_target, policy_loss
from .nets import GaussianActor, QEnsemble, build_mlp, polyak_update
from .safety import NativeUnitsSafetyLayer
from .transition import (
    Batch,
    DoneType,
    SourceType,
    field_specs,
    make_storage,
    validate_batch,
)

__all__ = [
    "RLTConfig",
    "ReplayBuffer",
    "AsyncLearner",
    "GaussianActor",
    "QEnsemble",
    "build_mlp",
    "polyak_update",
    "critic_loss",
    "critic_td_target",
    "policy_loss",
    "NativeUnitsSafetyLayer",
    "VLAWrapper",
    "EgomotionStream",
    "RewardInterface",
    "SafetyLayer",
    "DoneType",
    "SourceType",
    "Batch",
    "field_specs",
    "make_storage",
    "validate_batch",
]
