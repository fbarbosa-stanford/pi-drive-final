"""Direct Preference Optimization (DPO) with Alpamayo/AR1 trajectory preferences."""

from dpo.alpamayo_preference import (
    AlpamayoPreferenceConfig,
    PreferencePair,
    pick_preference_pair,
    score_candidate_vs_ar1,
)
from dpo.objective import DPOInputs, DPOOutputs, compute_dpo_loss

__all__ = [
    "AlpamayoPreferenceConfig",
    "PreferencePair",
    "pick_preference_pair",
    "score_candidate_vs_ar1",
    "DPOInputs",
    "DPOOutputs",
    "compute_dpo_loss",
]
