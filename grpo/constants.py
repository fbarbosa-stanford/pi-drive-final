"""HuggingFace / openpi identifiers for Mark's BC driving checkpoint."""

HF_BC_CHECKPOINT_REPO = "markmusic/pi05-driving-bc-v2-checkpoint"
HF_BC_DATASET_REPO = "markmusic/pi05-physical-av-bc"
HF_BC_EVAL_DATASET_REPO = "markmusic/pi05-physical-av-bc-eval"

OPENPI_CONFIG_NAME = "pi05_driving"
# Mark BC dataset: one frame per episode, all timestamps at 0.0
LEROBOT_TOLERANCE_S = 1.0

from rewards.flat_actions import ACTION_DIM_PER_STEP, ACTION_HORIZON, FLAT_ACTION_DIM
