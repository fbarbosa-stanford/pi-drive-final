"""Data transforms for π0.5 driving policy (from markmusic27/pi-drive)."""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


# Mark's BC checkpoint is action_dim=128, action_horizon=1: the whole driving chunk is
# one 128-d "action" at horizon 1 (matches the dataset's 128-flat action per frame).
ACTION_HORIZON = 1
ACTION_DIM = 128


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
            # (...,128) flat chunk -> (action_horizon=4, action_dim=32) for the model.
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32).reshape(
                ACTION_HORIZON, ACTION_DIM
            )

        if "prompt" in data:
            import random

            if random.random() < 0.3:
                inputs["prompt"] = "drive"
            else:
                inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DrivingOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Model emits (action_horizon=4, action_dim=32); flatten back to (1, 128).
        actions = np.asarray(data["actions"], dtype=np.float32).reshape(-1)[
            : ACTION_HORIZON * ACTION_DIM
        ]
        return {"actions": actions[np.newaxis, :]}
