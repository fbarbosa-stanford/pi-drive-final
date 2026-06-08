"""pi0.5 driving inference — standalone script.

Loads the trained BC checkpoint and runs inference on a single observation.
Returns predicted (acceleration, curvature) actions over a 6.4s horizon at 10Hz.

Requirements:
    - openpi repo cloned and installed (uv sync)
    - Checkpoint downloaded from HF: markmusic/pi05-driving-bc-v2-checkpoint
    - norm_stats.json in the same directory as this script

Usage (local):
    cd /path/to/openpi
    .venv/bin/python /path/to/infer.py --checkpoint /path/to/checkpoint --image /path/to/image.jpg

Usage (Modal):
    modal run pi05/modal_eval_bev.py::visualize_bev --n-samples 10
"""

import argparse
import dataclasses
import io
import json
import os
import sys
import time

import einops
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# openpi patching — adds driving config to the openpi registry
# ---------------------------------------------------------------------------

def patch_openpi(openpi_dir: str):
    """Write driving policy + config patches into the openpi source tree."""

    # 1. driving_policy.py
    dst = os.path.join(openpi_dir, "src/openpi/policies/driving_policy.py")
    with open(dst, "w") as f:
        f.write(_DRIVING_POLICY_SRC)

    # 2. gemma.py — add LoRA variant
    gemma_path = os.path.join(openpi_dir, "src/openpi/models/gemma.py")
    with open(gemma_path, "r") as f:
        content = f.read()
    if "gemma_2b_lora_driving" not in content:
        content = content.replace(
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora"]',
            'Variant = Literal["dummy", "gemma_300m", "gemma_300m_lora", "gemma_2b", "gemma_2b_lora", "gemma_2b_lora_driving"]',
        )
        content = content.replace(
            '    if variant == "gemma_300m_lora":',
            '    if variant == "gemma_2b_lora_driving":\n'
            '        return Config(\n'
            '            width=2048, depth=18, mlp_dim=16_384,\n'
            '            num_heads=8, num_kv_heads=1, head_dim=256,\n'
            '            lora_configs={"attn": lora.LoRAConfig(rank=32, alpha=64.0), "ffn": lora.LoRAConfig(rank=32, alpha=64.0)},\n'
            '        )\n'
            '    if variant == "gemma_300m_lora":',
        )
        with open(gemma_path, "w") as f:
            f.write(content)

    # 3. config.py — add pi05_driving training config
    config_path = os.path.join(openpi_dir, "src/openpi/training/config.py")
    with open(config_path, "r") as f:
        content = f.read()
    if "pi05_driving" not in content:
        content = content.replace(
            "import openpi.policies.droid_policy as droid_policy",
            "import openpi.policies.driving_policy as driving_policy\n"
            "import openpi.policies.droid_policy as droid_policy",
        )
        content = content.replace(
            "@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
            _DRIVING_DATA_CONFIG_SRC + "\n@dataclasses.dataclass(frozen=True)\nclass TrainConfig:",
        )
        content = content.replace(
            "    *polaris_config.get_polaris_configs(),\n]",
            "    *polaris_config.get_polaris_configs()," + _DRIVING_TRAIN_CONFIG_SRC + "]",
        )
        with open(config_path, "w") as f:
            f.write(content)

    print("openpi patched for driving inference")


# ---------------------------------------------------------------------------
# Unicycle integrator — converts (accel, curvature) to XY trajectory
# ---------------------------------------------------------------------------

def actions_to_trajectory(actions: np.ndarray, speed: float, dt: float = 0.1) -> np.ndarray:
    """Integrate (acceleration, curvature) actions via unicycle kinematic model.

    Args:
        actions: (64, 2) array of [acceleration m/s^2, curvature rad/m]
        speed: initial ego speed in m/s
        dt: timestep in seconds (0.1 for 10Hz)

    Returns:
        (64, 2) array of [x_forward, y_lateral] positions in ego frame (meters)
    """
    v = speed
    heading = 0.0
    x, y = 0.0, 0.0
    pts = []
    for accel, kappa in actions:
        v = max(v + float(accel) * dt, 0.0)
        heading += v * float(kappa) * dt
        x += v * np.cos(heading) * dt
        y += v * np.sin(heading) * dt
        pts.append([x, y])
    return np.array(pts)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------

def load_policy(openpi_dir: str, checkpoint_dir: str, norm_stats_path: str = None):
    """Load the trained pi0.5 driving policy.

    Args:
        openpi_dir: path to openpi repo root
        checkpoint_dir: path to checkpoint directory (contains params/ subdirectory)
        norm_stats_path: path to norm_stats.json (if not in checkpoint's assets dir)

    Returns:
        policy object with .infer() method
    """
    patch_openpi(openpi_dir)

    # Copy norm_stats into the expected location if provided
    if norm_stats_path:
        asset_dir = os.path.join(checkpoint_dir, "assets/markmusic/pi05-physical-av-bc")
        os.makedirs(asset_dir, exist_ok=True)
        target = os.path.join(asset_dir, "norm_stats.json")
        if not os.path.exists(target):
            import shutil
            shutil.copy2(norm_stats_path, target)

    from openpi.training import config as _config
    from openpi.policies.policy_config import create_trained_policy
    import openpi.transforms as transforms

    config = _config.get_config("pi05_driving")
    repack = transforms.Group(
        inputs=[transforms.RepackTransform({
            "observation/image": "observation.images.front",
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })]
    )

    policy = create_trained_policy(
        config, checkpoint_dir,
        repack_transforms=repack,
        default_prompt="drive",
    )
    return policy


def infer(policy, image: Image.Image, speed: float, heading_rate: float = 0.0) -> dict:
    """Run a single inference step.

    Args:
        policy: loaded policy from load_policy()
        image: PIL Image from front camera
        speed: current ego speed in m/s
        heading_rate: current heading rate in rad/s (default 0)

    Returns:
        dict with:
            actions: (64, 2) array of [acceleration, curvature]
            trajectory: (64, 2) array of [x, y] positions in ego frame (meters)
            infer_ms: inference time in milliseconds
    """
    state = np.array([speed, heading_rate], dtype=np.float32)
    dummy_actions = np.zeros(128, dtype=np.float32)

    obs = {
        "observation.images.front": image,
        "observation.state": state,
        "action": dummy_actions,
        "prompt": "drive",
    }

    result = policy.infer(obs)
    pred_flat = result["actions"]
    if pred_flat.ndim == 1:
        actions = pred_flat.reshape(64, 2)
    else:
        actions = pred_flat.reshape(-1, 2)[:64]

    trajectory = actions_to_trajectory(actions, speed)

    return {
        "actions": actions,
        "trajectory": trajectory,
        "infer_ms": result["policy_timing"]["infer_ms"],
    }


# ---------------------------------------------------------------------------
# Inline source strings for patching
# ---------------------------------------------------------------------------

_DRIVING_POLICY_SRC = '''
import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

def _parse_image(image) -> np.ndarray:
    if isinstance(image, dict) and "bytes" in image:
        import io
        from PIL import Image as _PILImage
        image = np.array(_PILImage.open(io.BytesIO(image["bytes"])))
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
            actions = actions[np.newaxis, :]
        return {"actions": actions}
'''

_DRIVING_DATA_CONFIG_SRC = """
@dataclasses.dataclass(frozen=True)
class LeRobotDrivingDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform({
                "observation/image": "observation.images.front",
                "observation/state": "observation.state",
                "actions": "action",
                "prompt": "prompt",
            })])
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
"""

_DRIVING_TRAIN_CONFIG_SRC = (
    "\n    TrainConfig(\n"
    '        name="pi05_driving",\n'
    "        model=pi0_config.Pi0Config(\n"
    "            pi05=True, action_dim=128, action_horizon=1,\n"
    '            paligemma_variant="gemma_2b_lora_driving",\n'
    '            action_expert_variant="gemma_300m",\n'
    "        ),\n"
    "        data=LeRobotDrivingDataConfig(\n"
    '            repo_id="markmusic/pi05-physical-av-bc",\n'
    "            base_config=DataConfig(prompt_from_task=True),\n"
    "        ),\n"
    "        weight_loader=weight_loaders.CheckpointWeightLoader(\n"
    '            "gs://openpi-assets/checkpoints/pi05_base/params"\n'
    "        ),\n"
    "        freeze_filter=pi0_config.Pi0Config(\n"
    "            pi05=True, action_dim=128, action_horizon=1,\n"
    '            paligemma_variant="gemma_2b_lora_driving",\n'
    '            action_expert_variant="gemma_300m",\n'
    "        ).get_freeze_filter(),\n"
    "        lr_schedule=_optimizer.CosineDecaySchedule(\n"
    "            warmup_steps=750, peak_lr=3e-5, decay_steps=15_000, decay_lr=3e-6,\n"
    "        ),\n"
    "        optimizer=_optimizer.AdamW(b1=0.9, b2=0.999, clip_gradient_norm=1.0),\n"
    "        num_train_steps=15_000, batch_size=96, fsdp_devices=1,\n"
    "        save_interval=500, log_interval=50,\n"
    '        checkpoint_base_dir="/cache/checkpoints",\n'
    "    ),\n"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pi0.5 driving inference")
    parser.add_argument("--openpi-dir", default="/opt/openpi", help="Path to openpi repo")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (parent of params/)")
    parser.add_argument("--norm-stats", default=None, help="Path to norm_stats.json")
    parser.add_argument("--image", required=True, help="Path to front camera image")
    parser.add_argument("--speed", type=float, default=5.0, help="Current speed in m/s")
    args = parser.parse_args()

    # Auto-detect norm_stats if not provided
    if args.norm_stats is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidate = os.path.join(script_dir, "norm_stats.json")
        if os.path.exists(candidate):
            args.norm_stats = candidate

    print("Loading policy...")
    policy = load_policy(args.openpi_dir, args.checkpoint, args.norm_stats)
    print("Policy loaded.")

    image = Image.open(args.image).convert("RGB")
    print(f"Running inference (speed={args.speed} m/s)...")
    result = infer(policy, image, args.speed)

    print(f"Inference time: {result['infer_ms']:.0f}ms")
    print(f"Actions shape: {result['actions'].shape}")
    print(f"Trajectory shape: {result['trajectory'].shape}")
    print(f"First 5 actions (accel, curvature):")
    for i in range(5):
        a, k = result['actions'][i]
        print(f"  t={i*0.1:.1f}s: accel={a:.3f} m/s^2, curvature={k:.5f} rad/m")
    print(f"Trajectory endpoint: x={result['trajectory'][-1, 0]:.1f}m forward, y={result['trajectory'][-1, 1]:.1f}m lateral")
