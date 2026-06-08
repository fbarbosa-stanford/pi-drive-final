"""
Generate AR1 pseudo-labels for GRPO RewardContext (Alpamayo-R1 on PhysicalAI-AV).

  modal run grpo/modal_ar1_labels.py::smoke_one_clip
  modal run --detach grpo/modal_ar1_labels.py::generate_labels --max-clips 100

Output: /cache/labels/ar1_labels.jsonl (LabelCache format)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

APP_NAME = "pi05-ar1-labels"
CACHE_DIR = "/cache"
ALPAMAYO_DIR = "/opt/alpamayo"
_PKG_DIR = Path(__file__).resolve().parents[1]

ar1_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs", "build-essential", "ninja-build", "ffmpeg")
    .pip_install(
        "torch==2.8.0",
        "torchvision>=0.23.0",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("wheel", "setuptools", "packaging", "numpy")
    .pip_install("flash-attn>=2.8.3", extra_options="--no-build-isolation")
    .run_commands(
        f"git clone --depth 1 https://github.com/NVlabs/alpamayo.git {ALPAMAYO_DIR}",
        f"sed -i '/cosmos-rl/d; /vllm/d' {ALPAMAYO_DIR}/pyproject.toml",
        f"pip install -e {ALPAMAYO_DIR}",
    )
    .pip_install("huggingface_hub")
    .env({"HF_HOME": f"{CACHE_DIR}/hf"})
    # add_local_* must be the LAST build steps (newer Modal forbids build steps after them).
    .add_local_dir(_PKG_DIR, remote_path="/app/pi_05_drives")
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}
app = modal.App(APP_NAME)

LABELS_PATH = f"{CACHE_DIR}/labels/ar1_labels.jsonl"
DEFAULT_CLIP = "030c760c-ae38-49aa-9ad8-f5650a545d26"


@app.function(
    image=ar1_image,
    gpu="H100",
    timeout=60 * 30,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def smoke_one_clip(clip_id: str = DEFAULT_CLIP, t0_us: int = 5_100_000) -> dict:
    """One AR1 forward pass + ADE vs GT + write one LabelRecord."""
    sys.path.insert(0, "/app/pi_05_drives")
    import numpy as np
    import torch

    from alpamayo_r1 import helper
    from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
    from grpo.label_cache import LabelRecord
    from rewards.flat_actions import flatten_actions

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    messages = helper.create_message(data["image_frames"].flatten(0, 1))
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device(
        {
            "tokenized_data": inputs,
            "ego_history_xyz": data["ego_history_xyz"],
            "ego_history_rot": data["ego_history_rot"],
        },
        "cuda",
    )
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=0.98,
            temperature=0.6,
            num_traj_samples=1,
            max_generation_length=256,
            return_extra=True,
        )

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    min_ade = float(np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1).min())

    coc_text = str(extra["cot"][0][0]) if extra.get("cot") else ""
    expert_xyz = pred_xyz.cpu().numpy()[0, 0, 0]  # (T, 3)

    # Placeholder actions — wire traj_to_action from alpamayo unicycle space in production
    try:
        from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace

        space = UnicycleAccelCurvatureActionSpace()
        expert_actions = space.traj_to_action(
            torch.from_numpy(pred_xyz[0, 0, 0]),
            torch.from_numpy(pred_rot[0, 0, 0]),
        )
        expert_actions = np.asarray(expert_actions.cpu(), dtype=np.float32)
        if expert_actions.ndim == 1:
            expert_actions = expert_actions.reshape(-1, 2)
    except Exception as e:
        print(f"traj_to_action fallback: {e}")
        expert_actions = np.zeros((64, 2), dtype=np.float32)

    record = LabelRecord(
        clip_id=clip_id,
        t0_us=t0_us,
        coc_text=coc_text,
        expert_actions=expert_actions,
        expert_xyz=expert_xyz.astype(np.float32),
        gt_xyz=gt_xy.T.astype(np.float32),
        initial_speed=float(np.linalg.norm(data["ego_history_xyz"].cpu()[0, -1])),
    )

    Path(LABELS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict()) + "\n")
    cache_volume.commit()

    return {
        "clip_id": clip_id,
        "min_ade_ar1_vs_gt_m": min_ade,
        "coc_preview": coc_text[:200],
        "expert_actions_shape": list(expert_actions.shape),
        "labels_path": LABELS_PATH,
    }


@app.function(
    image=ar1_image,
    gpu="H100",
    timeout=60 * 60 * 12,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def generate_labels(max_clips: int = 50) -> str:
    """Batch AR1 labeling — extend with clip list from PhysicalAI-AV."""
    # TODO: iterate clip manifest; for now single smoke clip
    return smoke_one_clip.remote()


@app.local_entrypoint()
def main(max_clips: int = 1):
    if max_clips <= 1:
        print(smoke_one_clip.remote())
    else:
        print(generate_labels.remote(max_clips=max_clips))
