"""Push the final GRPO checkpoint from the Modal volume straight to Hugging Face.

  modal run upload_grpo_hf.py                       # auto repo: <you>/pi05-driving-grpo-5k-checkpoint
  modal run upload_grpo_hf.py --repo-id me/my-repo  # explicit repo
  modal run upload_grpo_hf.py --include-optimizer   # also upload train_state/ (resumable)
  modal run upload_grpo_hf.py --private             # private repo

Uploads params/ + assets/ (+ metadata) so it loads exactly like the BC checkpoint via
openpi's create_trained_policy. train_state/ (optimizer) is omitted unless requested.
"""

from __future__ import annotations

import modal

APP_NAME = "pi05-hf-upload"
CACHE_DIR = "/cache"
OPENPI_CONFIG_NAME = "pi05_driving"

image = modal.Image.debian_slim().pip_install("huggingface_hub>=0.25", "hf-transfer>=0.1.6").env(
    {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
)
cache_volume = modal.Volume.from_name("pi05-cache")
app = modal.App(APP_NAME)

_README = """---
license: other
base_model: markmusic/pi05-driving-bc-v2-checkpoint
tags:
- robotics
- autonomous-driving
- openpi
- pi05
- flow-grpo
---

# pi0.5 Driving — Flow-GRPO post-trained ({exp_name}/{step})

Flow-GRPO post-train of [`markmusic/pi05-driving-bc-v2-checkpoint`](https://huggingface.co/markmusic/pi05-driving-bc-v2-checkpoint)
on NVIDIA PhysicalAI-AV (Alpamayo) driving clips.

## Held-out validation (10 clips, vs the BC-v2 base)

| Model | ADE (m) ↓ | val_loss (flow) ↓ |
|-------|-----------|-------------------|
| BC-v2 (base) | 3.56 | 0.237 |
| **GRPO-5k (this model)** | **2.81 (−21%)** | **0.190 (−20%)** |

GRPO improves both displacement and the held-out flow-matching loss, and (unlike the
GT-imitation DPO variant) it does not over-steer on turns.

## Method

Composite reward = PDMS driving quality (progress / comfort / drivable-area / time-to-collision)
+ language-command consistency + a clipped GT-proximity guardrail. Group-relative advantages
(group mean = baseline), PPO-clipped importance ratio, KL-anchored to the frozen BC reference.
Flow-policy log-prob via a flow-matching-MSE surrogate. 5000 steps, peak LR 1e-6, KL coef 1.0.

## Load (openpi)

```python
from huggingface_hub import snapshot_download
import openpi.training.config as config
import openpi.policies.policy_config as policy_config

ckpt = snapshot_download("{repo_id}")
cfg = config.get_config("pi05_driving")
policy = policy_config.create_trained_policy(cfg, ckpt)
```

Contains `params/` (weights) and `assets/` (norm stats).{opt_note}
"""


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 40,
)
def upload(
    repo_id: str = "",
    exp_name: str = "grpo-5k-v2",
    step: int = 5000,
    private: bool = False,
    include_optimizer: bool = False,
) -> dict:
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    api = HfApi(token=token)
    user = api.whoami()["name"]
    if not repo_id:
        repo_id = f"{user}/pi05-driving-grpo-5k-checkpoint"

    ckpt = Path(CACHE_DIR) / "checkpoints" / OPENPI_CONFIG_NAME / exp_name / str(step)
    if not (ckpt / "params").exists():
        raise FileNotFoundError(f"no params/ under {ckpt} — checkpoint missing/incomplete")

    opt_note = (
        " Optimizer state (`train_state/`) is included for resuming."
        if include_optimizer
        else " Optimizer state (`train_state/`) omitted (inference-ready)."
    )
    readme = _README.format(exp_name=exp_name, step=step, repo_id=repo_id, opt_note=opt_note)
    readme_path = Path("/tmp/README.md")
    readme_path.write_text(readme)

    print(f"[hf] user={user} repo={repo_id} private={private} src={ckpt}", flush=True)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    ignore = None if include_optimizer else ["train_state/*", "**/train_state/**", "train_state/**"]
    api.upload_folder(
        folder_path=str(ckpt),
        repo_id=repo_id,
        repo_type="model",
        ignore_patterns=ignore,
        commit_message=f"Flow-GRPO post-trained pi05_driving {exp_name}/{step}",
    )
    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add model card",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[hf] uploaded -> {url}", flush=True)
    return {"repo_id": repo_id, "url": url, "user": user, "included_optimizer": include_optimizer}


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 5,
)
def delete_repo(repo_id: str) -> dict:
    """Delete a model repo using the original ('huggingface'=markmusic) token that created it."""
    import os

    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    api = HfApi(token=token)
    user = api.whoami()["name"]
    api.delete_repo(repo_id, repo_type="model", missing_ok=True)
    print(f"[hf] {user} deleted {repo_id}", flush=True)
    return {"deleted": repo_id, "by": user}


@app.function(
    image=image,
    volumes={CACHE_DIR: cache_volume},
    secrets=[modal.Secret.from_name("huggingface-fbarbosa")],
    timeout=60 * 10,
)
def upload_card(repo_id: str = "", readme_path: str = "/cache/dpo_card.md") -> dict:
    """Upload ONLY a model card (README.md) to a HF repo — no weights."""
    import os
    from pathlib import Path

    from huggingface_hub import HfApi

    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    api = HfApi(token=token)
    user = api.whoami()["name"]
    if not repo_id:
        repo_id = f"{user}/pi05-driving-dpo-cosmos"
    readme = Path(readme_path).read_text()
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    tmp = Path("/tmp/README.md")
    tmp.write_text(readme)
    api.upload_file(
        path_or_fileobj=str(tmp), path_in_repo="README.md",
        repo_id=repo_id, repo_type="model",
        commit_message="Add Cosmos-3-judged DPO model card",
    )
    url = f"https://huggingface.co/{repo_id}"
    print(f"[hf] {user} uploaded card -> {url}", flush=True)
    return {"repo_id": repo_id, "url": url, "user": user}


@app.local_entrypoint()
def main(
    repo_id: str = "",
    exp_name: str = "grpo-5k-v2",
    step: int = 5000,
    private: bool = False,
    include_optimizer: bool = False,
):
    import json

    res = upload.remote(
        repo_id=repo_id,
        exp_name=exp_name,
        step=step,
        private=private,
        include_optimizer=include_optimizer,
    )
    print(json.dumps(res, indent=2))
