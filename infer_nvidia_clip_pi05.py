"""
Deprecated entrypoint name — use train_modal_pi05_lora_nvidia_driving.py instead.

  # All 11 unseen val videos:
  modal run --detach train_modal_pi05_lora_nvidia_driving.py::run_batch_inference --no-include-train

  # Single clip:
  modal run train_modal_pi05_lora_nvidia_driving.py::infer_main --val-episode 36
"""

from __future__ import annotations

import sys
from pathlib import Path

for _p in (str(Path(__file__).resolve().parent), "/app/pi_05_drives"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export so `modal run infer_nvidia_clip_pi05.py::infer_batch_main` still resolves
# when Modal mounts this file together with train_modal (same directory).
from train_modal_pi05_lora_nvidia_driving import (  # noqa: E402
    app,
    infer_batch_main,
    infer_main,
    run_batch_inference,
    run_clip_inference,
)

__all__ = [
    "app",
    "infer_batch_main",
    "infer_main",
    "run_batch_inference",
    "run_clip_inference",
]
