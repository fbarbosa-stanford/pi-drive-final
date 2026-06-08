"""
Flow-GRPO composite reward smoke test on Modal (CPU-only; no policy loss).

  modal run grpo/modal_smoke.py
  modal run grpo/modal_smoke.py --legacy
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = "pi05-grpo-smoke"
_PKG_DIR = Path(__file__).resolve().parents[1]

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy")
    .add_local_dir(_PKG_DIR, remote_path="/app/pi_05_drives")
)


@app.function(image=image, timeout=60 * 10)
def smoke_ranker(legacy: bool = False, group_size: int = 12, horizon: int = 64) -> dict:
    import os

    os.chdir("/app/pi_05_drives")
    cmd = [
        sys.executable,
        "-m",
        "grpo.smoke_test",
        "--group-size",
        str(group_size),
        "--horizon",
        str(horizon),
    ]
    if legacy:
        cmd.append("--legacy")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"smoke_test failed with code {proc.returncode}")
    return {"status": "ok", "action_format": action_format}


@app.local_entrypoint()
def main(legacy: bool = False, group_size: int = 12, horizon: int = 64):
    result = smoke_ranker.remote(legacy=legacy, group_size=group_size, horizon=horizon)
    print(result)
