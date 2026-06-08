"""Actor export with an atomic file swap (the hot-reload seam).

The learner periodically snapshots the **actor only** (deterministic μ path) so
the future Thor runtime can hot-reload weights without touching the live module
or the buffer/optimiser state. The write is atomic (``os.replace`` on a temp in
the same dir) so a reader never observes a half-written artifact.

Two formats:
  - ``torchscript`` -- a self-contained ``ScriptModule`` of the actor's
    deterministic forward (μ only), portable to the friend's PyTorch/TensorRT
    runtime; the default.
  - ``state_dict`` -- raw weights, for re-loading into an identical actor.
"""

from __future__ import annotations

import copy
import os
import tempfile

import torch

from .config import RLTConfig
from .nets import GaussianActor


def export_actor(
    actor: GaussianActor,
    path: str | None = None,
    fmt: str | None = None,
    cfg: RLTConfig | None = None,
) -> str:
    """Snapshot ``actor`` to ``path`` atomically. Returns the path written."""
    path = path or (cfg.export_path if cfg else "actor_latest.pt")
    fmt = fmt or (cfg.export_fmt if cfg else "torchscript")

    # export a CPU eval copy so the artifact is device-independent and the live
    # actor's mode/device are untouched.
    snapshot = copy.deepcopy(actor).to("cpu").eval()

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    os.close(fd)
    try:
        if fmt == "torchscript":
            with torch.no_grad():
                scripted = torch.jit.script(snapshot)
            scripted.save(tmp)
        elif fmt == "state_dict":
            torch.save(snapshot.state_dict(), tmp)
        else:
            raise ValueError(f"unknown export_fmt: {fmt!r}")
        os.replace(tmp, path)  # atomic within the same filesystem
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path
