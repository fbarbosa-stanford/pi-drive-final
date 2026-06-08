"""Cosmos-3 reasoner as a VLM judge for DPO preference labeling.

Cosmos 3 (released 2026-05-31) is an omni-model; its *reasoner* (Cosmos3-Nano = 16B
with an 8B reasoner) is VQA-capable and AV-domain, so unlike Alpamayo-R1 it can take
our prompt + a rendered scene and RANK candidate trajectories.

Served via vLLM's OpenAI-compatible API (per the HF model card):
  vllm serve nvidia/Cosmos3-Nano --hf-overrides '{"architectures":["Cosmos3ReasonerForConditionalGeneration"]}' ...

Smoke first (de-risk the bleeding-edge stack + confirm it ranks sanely):
  modal run --detach dpo/modal_cosmos_judge.py::smoke_reason
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import modal

APP_NAME = "pi05-cosmos-judge"
CACHE_DIR = "/cache"
MODEL_ID = "nvidia/Cosmos3-Nano"
_PKG_DIR = Path(__file__).resolve().parents[1]

# Bleeding-edge stack: vllm 0.19.1 (cu128 path) + NVIDIA's vllm-cosmos3 plugin that
# registers Cosmos3ReasonerForConditionalGeneration. CUDA 12.8 base matches cu128.
cosmos_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git", "git-lfs", "build-essential", "ffmpeg")
    .pip_install("vllm==0.19.1")
    # vllm-cosmos3 depends on its sibling transformers-cosmos3 (NOT on PyPI). Install BOTH
    # in one pip command so the git URL satisfies the transitive dep (uv resolves this
    # automatically; pip needs the URL provided explicitly).
    .pip_install(
        "transformers-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/transformers-cosmos3",
        "vllm-cosmos3 @ git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3",
    )
    .pip_install("openai", "pillow", "matplotlib", "numpy", "huggingface_hub")
    .env({"HF_HOME": f"{CACHE_DIR}/hf", "VLLM_USE_V1": "1"})
    .add_local_dir(_PKG_DIR, remote_path="/app/pi_05_drives")
)

cache_volume = modal.Volume.from_name("pi05-cache", create_if_missing=True)
VOLUMES = {CACHE_DIR: cache_volume}
app = modal.App(APP_NAME)

VLLM_ARGS = [
    "vllm", "serve", MODEL_ID,
    "--hf-overrides", '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}',
    "--tensor-parallel-size", "1",
    "--mm-encoder-tp-mode", "data",
    "--async-scheduling",
    "--allowed-local-media-path", "/",
    "--media-io-kwargs", '{"video": {"num_frames": -1}}',
    "--port", "8000",
]


def _render_three_paths(png_path: str) -> None:
    """Top-down BEV: 1=straight, 2=hard-left, 3=hard-right. 'Straight' is the safe answer."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 40, 40)
    straight = np.zeros_like(x)
    left = 0.02 * x**2
    right = -0.02 * x**2

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, straight, "r-", lw=3, label="1 (red)")
    ax.plot(x, left, "g-", lw=3, label="2 (green)")
    ax.plot(x, right, "b-", lw=3, label="3 (blue)")
    ax.set_xlabel("x forward (m)")
    ax.set_ylabel("y left (m)")
    ax.set_title("3 candidate future paths (top-down)")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)
    fig.savefig(png_path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def _wait_for_server(base_url: str, timeout_s: int = 1200):
    import openai

    client = openai.OpenAI(api_key="EMPTY", base_url=base_url)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            models = client.models.list()
            if models.data:
                print(f"[cosmos] server ready after {int(time.time()-t0)}s", flush=True)
                return client, models.data[0].id
        except Exception:
            pass
        time.sleep(5)
    raise RuntimeError(f"vllm server not ready within {timeout_s}s")


@app.function(
    image=cosmos_image,
    gpu="H100",
    timeout=60 * 60,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def smoke_reason() -> dict:
    """De-risk: build stack, serve Cosmos3-Nano reasoner, ask it to rank 3 paths."""
    img = "/tmp/three_paths.png"
    _render_three_paths(img)
    image_url = Path(img).resolve().as_uri()

    print("[cosmos] starting vllm serve ...", flush=True)
    server = subprocess.Popen(VLLM_ARGS)
    try:
        client, model_id = _wait_for_server("http://localhost:8000/v1")
        prompt = (
            "You are an expert driving-safety evaluator. The top-down image shows three "
            "candidate future paths for a car that must DRIVE STRAIGHT AHEAD on a straight road: "
            "path 1 (red), path 2 (green), path 3 (blue). Rank them by how well they keep the car "
            "driving straight and safely. Respond on the first line EXACTLY as: "
            "BEST=<n>, WORST=<n>  (n in 1,2,3). Then give one sentence of reasoning."
        )
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=512,
            seed=0,
        )
        out = resp.choices[0].message.content
        print("[cosmos] ===== reasoner output =====", flush=True)
        print(out, flush=True)
        print("[cosmos] ============================", flush=True)
        return {"status": "ok", "output": out, "model": model_id}
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()


@app.function(
    image=cosmos_image,
    gpu="H100",
    timeout=60 * 60 * 8,
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface")],
    memory=64 * 1024,
)
def judge_candidates(tag: str = "cosmos_g2", limit: int = 0) -> dict:
    """Stage B: Cosmos ranks each Stage-A candidate image -> (chosen, rejected) labels.

    g=2 -> a direct pairwise preference. Reads /cache/cosmos_stage/<tag>/{manifest.json,imgs/},
    writes /cache/labels/cosmos3_<tag>.jsonl with chosen_idx/rejected_idx + the reasoning.
    """
    import json
    import re
    import subprocess
    from pathlib import Path

    stage = Path(CACHE_DIR) / "cosmos_stage" / tag
    manifest = json.loads((stage / "manifest.json").read_text())
    if limit > 0:
        manifest = manifest[:limit]
    imgs = stage / "imgs"

    prompt = (
        "The LEFT image is the forward camera from a car. The RIGHT image shows TWO candidate "
        "future driving paths (path 1 and path 2) top-down (x = forward, y = left, meters). "
        "Considering the scene — lane, road geometry, and any other vehicles — which path is the "
        "safer and more appropriate continuation that follows the road and the implied intent? "
        "Respond on the FIRST line EXACTLY: BETTER=<n>  (n is 1 or 2). Then one sentence of reasoning."
    )

    print(f"[stageB] starting vllm serve for {len(manifest)} clips ...", flush=True)
    server = subprocess.Popen(VLLM_ARGS)
    labels, n_ok = [], 0
    try:
        client, model_id = _wait_for_server("http://localhost:8000/v1")
        for e in manifest:
            img_uri = (imgs / f"{e['stem']}.png").resolve().as_uri()
            try:
                resp = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": img_uri}},
                        {"type": "text", "text": prompt},
                    ]}],
                    max_tokens=300, seed=0,
                )
                out = (resp.choices[0].message.content or "").strip()
            except Exception as ex:
                out = f"__error__: {ex}"
            m = re.search(r"BETTER\s*=\s*([12])", out)
            if not m:
                labels.append({"index": e["index"], "stem": e["stem"], "status": "unparsed", "raw": out[:300]})
                continue
            chosen_idx = int(m.group(1)) - 1
            labels.append({
                "index": e["index"], "stem": e["stem"], "status": "ok",
                "chosen_idx": chosen_idx, "rejected_idx": 1 - chosen_idx,
                "reasoning": out[:300],
            })
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"[stageB] judged {n_ok}/{len(manifest)}", flush=True)
                cache_volume.commit()
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except Exception:
            server.kill()

    labels_dir = Path(CACHE_DIR) / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    out_path = labels_dir / f"cosmos3_{tag}.jsonl"
    with out_path.open("w") as f:
        for r in labels:
            f.write(json.dumps(r) + "\n")
    cache_volume.commit()
    print(f"[stageB] wrote {n_ok}/{len(manifest)} ok labels -> {out_path}", flush=True)
    # surface a few verdicts for eyeballing
    for r in labels[:6]:
        print(f"[stageB] {r.get('status')} idx={r['index']} -> {r.get('reasoning', r.get('raw',''))[:160]}", flush=True)
    return {"tag": tag, "n_ok": n_ok, "n_total": len(manifest), "labels": str(out_path)}


if __name__ == "__main__":
    pass
