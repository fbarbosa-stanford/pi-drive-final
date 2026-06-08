#!/usr/bin/env python3
"""
camera_view.py — see the feed of all four onboard USB cameras at once.

The cart has four UVC-class USB cameras:

  - Front narrow  ELP-USBFHD01M-FV, 2.8-12mm VARIFOCAL lens, 30-115° FOV
                  depending on zoom, OV2710 1080p sensor.
                  (Amazon B01N8QBO2G)
  - Front wide    ELP 170° fisheye family, OV2710 1080p sensor,
                  ~170° lens / ~139° horizontal FOV.
                  (Amazon B01E8OC3EO)
  - Left wide     same fisheye family as "front wide"
  - Right wide    same fisheye family as "front wide"

All four are standard UVC so no drivers are needed — macOS uses
AVFoundation, Linux uses V4L2. Enumeration order on USB is not stable
across boots — the mapping in ``LABELS`` below is what's currently
plugged in on this machine. Each tile is annotated in the top-left
with its physical name (``front wide``, ``front narrow``, ``left``,
``right``); if a name doesn't match reality, re-order ``LABELS`` or
re-plug cables. We'll lock it into a ``cart.cameras`` config once the
``cart/`` package lands.

Bandwidth note: four MJPEG streams at 1080p30 will saturate a single
USB 2.0 controller. Default is 640x480 which fits comfortably. If one
tile goes black or locks up, try a lower resolution (``--width 320
--height 240``) or plug cameras across different USB controllers.

Usage:
    uv run python scripts/camera_view.py                      # auto-find 4
    uv run python scripts/camera_view.py --indices 0 1 2 3    # pin indices
    uv run python scripts/camera_view.py --width 320 --height 240
    uv run python scripts/camera_view.py --list               # scan and exit

Controls (while the window is focused):
    s        save a snapshot of the grid to ./camera_snapshots/
    q / Esc  quit
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Physical camera names, in the order they're assigned to OS indices
# after auto-find. This must match the physical enumeration on this
# machine; re-order if a tile doesn't match reality.
LABELS: list[str] = [
    "front wide",    # idx 0 — front fisheye
    "left",          # idx 1 — left fisheye
    "front narrow",  # idx 2 — varifocal narrow
    "right",         # idx 3 — right fisheye
]

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_COUNT = 4
# How far to scan when auto-finding cameras — a fresh macOS with a
# built-in FaceTime camera + 4 USB cameras can have gaps / duplicates,
# so scan a bit past the expected count.
MAX_INDEX_SCAN = 12
PREFERRED_FOURCC = "MJPG"


def _capture_backend() -> int:
    """Pick the OS-native capture backend so OpenCV doesn't guess poorly.

    On macOS the default sometimes falls back to a slow bridge; force
    AVFoundation. On Linux use V4L2 explicitly. Everywhere else let
    OpenCV choose.
    """
    system = platform.system()
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def open_camera(index: int, width: int, height: int,
                force_mjpg: bool = True) -> cv2.VideoCapture | None:
    """Open a camera by index and verify it actually produces a frame.

    Returns None if the device doesn't exist, won't open, or opens but
    won't serve a frame (all of those look the same to higher code).

    force_mjpg: when True (default), request MJPEG FourCC. Some H.264-native
    or "HD USB Camera"-style devices mis-behave when forced to MJPG and
    stall after a few frames — pass False to let OpenCV negotiate.
    """
    backend = _capture_backend()
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return None

    # Request MJPEG — the ELP modules go up to 30 fps @ 1080p with MJPEG
    # but cap out at ~5-6 fps with YUY2 at the same resolution.
    if force_mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PREFERRED_FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def auto_find_indices(count: int, width: int, height: int,
                      force_mjpg: bool = True) -> list[int]:
    """Scan indices 0..MAX_INDEX_SCAN and return the first ``count`` working ones."""
    found: list[int] = []
    for idx in range(MAX_INDEX_SCAN):
        cap = open_camera(idx, width, height, force_mjpg=force_mjpg)
        if cap is None:
            continue
        cap.release()
        found.append(idx)
        if len(found) >= count:
            break
    return found


def list_cameras(width: int, height: int) -> None:
    """Print every openable camera index and its default resolution."""
    backend = _capture_backend()
    print(f"Scanning indices 0..{MAX_INDEX_SCAN - 1} (backend: {backend})")
    any_found = False
    for idx in range(MAX_INDEX_SCAN):
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*PREFERRED_FOURCC))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        ok, _ = cap.read()
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        status = "OK" if ok else "opened but no frame"
        print(f"  [{idx}] {aw}x{ah}  {status}")
        cap.release()
        any_found = True
    if not any_found:
        print("  (nothing found)")


def annotate(frame: np.ndarray, label: str, idx: int, fps: float) -> np.ndarray:
    """Draw the label in the top-left + telemetry in the bottom-left.

    No colored border — keeps the frame clean and leaves the image
    itself as the signal. Text gets a black outline so it stays legible
    over any background.
    """
    out = frame.copy()
    cv2.putText(out, label, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (18, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    h, w = out.shape[:2]
    meta = f"idx {idx}  {w}x{h}  {fps:.0f} fps"
    cv2.putText(out, meta, (18, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(out, meta, (18, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return out


def make_grid(frames: list[np.ndarray], cols: int = 2) -> np.ndarray:
    """Pack frames into a rows × cols grid, padding with black if short."""
    rows = (len(frames) + cols - 1) // cols
    # Force every tile to the smallest shared dimensions so hstack lines up
    # even if the cameras are running at slightly different resolutions.
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    resized = [cv2.resize(f, (w, h)) for f in frames]
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))
    row_imgs = [np.hstack(resized[i * cols:(i + 1) * cols]) for i in range(rows)]
    return np.vstack(row_imgs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--indices", type=int, nargs="*", default=None,
        help="Camera indices (e.g. --indices 0 1 2 3). Omit to auto-find.",
    )
    parser.add_argument(
        "--count", type=int, default=DEFAULT_COUNT,
        help=f"Cameras to open when auto-finding (default {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--width", type=int, default=DEFAULT_WIDTH,
        help=f"Requested frame width (default {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height", type=int, default=DEFAULT_HEIGHT,
        help=f"Requested frame height (default {DEFAULT_HEIGHT}).",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Scan indices, print what's there, and exit.",
    )
    parser.add_argument(
        "--no-fourcc", action="store_true",
        help="Skip the MJPG FourCC request; let OpenCV negotiate the format. "
             "Useful when a camera opens but stalls after a few frames.",
    )
    args = parser.parse_args()
    force_mjpg = not args.no_fourcc

    if args.list:
        list_cameras(args.width, args.height)
        return 0

    if args.indices is None:
        print(f"[cams] auto-scanning for {args.count} cameras "
              f"(indices 0..{MAX_INDEX_SCAN - 1})...")
        indices = auto_find_indices(args.count, args.width, args.height,
                                    force_mjpg=force_mjpg)
        print(f"[cams] found indices: {indices}")
    else:
        indices = args.indices

    if not indices:
        print("ERROR: no working cameras found.")
        print("  Try: uv run python scripts/camera_view.py --list")
        return 1
    if len(indices) > len(LABELS):
        print(f"ERROR: only {len(LABELS)} labels defined "
              f"({LABELS}). Pass fewer indices.")
        return 1

    caps: list[tuple[int, cv2.VideoCapture]] = []
    for idx in indices:
        cap = open_camera(idx, args.width, args.height, force_mjpg=force_mjpg)
        if cap is None:
            print(f"[cams] WARN: failed to open index {idx}, skipping")
            continue
        caps.append((idx, cap))

    if not caps:
        print("ERROR: opened 0 cameras. Try `--indices` with explicit numbers.")
        return 1

    # Show what we actually got vs. what we requested — useful when the
    # driver silently quantizes resolution to the nearest supported mode.
    active_labels = LABELS[:len(caps)]
    name_width = max(len(name) for name in active_labels)
    for (idx, cap), name in zip(caps, active_labels):
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[cams] {name:<{name_width}} -> index {idx}   actual {aw}x{ah}")

    window = "Cart cameras   (s=snap, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    snapshots_dir = Path(__file__).resolve().parent.parent / "camera_snapshots"
    # Exponential-moving-average FPS per camera (avoids jitter in the readout).
    fps_state: dict[int, tuple[float, float]] = {idx: (time.time(), 0.0) for idx, _ in caps}

    try:
        while True:
            frames = []
            for (idx, cap), label in zip(caps, active_labels):
                ok, frame = cap.read()
                if not ok or frame is None:
                    frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, f"{label}: no frame (idx {idx})",
                        (20, args.height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255),
                        2, cv2.LINE_AA,
                    )

                last_t, last_fps = fps_state[idx]
                now = time.time()
                inst = 1.0 / (now - last_t) if now > last_t else 0.0
                # 0.9/0.1 EMA — smooth enough to read, still reacts to stalls.
                fps = 0.9 * last_fps + 0.1 * inst
                fps_state[idx] = (now, fps)

                frames.append(annotate(frame, label, idx, fps))

            grid = make_grid(frames)
            cv2.imshow(window, grid)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('s'):
                snapshots_dir.mkdir(exist_ok=True)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = snapshots_dir / f"cams_{ts}.png"
                cv2.imwrite(str(out), grid)
                print(f"[cams] saved {out}")
    except KeyboardInterrupt:
        pass
    finally:
        for _, cap in caps:
            cap.release()
        cv2.destroyAllWindows()
        print("[cams] done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
