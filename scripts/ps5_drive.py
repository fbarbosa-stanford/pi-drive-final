#!/usr/bin/env python3
"""
ps5_drive.py — Drive the cart with a PS5 DualSense controller.

Reads the DualSense over pygame and commands both cart subsystems with
authority clamped by ``limits.py``:

  - Left stick X  → steering column (ODrive **POSITION + TRAP_TRAJ**, same
    planner as ``main.py`` sweeps). ``integrated`` slews a held target angle;
    ``absolute`` maps stick position to angle.
  - R2 trigger    → gas pedal target (Arduino Mega over USB serial)
  - L2 trigger    → brake pedal target (Arduino Mega over USB serial)

Gas is always clamped to
``effective_gas_cap(PS5_GAS_LIMIT) = min(GAS_POT_MAX, GLOBAL_SPEED_LIMIT, PS5_GAS_LIMIT)``
so the top-level governor in ``limits.py`` wins even if this script is
asked to push harder.

Modes (``--mode``):

  full      (default) steering + pedals
  steering  ODrive only; pedals are left alone
  pedals    pedals only; ODrive is not even opened

Expected Arduino pedal protocol (USB serial, 115200 baud, newline-terminated):

  ``G <value>``   gas target pot value (0.0 .. GAS_POT_MAX)
  ``B <value>``   brake target pot value (0.0 .. BRAKE_POT_MAX)
  ``S``           stop both actuators immediately

Safety:

  - The pygame window must be focused for input.
  - Esc / Q / close-window → graceful stop of everything that was opened.
  - ``--dry-run`` skips all hardware and just prints what would be sent,
    handy for checking input mapping without the cart present.

Stick steering (``--stick-steering``):

  ``integrated`` — stick slews a **target column angle** (°); the ODrive
  **trap trajectory** tracks it with the same vel/accel caps as calibration
  sweeps (``main.py``). Release the stick and the target (and wheel) **hold**.

  ``absolute`` — stick position maps directly to target angle; center → 0°.

Usage:

    uv run python scripts/ps5_drive.py                              # full, integrated steer
    uv run python scripts/ps5_drive.py --stick-steering absolute    # wheel follows stick
    uv run python scripts/ps5_drive.py --mode steering              # steering only
    uv run python scripts/ps5_drive.py --mode pedals                # pedals only
    uv run python scripts/ps5_drive.py --dry-run                    # no hardware
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run-from-anywhere: make the project root importable so ``limits.py`` at
# the repo root resolves regardless of the current working directory.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pygame

from limits import (
    BRAKE_POT_MAX,
    FSD_GAS_LIMIT,
    GAS_POT_MAX,
    GLOBAL_SPEED_LIMIT,
    PS5_GAS_LIMIT,
    STEERING_MAX_DEG,
    STEERING_MIN_DEG,
    effective_gas_cap,
    motor_turns_to_steering_deg,
    steering_deg_to_motor_turns,
)

# --- Controller mapping (matches scripts/ps5_controller_test.py) -----------

AXIS_LEFT_X = 0
AXIS_L2 = 4
AXIS_R2 = 5

# Default deadzone (e.g. other scripts). Steering X uses ``STEER_STICK_DEADZONE`` so
# more physical travel maps to rate — a large deadzone + remap makes mid‑stick feel weak.
STICK_DEADZONE = 0.08
STEER_STICK_DEADZONE = 0.055

# Per-trigger calibration — some DualSense units/driver stacks don't quite
# report full-scale on the triggers; rescale so full-squeeze reads 1.0.
TRIGGER_MAX_L2 = 0.91
TRIGGER_MAX_R2 = 1.00

# --- Cart-side mapping ------------------------------------------------------

# Steering authority for PS5 driving. Clamped against the cart-wide soft
# limits, then further narrowed while we're still early-testing — widen
# this once the full lock-to-lock range is known and trusted.
PS5_STEERING_MAX_DEG = min(60.0, STEERING_MAX_DEG, -STEERING_MIN_DEG)

# Trapezoidal planner caps — **match ``main.py``** so PS5 steering uses the same
# consistent ODrive-generated motion as calibration sweeps (motor-side units).
PS5_TRAP_VEL_MAX = 8.0      # motor turns/s
PS5_TRAP_ACCEL_MAX = 15.0  # motor turns/s²
PS5_TRAP_DECEL_MAX = 15.0

# Integrated mode: how fast the **target angle** slews with full stick (°/s at |lx|=1).
# The wheel follows via TRAP_TRAJ; high values let the planner saturate at PS5_TRAP_*.
STEER_INTEGRATE_RATE_DPS = 320.0

# Motor current headroom: scale current_soft_max toward current_hard_max with
# demand ∈ [0,1] from |stick|, |d(lx)/dt|, and |d(ω_cmd)/dt| so fast reversals
# keep torque for acceleration instead of hitting an overly low soft limit.
PS5_STEER_CURRENT_BLEND_MIN = 0.45   # demand=0 → keep at least this fraction of headroom
PS5_STEER_CURRENT_HARD_CAP_A = 80.0  # never raise soft_max above this (motor / fuse sanity)

# Reference scales for mapping stick dynamics into current demand (tunable).
STICK_DLX_REF = 10.0                  # |dlx/dt| ≈ this → full “stick dynamics” contribution
VEL_CMD_ACCEL_REF = 120.0             # motor turns/s² reference for ω_cmd slew

# Main-loop rate for reading the controller and pushing commands.
CONTROL_HZ = 50.0

MODES = ("full", "steering", "pedals")
STICK_STEERING = ("absolute", "integrated")

# --- UI colors --------------------------------------------------------------

WINDOW_W, WINDOW_H = 680, 360
BG = (18, 20, 26)
TEXT = (230, 232, 238)
MUTED = (130, 135, 150)
ACCENT = (90, 170, 255)
GOOD = (120, 220, 140)
WARN = (240, 180, 90)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def apply_deadzone(value: float, deadzone: float) -> float:
    """Rescale a stick value so |v| <= deadzone maps to 0 and the remainder
    is re-expanded to -1..1 for smooth past-the-deadzone behavior."""
    if abs(value) <= deadzone:
        return 0.0
    sign = 1.0 if value > 0 else -1.0
    return sign * (abs(value) - deadzone) / (1.0 - deadzone)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def read_trigger(js, axis: int, scale: float) -> float:
    """Read an SDL trigger axis (-1 released, +1 squeezed) and return 0..1."""
    if js.get_numaxes() <= axis:
        return 0.0
    raw = js.get_axis(axis)
    value = max(0.0, min(1.0, (raw + 1.0) / 2.0))
    return min(1.0, value / scale) if scale > 0 else 0.0


def find_arduino_port() -> str | None:
    """Best-effort Mega auto-detect; returns None if nothing obvious found.

    Avoids grabbing the ODrive (VID 0x1209 / 0x0483). Prefers USB-VID
    matches for Arduino / clones / FTDI, then falls back to the
    shortest-named usbmodem-ish device.
    """
    try:
        import serial.tools.list_ports  # type: ignore
    except ImportError:
        return None

    ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}
    ODRIVE_VIDS = {0x1209, 0x0483}

    ports = [
        p for p in serial.tools.list_ports.comports()
        if any(tag in p.device for tag in ("usbmodem", "ttyACM", "ttyUSB", "usbserial"))
    ]
    arduino = [p for p in ports if p.vid in ARDUINO_VIDS]
    if arduino:
        return arduino[0].device
    non_odrive = [p for p in ports if p.vid not in ODRIVE_VIDS]
    if non_odrive:
        return min(non_odrive, key=lambda p: len(p.device)).device
    return None


# ---------------------------------------------------------------------------
# hardware wrappers
# ---------------------------------------------------------------------------

# Host side of the heartbeat contract with firmware/pedal_control.ino.
# Every successful write to the Mega resets its watchdog; if the host
# goes quiet for longer than this, the firmware drops both pedals. We
# mirror that threshold here so the UI can turn red just before the cart
# actually fails safe, and so the main loop aborts steering too (losing
# pedals while still commanding the wheel is worse than stopping both).
HEARTBEAT_TIMEOUT_S = 0.30


class PedalLink:
    """Sends gas/brake target pot values to the Arduino Mega over USB serial
    and watches the link health.

    Gas and brake are clamped against their pot-max ceilings as a
    last-ditch safety net — callers are expected to have already applied
    the per-mode cap from ``limits.effective_gas_cap()``.

    Heartbeat contract:
      - Every successful write timestamps ``self.last_ok_s`` (monotonic).
      - Any OS-level serial failure (cable yanked, port vanished, write
        throws OSError / SerialException) sets ``self.faulted = True`` and
        stops touching the port. The main loop should check ``.faulted``
        and exit — there's no automatic reconnect here by design: if we
        lose pedals, we can't trust the rest of the cart state either.
      - The firmware also runs its own watchdog (see
        ``sketches/pedal_control/pedal_control.ino``). Even if this Python
        process is frozen or killed, the Mega drops both actuators to MIN
        ~300 ms after the last received byte.

    Under ``--dry-run`` the serial port is never opened; all writes are
    no-ops but ``last_ok_s`` still advances so the UI looks normal.
    """

    def __init__(self, port: str | None, baud: int = 115200, dry_run: bool = False):
        self.ser = None
        self.dry_run = dry_run
        self.faulted = False
        self.fault_reason: str | None = None
        self.last_ok_s = time.monotonic()

        if dry_run:
            print("[pedals] dry-run: commands will NOT be sent over serial.")
            return

        import serial  # type: ignore

        resolved = port or find_arduino_port()
        if resolved is None:
            raise RuntimeError(
                "Couldn't auto-detect the Arduino Mega. Pass --arduino-port /dev/tty.X "
                "(run scripts/sensor_test.py --list to see attached ports)."
            )
        print(f"[pedals] opening {resolved} @ {baud} baud")
        self.ser = serial.Serial(resolved, baud, timeout=0.5)
        # Mega resets on port open; give the bootloader a moment before the
        # first write so commands aren't sent into the DTR-reset void.
        time.sleep(2.0)

    @property
    def heartbeat_age_s(self) -> float:
        """Seconds since the last successful write (or link init)."""
        return time.monotonic() - self.last_ok_s

    @property
    def healthy(self) -> bool:
        return not self.faulted and self.heartbeat_age_s < HEARTBEAT_TIMEOUT_S

    def _mark_fault(self, reason: str) -> None:
        if self.faulted:
            return
        self.faulted = True
        self.fault_reason = reason
        print(f"[pedals] FAULT: {reason} — bailing out.")

    def send(self, gas: float, brake: float) -> None:
        """Push a gas+brake target pair. Records heartbeat on success."""
        gas = clamp(gas, 0.0, GAS_POT_MAX)
        brake = clamp(brake, 0.0, BRAKE_POT_MAX)

        if self.dry_run:
            self.last_ok_s = time.monotonic()
            return
        if self.ser is None or self.faulted:
            return

        payload = f"G {gas:.3f}\nB {brake:.3f}\n".encode("ascii")
        try:
            self.ser.write(payload)
            self.last_ok_s = time.monotonic()
        except Exception as e:  # SerialException, OSError, etc — all fatal
            self._mark_fault(f"serial write failed: {e!r}")

    def stop(self) -> None:
        """Best-effort 'release both pedals NOW' on the way out.

        Repeated writes because the first one sometimes gets swallowed if
        the port is already half-gone (USB yank race). If this fails the
        firmware's own heartbeat watchdog will do the same thing ~300 ms
        after the last byte it saw anyway.
        """
        if self.dry_run or self.ser is None:
            print("[pedals] STOP")
            return
        for _ in range(3):
            try:
                self.ser.write(b"S\n")
                time.sleep(0.02)
            except Exception:
                break

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass


def _odrive_cfg(obj: object) -> object:
    """Return ``obj.config`` when it exists; otherwise the flat ``obj`` (libodrive style).

    Never fall back to ``obj`` when ``config`` is missing on **motor** — motor
    fields like ``current_soft_max`` only exist on ``motor.config`` in legacy
    layouts, not on ``motor`` itself.
    """
    if hasattr(obj, "config"):
        return obj.config
    return obj


def _motor_amp_limits(axis) -> tuple[float | None, float | None]:
    """``(current_soft_max, current_hard_max)`` in A, or ``(None, None)`` if not on this API."""
    m = axis.motor
    if hasattr(m, "config"):
        c = m.config
        if hasattr(c, "current_soft_max") and hasattr(c, "current_hard_max"):
            return float(c.current_soft_max), float(c.current_hard_max)
    return None, None


def _set_motor_current_soft_max(axis, value: float) -> None:
    m = axis.motor
    if hasattr(m, "config") and hasattr(m.config, "current_soft_max"):
        m.config.current_soft_max = value


class SteeringLink:
    """ODrive S1 wrapper for PS5 steering.

    Both ``integrated`` and ``absolute`` use **POSITION_CONTROL** and
    **TRAP_TRAJ** with the same ``vel_limit`` / ``accel_limit`` / ``decel_limit``
    as ``main.py`` calibration sweeps so motion is generated consistently on
    the ODrive, not by host-side velocity filtering.
    """

    def __init__(self, dry_run: bool = False, stick_steering: str = "integrated"):
        self.dry_run = dry_run
        self.stick_steering = stick_steering
        self.odrv = None
        self.axis = None
        self.start_pos = 0.0
        self._AxisState = None
        self._InputMode = None
        self._saved_current_soft = None
        self._saved_current_hard = None
        self._prev_target_motor = 0.0
        self._prev_vel_cmd_motor = 0.0

        if dry_run:
            print("[steering] dry-run: ODrive will NOT be opened.")
            return

        import odrive  # type: ignore
        from odrive.enums import AxisState, ControlMode, InputMode  # type: ignore

        self._AxisState = AxisState
        self._InputMode = InputMode

        print("[steering] connecting to ODrive...")
        self.odrv = odrive.find_any(timeout=10)
        self.axis = self.odrv.axis0
        print(
            f"[steering] connected: serial {self.odrv.serial_number}, "
            f"bus {self.odrv.vbus_voltage:.1f}V"
        )

        if self.axis.active_errors != 0:
            print(f"[steering] clearing active errors: {self.axis.active_errors}")
            self.odrv.clear_errors()
            time.sleep(0.3)

        self._controller_cfg = _odrive_cfg(self.axis.controller)
        self._trap_traj_cfg = _odrive_cfg(self.axis.trap_traj)

        self._saved_current_soft, self._saved_current_hard = _motor_amp_limits(self.axis)
        if self._saved_current_soft is not None and self._saved_current_hard is not None:
            print(
                f"[steering] motor current_soft_max={self._saved_current_soft:.2f}A  "
                f"current_hard_max={self._saved_current_hard:.2f}A "
                f"(PS5 scales soft toward hard under demand)"
            )
        else:
            print(
                "[steering] motor amp limits not on ``motor.config`` (ODrive 0.6+ layout) — "
                "skipping dynamic current_soft_max scaling"
            )

        mode_label = "integrated (trap target slew)" if stick_steering == "integrated" else "absolute"
        print(f"[steering] mode: POSITION + TRAP_TRAJ — {mode_label}")
        print(
            f"[steering] trap_traj vel/accel/decel = "
            f"{PS5_TRAP_VEL_MAX}/{PS5_TRAP_ACCEL_MAX}/{PS5_TRAP_DECEL_MAX} "
            f"(same as main.py sweeps)"
        )
        self._controller_cfg.control_mode = ControlMode.POSITION_CONTROL
        self._controller_cfg.input_mode = InputMode.TRAP_TRAJ
        self._trap_traj_cfg.vel_limit = PS5_TRAP_VEL_MAX
        self._trap_traj_cfg.accel_limit = PS5_TRAP_ACCEL_MAX
        self._trap_traj_cfg.decel_limit = PS5_TRAP_DECEL_MAX

        self.axis.requested_state = AxisState.CLOSED_LOOP_CONTROL
        time.sleep(0.35)
        if self.axis.current_state != AxisState.CLOSED_LOOP_CONTROL:
            raise RuntimeError(
                f"ODrive failed to enter closed-loop control: "
                f"state={self.axis.current_state}, disarm={self.axis.disarm_reason}"
            )

        self.start_pos = self.axis.pos_estimate
        print(
            f"[steering] zero reference = {self.start_pos:.4f} turns "
            f"(whatever the wheel looks like right now is 0°)"
        )

    def column_deg_estimate(self) -> float:
        """Column angle (deg) relative to startup, from encoder."""
        if self.dry_run or self.axis is None:
            return 0.0
        return motor_turns_to_steering_deg(self.axis.pos_estimate - self.start_pos)

    def _apply_dynamic_current(self, demand: float) -> None:
        """Raise current_soft_max toward hard_max when demand is high (0..1)."""
        if self.dry_run or self.axis is None:
            return
        if self._saved_current_soft is None or self._saved_current_hard is None:
            return
        demand = clamp(float(demand), 0.0, 1.0)
        soft0 = self._saved_current_soft
        hard = min(self._saved_current_hard, PS5_STEER_CURRENT_HARD_CAP_A)
        if hard <= soft0 * 1.01:
            return
        span = hard - soft0
        head = PS5_STEER_CURRENT_BLEND_MIN + (1.0 - PS5_STEER_CURRENT_BLEND_MIN) * demand
        target = soft0 + span * head
        target = min(hard, max(soft0, target))
        try:
            _set_motor_current_soft_max(self.axis, target)
        except Exception:
            pass

    def _current_demand(
        self,
        lx: float,
        dlx_dt: float,
        vel_cmd_motor: float,
        prev_vel_cmd_motor: float,
        dt_s: float,
    ) -> float:
        a = abs(lx)
        b = min(1.0, abs(dlx_dt) / STICK_DLX_REF)
        c = 0.0
        if dt_s > 1e-6:
            c = min(1.0, abs(vel_cmd_motor - prev_vel_cmd_motor) / dt_s / VEL_CMD_ACCEL_REF)
        return max(a, 0.55 * b + 0.55 * c)

    def command_deg(
        self,
        column_deg: float,
        lx: float = 0.0,
        dlx_dt: float = 0.0,
        dt_s: float = 1.0 / CONTROL_HZ,
    ) -> None:
        """Send column angle (deg) vs startup as a trap-traj ``input_pos`` setpoint."""
        column_deg = clamp(column_deg, -PS5_STEERING_MAX_DEG, PS5_STEERING_MAX_DEG)
        if self.dry_run or self.axis is None:
            return

        target_motor = steering_deg_to_motor_turns(column_deg)
        vel_cmd_motor = (target_motor - self._prev_target_motor) / max(dt_s, 1e-6)
        self._prev_target_motor = target_motor

        self.axis.controller.input_pos = self.start_pos + target_motor
        dem = self._current_demand(
            lx, dlx_dt, vel_cmd_motor, self._prev_vel_cmd_motor, dt_s,
        )
        self._prev_vel_cmd_motor = vel_cmd_motor
        self._apply_dynamic_current(dem)

    def stop(self) -> None:
        if self.dry_run or self.axis is None:
            return
        try:
            self.axis.controller.input_pos = self.start_pos
            time.sleep(0.25)
            if self._saved_current_soft is not None:
                _set_motor_current_soft_max(self.axis, self._saved_current_soft)
            self._controller_cfg.input_mode = self._InputMode.PASSTHROUGH
            self.axis.requested_state = self._AxisState.IDLE
        except Exception as e:
            print(f"[steering] stop failed: {e}")


# ---------------------------------------------------------------------------
# controller + UI
# ---------------------------------------------------------------------------

def init_controller(index: int):
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERROR: no controllers detected. Pair the DualSense first,")
        print("  then confirm with: uv run python scripts/ps5_controller_test.py --list")
        sys.exit(1)
    if index >= pygame.joystick.get_count():
        print(
            f"ERROR: --index {index} but only "
            f"{pygame.joystick.get_count()} controller(s) connected."
        )
        sys.exit(1)
    js = pygame.joystick.Joystick(index)
    js.init()
    name = js.get_name()
    print(f"[ps5] using [{index}]: {name}")
    if not any(tag in name for tag in ("DualSense", "Wireless Controller", "PS5")):
        print("[ps5] WARNING: name doesn't look like a DualSense; axes may be mapped differently.")
    return js


def draw_ui(screen, font, font_small, state: dict) -> None:
    screen.fill(BG)

    title = font.render(f"PS5 drive — mode: {state['mode'].upper()}", True, ACCENT)
    screen.blit(title, (18, 14))
    sub = font_small.render(
        f"{'DRY RUN — nothing sent to hardware' if state['dry_run'] else 'live hardware'}",
        True, WARN if state['dry_run'] else MUTED,
    )
    screen.blit(sub, (18, 48))

    y = 82
    steer_mode = state.get("stick_steering", "integrated")
    steer_hint = "trap slew" if steer_mode == "integrated" else "absolute"
    lines = [
        ("steer",
         f"{state['steer_deg']:+6.1f}°   (stick X {state['lx']:+.2f}, {steer_hint}, "
         f"cap ±{PS5_STEERING_MAX_DEG:.0f}°)",
         TEXT if state['control_steering'] else MUTED),
        ("gas target",
         f"{state['gas']:.3f}    (R2 {state['r2']:.2f}, cap {state['gas_cap']:.3f})",
         GOOD if state['gas'] > 0.01 and state['control_pedals'] else MUTED),
        ("brake target",
         f"{state['brake']:.3f}    (L2 {state['l2']:.2f}, cap {BRAKE_POT_MAX:.3f})",
         WARN if state['brake'] > 0.01 and state['control_pedals'] else MUTED),
    ]
    for label, value, color in lines:
        screen.blit(font_small.render(label, True, MUTED), (18, y))
        screen.blit(font.render(value, True, color), (160, y - 4))
        y += 32

    y += 12
    caps = [
        f"GAS_POT_MAX         = {GAS_POT_MAX:.3f}   (hardware ceiling)",
        f"GLOBAL_SPEED_LIMIT  = {GLOBAL_SPEED_LIMIT:.3f}   (top-level, applies to every mode)",
        f"PS5_GAS_LIMIT       = {PS5_GAS_LIMIT:.3f}",
        f"FSD_GAS_LIMIT       = {FSD_GAS_LIMIT:.3f}   (reference — autonomy only)",
        f"effective PS5 cap   = {state['gas_cap']:.3f}   (min of the three above)",
    ]
    for c in caps:
        screen.blit(font_small.render(c, True, MUTED), (18, y))
        y += 18

    # Heartbeat indicator — only meaningful when pedals are being driven.
    if state["control_pedals"]:
        hb_age = state["hb_age_s"]
        if state["hb_faulted"]:
            hb_color = (235, 100, 100)
            hb_text = f"PEDAL LINK FAULTED — {state['hb_fault_reason']}"
        elif hb_age > HEARTBEAT_TIMEOUT_S:
            hb_color = (235, 100, 100)
            hb_text = f"HEARTBEAT STALE  {hb_age * 1000:.0f} ms  (firmware failsafe imminent)"
        elif hb_age > HEARTBEAT_TIMEOUT_S * 0.5:
            hb_color = WARN
            hb_text = f"heartbeat slow   {hb_age * 1000:.0f} ms"
        else:
            hb_color = GOOD
            hb_text = f"heartbeat OK     {hb_age * 1000:.0f} ms"
        screen.blit(font_small.render(hb_text, True, hb_color), (18, y + 6))

    hint = font_small.render(
        "Esc / Q / close window → graceful stop.", True, MUTED,
    )
    screen.blit(hint, (18, WINDOW_H - 26))
    pygame.display.flip()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=MODES, default="full",
        help="Which subsystems to actually drive (default: full).",
    )
    parser.add_argument(
        "--index", type=int, default=0,
        help="Controller index when multiple are paired (default 0).",
    )
    parser.add_argument(
        "--arduino-port", default=None,
        help="Serial device for the Mega (e.g. /dev/tty.usbmodem1401). "
             "Auto-detects when omitted.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Don't open the ODrive or serial port — just read inputs and draw the UI.",
    )
    parser.add_argument(
        "--stick-steering", choices=STICK_STEERING, default="integrated",
        help="integrated = trap-traj target slew (hold at release); absolute = stick maps to angle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    control_steering = args.mode in ("full", "steering")
    control_pedals = args.mode in ("full", "pedals")

    gas_cap = effective_gas_cap(PS5_GAS_LIMIT)
    print(
        f"[limits] effective PS5 gas cap = {gas_cap:.3f}  "
        f"(min of GAS_POT_MAX={GAS_POT_MAX}, "
        f"GLOBAL_SPEED_LIMIT={GLOBAL_SPEED_LIMIT}, "
        f"PS5_GAS_LIMIT={PS5_GAS_LIMIT})"
    )
    print(f"[mode] {args.mode}: steering={control_steering}, pedals={control_pedals}")
    print(f"[stick-steering] {args.stick_steering}")

    js = init_controller(args.index)

    pedals: PedalLink | None = None
    steering: SteeringLink | None = None
    exit_code = 0

    try:
        if control_pedals:
            pedals = PedalLink(args.arduino_port, dry_run=args.dry_run)
        if control_steering:
            steering = SteeringLink(
                dry_run=args.dry_run, stick_steering=args.stick_steering,
            )

        screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        pygame.display.set_caption(
            f"PS5 drive — {args.mode} ({args.stick_steering} steer)"
        )
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Menlo", 22, bold=True)
        font_small = pygame.font.SysFont("Menlo", 14)

        steer_target_deg = 0.0
        if (
            args.stick_steering == "integrated"
            and control_steering
            and steering is not None
            and not steering.dry_run
        ):
            # Match trap target to the column angle at launch so we do not
            # command a jump toward 0° while the wheel is already offset.
            steer_target_deg = steering.column_deg_estimate()

        prev_lx = 0.0
        last_mono = time.monotonic()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print("[ps5] controller disconnected — stopping.")
                    running = False

            lx_raw = js.get_axis(AXIS_LEFT_X) if js.get_numaxes() > AXIS_LEFT_X else 0.0
            lx = apply_deadzone(lx_raw, STEER_STICK_DEADZONE)
            l2 = read_trigger(js, AXIS_L2, TRIGGER_MAX_L2)
            r2 = read_trigger(js, AXIS_R2, TRIGGER_MAX_R2)

            now = time.monotonic()
            dt_s = now - last_mono
            last_mono = now
            if dt_s <= 0.0 or dt_s > 0.25:
                dt_s = 1.0 / CONTROL_HZ

            raw_dlx = (lx - prev_lx) / dt_s if dt_s > 0 else 0.0
            dlx_dt = clamp(raw_dlx, -28.0, 28.0)

            if args.stick_steering == "integrated":
                steer_target_deg += lx * STEER_INTEGRATE_RATE_DPS * dt_s
                steer_target_deg = clamp(
                    steer_target_deg,
                    -PS5_STEERING_MAX_DEG,
                    PS5_STEERING_MAX_DEG,
                )
                if (
                    control_steering
                    and steering is not None
                    and not steering.dry_run
                ):
                    steer_deg = steering.column_deg_estimate()
                else:
                    steer_deg = steer_target_deg
            else:
                steer_deg = lx * PS5_STEERING_MAX_DEG

            prev_lx = lx

            gas = r2 * gas_cap
            brake = l2 * BRAKE_POT_MAX

            # Drive pedals first so the heartbeat ticks before we commit
            # to moving the steering wheel this frame.
            if control_pedals and pedals is not None:
                pedals.send(gas, brake)
                if pedals.faulted:
                    # Pedal link is gone — the Mega will retract both
                    # pedals on its own watchdog in ~300 ms, but we must
                    # also stop asking the ODrive to steer an uncommanded
                    # cart. Bail out of the loop; the finally-block idles
                    # the ODrive and sends a final S (best effort).
                    print("[main] pedal link faulted — aborting control loop.")
                    running = False

            if running and control_steering and steering is not None:
                cmd_deg = (
                    steer_target_deg
                    if args.stick_steering == "integrated"
                    else steer_deg
                )
                steering.command_deg(cmd_deg, lx=lx, dlx_dt=dlx_dt, dt_s=dt_s)

            draw_ui(screen, font, font_small, {
                "mode": args.mode,
                "dry_run": args.dry_run,
                "control_steering": control_steering,
                "control_pedals": control_pedals,
                "stick_steering": args.stick_steering,
                "lx": lx, "l2": l2, "r2": r2,
                "steer_deg": steer_deg,
                "gas": gas, "brake": brake,
                "gas_cap": gas_cap,
                "hb_age_s": pedals.heartbeat_age_s if pedals is not None else 0.0,
                "hb_faulted": pedals.faulted if pedals is not None else False,
                "hb_fault_reason": pedals.fault_reason if pedals is not None else None,
            })
            clock.tick(CONTROL_HZ)

    except KeyboardInterrupt:
        print("\n[main] KeyboardInterrupt — stopping.")
    except Exception as e:
        print(f"[main] fatal: {e}")
        exit_code = 1
    finally:
        if pedals is not None:
            pedals.stop()
            pedals.close()
        if steering is not None:
            steering.stop()
        pygame.quit()
        print("[main] done.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
