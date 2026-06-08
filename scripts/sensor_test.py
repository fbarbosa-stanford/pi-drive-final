#!/usr/bin/env python3
"""
sensor_test.py — Validate GT-U7 GPS wiring via Arduino Mega passthrough.

Run `sketches/sensor_validation/sensor_validation.ino` on the Mega first,
then run this script. Reads the USB serial stream, parses NMEA sentences,
and reports fix status.

IMU support is intentionally removed until the replacement BerryIMU arrives.

Usage:
    uv run python scripts/sensor_test.py                     # auto-detect Mega
    uv run python scripts/sensor_test.py --port /dev/tty.X   # explicit port
    uv run python scripts/sensor_test.py --list              # list serial ports and quit
"""

import argparse
import sys
import time

import pynmea2
import serial
import serial.tools.list_ports

ARDUINO_VIDS = {0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4}  # Arduino, clones, FTDI, SiLabs
ODRIVE_VIDS = {0x1209, 0x0483}  # ODrive, ST — never pick these


def list_ports_verbose() -> None:
    """Print every serial port with USB descriptors so the user can pick one."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found.")
        return
    print(f"{'Device':<42} {'VID:PID':<12} {'Manufacturer / Product':<40} Serial")
    print("-" * 120)
    for p in ports:
        vidpid = f"{p.vid:04x}:{p.pid:04x}" if p.vid else "-"
        who = f"{p.manufacturer or '?'} / {p.product or '?'}"
        print(f"{p.device:<42} {vidpid:<12} {who:<40} {p.serial_number or '-'}")


def find_arduino_port() -> str:
    """
    Find the Arduino Mega's USB serial port.

    Strategy:
      1. Prefer ports whose VID matches a known Arduino / clone vendor.
      2. Explicitly reject ports whose VID matches the ODrive.
      3. Fall back to the shortest usbmodem name, which on macOS is
         typically the Arduino (ODrive uses its long USB serial).
    """
    ports = [
        p for p in serial.tools.list_ports.comports()
        if any(tag in p.device for tag in ("usbmodem", "usbserial", "ttyACM", "ttyUSB"))
    ]
    if not ports:
        print("ERROR: No USB serial devices found. Plug in the Mega and try again.")
        print("Run with --list to see what's attached.")
        sys.exit(1)

    arduino_ports = [p for p in ports if p.vid in ARDUINO_VIDS]
    non_odrive = [p for p in ports if p.vid not in ODRIVE_VIDS]

    if arduino_ports:
        chosen = arduino_ports[0]
        reason = f"matched Arduino VID {chosen.vid:04x}"
    elif non_odrive:
        chosen = min(non_odrive, key=lambda p: len(p.device))
        reason = "fallback: shortest non-ODrive device name"
    else:
        print("ERROR: Only ODrive-looking ports found. Is the Mega plugged in?")
        print("Run with --list to see what's attached.")
        sys.exit(1)

    if len(ports) > 1:
        print("Multiple serial ports found:")
        for p in ports:
            marker = "  -> " if p.device == chosen.device else "     "
            vidpid = f"{p.vid:04x}:{p.pid:04x}" if p.vid else "no-vid"
            product = p.product or "?"
            print(f"{marker}{p.device}  ({vidpid}, {product})")
        print(f"Picked {chosen.device} ({reason}).")
        print("Override with --port /dev/tty.X if wrong.")
    return chosen.device


def _safe(fn, default=None):
    """pynmea2 property accessors raise on malformed fields; wrap them."""
    try:
        return fn()
    except (ValueError, AttributeError, TypeError):
        return default


def _as_float(v) -> float:
    try:
        return float(v) if v not in (None, "") else 0.0
    except (TypeError, ValueError):
        return 0.0


def handle_nmea(line: str, state: dict) -> None:
    try:
        msg = pynmea2.parse(line)
    except (pynmea2.ParseError, pynmea2.SentenceTypeError, pynmea2.ChecksumError):
        state["gps_invalid"] += 1
        return
    state["gps_valid"] += 1

    if isinstance(msg, pynmea2.GGA):
        fix = {
            0: "NO FIX",
            1: "GPS FIX",
            2: "DGPS FIX",
            4: "RTK FIX",
            5: "RTK FLOAT",
        }.get(_safe(lambda: int(msg.gps_qual), 0), "UNKNOWN")
        lat = _as_float(_safe(lambda: msg.latitude))
        lon = _as_float(_safe(lambda: msg.longitude))
        alt = _as_float(_safe(lambda: msg.altitude))
        sats = _safe(lambda: int(msg.num_sats), 0) or 0
        print(
            f"[GPS/GGA] {fix:10} sats={sats:>2} "
            f"lat={lat:10.6f} lon={lon:11.6f} alt={alt:.1f}m"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", help="Serial device path (e.g. /dev/tty.usbmodem1401)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    parser.add_argument("--list", action="store_true", help="List serial ports and exit")
    args = parser.parse_args()

    if args.list:
        list_ports_verbose()
        return

    port = args.port or find_arduino_port()
    print(f"Opening {port} @ {args.baud} baud...")

    try:
        ser = serial.Serial(port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: Could not open {port}: {e}")
        if "Resource busy" in str(e) or "could not open port" in str(e):
            print()
            print("  -> Another program probably owns this port.")
            print("     Close the Arduino IDE Serial Monitor / Plotter, then retry.")
            print("     (Only one process can have the port open at a time on macOS.)")
        sys.exit(1)

    # Arduino resets on serial open; give it time to boot. Don't flush the
    # input buffer — we want to see the boot banner.
    time.sleep(2)

    state = {"gps_valid": 0, "gps_invalid": 0}
    start_time = time.time()
    last_stats = time.time()

    print("-" * 70)
    print("Reading GPS... (Ctrl+C to stop)")
    print("-" * 70)

    try:
        while True:
            raw = ser.readline()
            if raw:
                line = raw.decode("ascii", errors="replace").strip()
                if line:
                    if line.startswith("$"):
                        handle_nmea(line, state)
                    elif line.startswith("INFO,"):
                        print(f"[INFO]    {line[5:]}")

            if time.time() - last_stats > 5:
                elapsed = time.time() - start_time
                gps_rate = state["gps_valid"] / elapsed if elapsed else 0
                print(
                    f"--- {elapsed:.0f}s | "
                    f"GPS: {state['gps_valid']} valid @ {gps_rate:.1f}/s ---"
                )
                if elapsed > 5 and state["gps_valid"] == 0:
                    print(
                        "           NO GPS DATA — check TX/RX wiring "
                        "(try swapping blue/green at GPS side)"
                    )
                last_stats = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("\nStopped.")


if __name__ == "__main__":
    main()
