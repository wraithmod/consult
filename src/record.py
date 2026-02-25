#!/usr/bin/env python3
"""
record.py — CLI audio recorder for GP consult transcription.

Records at 16 000 Hz mono (Whisper-optimal) and saves a timestamped
WAV file to the audio/ directory.

Usage:
    python3 src/record.py                  # record using default input device
    python3 src/record.py --device 2       # use device index 2
    python3 src/record.py --list-devices   # print available input devices and exit
"""

import argparse
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16_000   # Hz — Whisper's preferred rate
CHANNELS = 1           # mono
DTYPE = "float32"      # sounddevice native type; converted to int16 on save
AUDIO_DIR = Path(__file__).parent.parent / "audio"


def list_devices() -> None:
    devices = sd.query_devices()
    print("Available input devices:")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            marker = " (default)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {d['name']}{marker}")


def _timer_thread(stop_event: threading.Event) -> None:
    """Print a live elapsed-time counter to stderr until stop_event is set."""
    start = time.monotonic()
    while not stop_event.is_set():
        elapsed = int(time.monotonic() - start)
        mins, secs = divmod(elapsed, 60)
        sys.stderr.write(f"\r  Recording: {mins}:{secs:02d}   ")
        sys.stderr.flush()
        time.sleep(0.5)
    sys.stderr.write("\r" + " " * 30 + "\r")
    sys.stderr.flush()


def record(device: int | None = None) -> Path:
    """Record until the user presses Enter; return the saved WAV path."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    stop_event = threading.Event()

    def _callback(indata, frame_count, time_info, status):
        if status:
            print(f"  [audio status] {status}", file=sys.stderr)
        frames.append(indata.copy())

    stream_kwargs = dict(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=_callback,
    )
    if device is not None:
        stream_kwargs["device"] = device

    print("Press Enter to start recording...")
    input()

    timer = threading.Thread(target=_timer_thread, args=(stop_event,), daemon=True)

    with sd.InputStream(**stream_kwargs):
        timer.start()
        print("  [recording — press Enter to stop]", file=sys.stderr)
        input()
        stop_event.set()

    timer.join()

    if not frames:
        print("No audio captured.", file=sys.stderr)
        sys.exit(1)

    # Concatenate float32 frames and convert to int16 for WAV
    audio = np.concatenate(frames, axis=0)
    audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = AUDIO_DIR / f"{timestamp}.wav"

    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)          # 2 bytes = int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    duration = len(audio_int16) / SAMPLE_RATE
    mins, secs = divmod(int(duration), 60)
    print(f"Saved ({mins}:{secs:02d}): {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record a GP consult to a local WAV file."
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print available input devices and exit.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="Input device index (see --list-devices).",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    record(device=args.device)


if __name__ == "__main__":
    main()
