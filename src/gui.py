from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import wave
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Header, Footer, Button, Static, Label, Markdown, LoadingIndicator
from textual.screen import Screen
from textual.worker import Worker, WorkerState

# Add current directory to path so we can import local modules if needed
sys.path.append(str(Path(__file__).parent))

import argparse

# Constants from record.py
SAMPLE_RATE = 16_000
CHANNELS = 1
DTYPE = "float32"
AUDIO_DIR = Path(__file__).parent.parent / "audio"
MIN_AUDIO_RMS = 1e-4

def parse_args():
    parser = argparse.ArgumentParser(description="GP Consultation GUI")
    parser.add_argument(
        "--server",
        default="http://10.10.200.113:8000/process",
        help="Server URL for processing audio (default: http://10.10.200.113:8000/process)"
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Sounddevice input device index (optional).",
    )
    # We use parse_known_args to avoid conflicts with Textual's own arguments
    return parser.parse_known_args()[0]

ARGS = parse_args()
SERVER_URL = ARGS.server
INPUT_DEVICE = ARGS.input_device

class RecordScreen(Screen):
    """The screen where audio recording happens."""

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Label("GP Consultation Recorder", id="title"),
                Label("Press Start to begin recording", id="status"),
                Static("00:00", id="timer"),
                Horizontal(
                    Button("Start Recording", variant="success", id="start"),
                    Button("Stop & Process", variant="error", id="stop", disabled=True),
                    classes="buttons"
                ),
                id="record-box"
            )
        )
        yield Footer()

    def on_mount(self) -> None:
        self.recording = False
        self.start_time = None
        self.frames = []
        self.stream = None
        self.update_timer = None

    def _callback(self, indata, frame_count, time_info, status):
        if status:
            print(f"[audio status] {status}", file=sys.stderr)
        if self.recording:
            self.frames.append(indata.copy())

    def update_clock(self) -> None:
        if self.recording and self.start_time:
            elapsed = int(datetime.now().timestamp() - self.start_time)
            mins, secs = divmod(elapsed, 60)
            self.query_one("#timer", Static).update(f"{mins:02d}:{secs:02d}")

    @on(Button.Pressed, "#start")
    def start_recording(self) -> None:
        self.recording = True
        self.start_time = datetime.now().timestamp()
        self.frames = []
        self.query_one("#status", Label).update("Recording in progress...")
        self.query_one("#start", Button).disabled = True
        self.query_one("#stop", Button).disabled = False
        
        stream_kwargs = dict(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._callback,
        )
        if INPUT_DEVICE is not None:
            stream_kwargs["device"] = INPUT_DEVICE

        self.stream = sd.InputStream(**stream_kwargs)
        self.stream.start()
        self.update_timer = self.set_interval(1, self.update_clock)

    @on(Button.Pressed, "#stop")
    def stop_recording(self) -> None:
        if not self.recording:
            return

        self.recording = False
        if self.update_timer:
            self.update_timer.stop()
        
        if self.stream:
            self.stream.stop()
            self.stream.close()

        self.query_one("#status", Label).update("Processing audio...")
        self.query_one("#stop", Button).disabled = True
        self.process_audio()

    @work(exclusive=True)
    async def process_audio(self) -> None:
        if not self.frames:
            self.query_one("#status", Label).update("Error: No audio captured.")
            self.query_one("#start", Button).disabled = False
            return

        # Save audio locally
        audio = np.concatenate(self.frames, axis=0)
        rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < MIN_AUDIO_RMS:
            self.query_one("#status", Label).update(
                "No microphone signal detected. Check input device and try again."
            )
            self.query_one("#start", Button).disabled = False
            return

        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = AUDIO_DIR / f"{timestamp}.wav"

        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        print(f"[audio] saved {out_path.name} rms={rms:.6f} peak={peak:.6f}", file=sys.stderr)

        # Send to server
        try:
            # Use run_in_executor for synchronous requests call
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self._send_to_server, out_path)
            
            if response and "note" in response:
                self.app.push_screen(BriefScreen(response["note"]))
            else:
                self.query_one("#status", Label).update("Error: Server failed to process.")
                self.query_one("#start", Button).disabled = False
        except Exception as e:
            self.query_one("#status", Label).update(f"Error: {str(e)}")
            self.query_one("#start", Button).disabled = False

    def _send_to_server(self, path: Path):
        with path.open("rb") as f:
            response = requests.post(SERVER_URL, files={"file": (path.name, f, "audio/wav")})
            response.raise_for_status()
            return response.json()

class BriefScreen(Screen):
    """The screen where the generated note is displayed."""

    def __init__(self, note_content: str):
        super().__init__()
        self.note_content = note_content

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Vertical(
                Label("Consultation Brief", id="brief-title"),
                Markdown(self.note_content, id="note-display"),
                Horizontal(
                    Button("Done", variant="primary", id="done"),
                    classes="buttons"
                ),
                id="brief-box"
            )
        )
        yield Footer()

    @on(Button.Pressed, "#done")
    def exit_brief(self) -> None:
        self.app.pop_screen()
        # Reset the record screen status
        record_screen = self.app.query_one(RecordScreen)
        record_screen.query_one("#status", Label).update("Press Start to begin recording")
        record_screen.query_one("#timer", Static).update("00:00")
        record_screen.query_one("#start", Button).disabled = False

class ConsultApp(App):
    """The main Textual application."""

    CSS = """
    Screen {
        align: center middle;
    }

    #record-box, #brief-box {
        width: 80%;
        height: 80%;
        border: thick $primary;
        background: $surface;
        padding: 1 2;
        align: center middle;
    }

    #title, #brief-title {
        text-align: center;
        width: 100%;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #status {
        text-align: center;
        width: 100%;
        margin-bottom: 1;
    }

    #timer {
        text-align: center;
        width: 100%;
        text-style: bold;
        color: $success;
        margin: 1 0;
    }

    .buttons {
        align: center middle;
        height: auto;
        margin-top: 1;
    }

    Button {
        margin: 0 1;
    }

    #note-display {
        height: 1fr;
        border: solid $primary-muted;
        padding: 1;
        margin: 1 0;
        background: $boost;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("d", "toggle_dark", "Toggle dark mode"),
    ]

    def on_mount(self) -> None:
        self.push_screen(RecordScreen())

if __name__ == "__main__":
    app = ConsultApp()
    app.run()
