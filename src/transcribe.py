#!/usr/bin/env python3
"""
transcribe.py — CLI Whisper transcription wrapper for GP consult recordings.

Transcribes a local WAV file (or all WAVs in a directory) using OpenAI Whisper,
then saves the resulting text to the transcripts/ directory.

Optionally labels speakers (GP vs patient) using pyannote.audio diarisation.
Pass --diarise and --hf-token to enable this feature.

All audio and transcripts are processed and stored locally — no data leaves
this machine.

Usage:
    python3 src/transcribe.py --audio audio/20250225_143022.wav
    python3 src/transcribe.py --audio-dir audio/
    python3 src/transcribe.py --audio audio/20250225_143022.wav --model large
    python3 src/transcribe.py --audio audio/20250225_143022.wav --model medium --language en
    python3 src/transcribe.py --audio audio/20250225_143022.wav --diarise --hf-token hf_...
    python3 src/transcribe.py --audio audio/20250225_143022.wav --device cuda
"""

import argparse
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

TRANSCRIPT_DIR = Path(__file__).parent.parent / "transcripts"
_MIN_AUDIO_RMS = 1e-4


def _estimate_wav_rms(audio_path: Path) -> float | None:
    """Return RMS amplitude for WAV files, or None if unreadable/non-WAV."""
    try:
        with wave.open(str(audio_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return None

    if sampwidth == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        audio = np.frombuffer(frames, dtype=np.int8).astype(np.float32) / 128.0
    else:
        return None

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio * audio)))


def _diarise_with_pyannote(
    audio_path: Path,
    whisper_segments: list,
    hf_token: str,
) -> str:
    """
    Combine Whisper segments (each with 'start', 'end', 'text') with pyannote
    speaker diarisation to produce a labelled transcript.

    Returns a string with lines of the form:
        [SPEAKER_00]: some text here
        [SPEAKER_01]: some text here
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pyannote.audio is required for diarisation. Install it with:\n"
            "  pip3 install pyannote.audio --break-system-packages\n"
            "You must also accept the model licence at:\n"
            "  https://hf.co/pyannote/speaker-diarization-3.1"
        ) from exc

    print("  Loading pyannote diarisation pipeline…", file=sys.stderr)
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    print(f"  Running diarisation on {audio_path.name}…", file=sys.stderr)
    diarization = pipeline(str(audio_path))

    # pyannote 4.x wraps output in DiarizeOutput; unwrap to the Annotation
    annotation = (
        diarization.speaker_diarization
        if hasattr(diarization, "speaker_diarization")
        else diarization
    )

    # Build sorted list of (start, end, speaker_label)
    speaker_segments = sorted(
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    )

    def _speaker_at(t: float) -> str:
        for start, end, speaker in speaker_segments:
            if start <= t <= end:
                return speaker
        # Fall back to nearest segment midpoint
        if speaker_segments:
            nearest = min(speaker_segments, key=lambda s: abs((s[0] + s[1]) / 2 - t))
            return nearest[2]
        return "SPEAKER_?"

    # Assign each Whisper segment to a speaker and group consecutive same-speaker
    # segments together. Use a list of lists internally so entries can be mutated.
    turns: list[list] = []  # each entry is [speaker, text]
    for seg in whisper_segments:
        seg_mid = (seg["start"] + seg["end"]) / 2
        speaker = _speaker_at(seg_mid)
        text = seg["text"].strip()
        if not text:
            continue
        if turns and turns[-1][0] == speaker:
            turns[-1][1] = turns[-1][1] + " " + text
        else:
            turns.append([speaker, text])

    return "\n".join(f"[{speaker}]: {text}" for speaker, text in turns)


def _resolve_device(device_arg: str) -> str:
    """Resolve ``"auto"`` to ``"cuda"`` or ``"cpu"`` based on availability."""
    if device_arg != "auto":
        return device_arg
    try:
        import torch  # type: ignore
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _transcribe_with_faster_whisper(
    audio_path: Path,
    model: str,
    language: str,
    device: str,
) -> tuple[str, list]:
    """Transcribe using the faster-whisper library (CTranslate2 backend).

    Much faster than openai-whisper on the same hardware, especially on CPU
    (int8 quantisation) or CUDA (float16).

    Returns a tuple of (transcript_text, segments) where each segment dict has
    ``start``, ``end``, and ``text`` keys, matching the contract of the other
    backends.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is not installed. Install it with:\n"
            "  pip3 install faster-whisper>=1.0 --break-system-packages"
        ) from exc

    compute_type = "float16" if device == "cuda" else "int8"
    print(
        f"  Loading faster-whisper model '{model}' on {device} ({compute_type})…",
        file=sys.stderr,
    )
    wmodel = WhisperModel(model, device=device, compute_type=compute_type)

    print(f"  Transcribing {audio_path.name} with faster-whisper…", file=sys.stderr)
    segments_iter, _info = wmodel.transcribe(
        str(audio_path),
        language=language,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    segments: list[dict] = []
    text_parts: list[str] = []
    for seg in segments_iter:
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
        text_parts.append(seg.text)

    full_text = "".join(text_parts).strip()
    return full_text, segments


def _transcribe_with_library(audio_path: Path, model: str, language: str) -> tuple[str, list]:
    """Transcribe using the openai-whisper Python library.

    Returns a tuple of (transcript_text, segments) where segments is the list
    of per-segment dicts produced by Whisper (each has 'start', 'end', 'text').
    """
    import whisper  # type: ignore

    print(f"  Loading Whisper model '{model}'…", file=sys.stderr)
    wmodel = whisper.load_model(model)
    print(f"  Transcribing {audio_path.name}…", file=sys.stderr)
    result = wmodel.transcribe(str(audio_path), language=language)
    return result["text"].strip(), result.get("segments", [])


def _transcribe_with_cli(
    audio_path: Path, model: str, language: str, output_dir: Path
) -> tuple[str, list]:
    """Fallback: transcribe by shelling out to the whisper CLI.

    The CLI does not expose per-segment timing data, so the segments list is
    always empty. Diarisation requires the Python library.
    """
    print(f"  (Using whisper CLI fallback for {audio_path.name}…)", file=sys.stderr)
    cmd = [
        "whisper",
        str(audio_path),
        "--model", model,
        "--language", language,
        "--output_dir", str(output_dir),
        "--output_format", "txt",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError("whisper CLI executable not found") from exc
    if result.returncode != 0:
        raise RuntimeError(f"whisper CLI failed with exit code {result.returncode}")

    cli_out = output_dir / (audio_path.stem + ".txt")
    if cli_out.exists():
        text = cli_out.read_text(encoding="utf-8").strip()
        cli_out.unlink()
        return text, []
    raise RuntimeError("whisper CLI completed but did not produce a transcript file")


def transcribe(
    audio_path: Path,
    model: str = "medium",
    language: str = "en",
    output_dir: Path = TRANSCRIPT_DIR,
    diarise: bool = False,
    hf_token: str = "",
    device: str = "auto",
) -> Path:
    """
    Transcribe *audio_path* and save the result as a .txt file in *output_dir*.

    The transcription backend is chosen in this order:
    1. ``faster-whisper`` (fastest; requires ``pip install faster-whisper``).
    2. ``openai-whisper`` Python library.
    3. ``whisper`` CLI (slowest; no segment timing — diarisation unavailable).

    When *diarise* is ``True`` and a *hf_token* is supplied, speaker diarisation
    is attempted via pyannote.audio. Each line of the saved transcript will be
    prefixed with the speaker label, e.g. ``[SPEAKER_00]: …``. If pyannote.audio
    is not installed, a warning is printed and the plain transcript is saved
    instead.

    *device* controls where faster-whisper runs: ``"auto"`` (default) selects
    CUDA if available, otherwise CPU; pass ``"cuda"`` or ``"cpu"`` explicitly to
    override.

    Returns the path of the saved transcript.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path.name}")

    rms = _estimate_wav_rms(audio_path)
    if rms is not None and rms < _MIN_AUDIO_RMS:
        raise RuntimeError(
            f"Audio appears silent or near-silent (RMS={rms:.6f}). "
            "Check microphone/input device and re-record."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (audio_path.stem + ".txt")

    resolved_device = _resolve_device(device)

    # Attempt backends in order: faster-whisper → openai-whisper lib → whisper CLI.
    text: str
    segments: list

    try:
        text, segments = _transcribe_with_faster_whisper(
            audio_path, model, language, resolved_device
        )
    except ImportError:
        # faster-whisper not installed — fall through to openai-whisper.
        try:
            text, segments = _transcribe_with_library(audio_path, model, language)
        except ModuleNotFoundError:
            try:
                text, segments = _transcribe_with_cli(audio_path, model, language, output_dir)
            except FileNotFoundError:
                print(
                    "ERROR: Neither faster-whisper, the openai-whisper Python package,\n"
                    "       nor the whisper CLI could be found. Install one of:\n"
                    "         pip3 install faster-whisper>=1.0 --break-system-packages\n"
                    "         pip3 install openai-whisper torch --break-system-packages\n"
                    "         sudo apt-get install -y ffmpeg && pip3 install openai-whisper",
                    file=sys.stderr,
                )
                sys.exit(1)
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                sys.exit(1)
        except Exception as exc:
            print(f"ERROR: Whisper transcription failed: {exc}", file=sys.stderr)
            sys.exit(1)
    except Exception as exc:
        print(f"ERROR: faster-whisper transcription failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if diarise and hf_token and segments:
        try:
            text = _diarise_with_pyannote(audio_path, segments, hf_token)
        except ModuleNotFoundError as exc:
            print(
                f"WARNING: Diarisation requested but pyannote.audio is not available "
                f"— saving plain transcript instead.\n  ({exc})",
                file=sys.stderr,
            )

    out_path.write_text(text, encoding="utf-8")
    print(f"Saved transcript: {out_path.name}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe a GP consult WAV file to text using Whisper (local)."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--audio",
        metavar="FILE",
        type=Path,
        help="Path to a single WAV file to transcribe.",
    )
    source.add_argument(
        "--audio-dir",
        metavar="DIR",
        type=Path,
        help="Transcribe all WAV files in this directory.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        metavar="MODEL",
        help="Whisper model size: tiny, base, small, medium, large (default: medium).",
    )
    parser.add_argument(
        "--language",
        default="en",
        metavar="LANG",
        help="Language code passed to Whisper (default: en).",
    )
    parser.add_argument(
        "--output-dir",
        default=TRANSCRIPT_DIR,
        metavar="DIR",
        type=Path,
        help=f"Directory for saved transcripts (default: {TRANSCRIPT_DIR}).",
    )
    parser.add_argument(
        "--diarise",
        action="store_true",
        help="Label speakers in the transcript using pyannote.audio (requires --hf-token and pyannote.audio install).",
    )
    parser.add_argument(
        "--hf-token",
        default="",
        metavar="TOKEN",
        help="HuggingFace access token required for the pyannote diarisation model.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        metavar="DEVICE",
        help="Device for faster-whisper: 'auto' (default), 'cuda', or 'cpu'.",
    )
    args = parser.parse_args()

    if args.audio:
        if not args.audio.is_file():
            print(f"ERROR: Audio file not found: {args.audio.name}", file=sys.stderr)
            sys.exit(1)
        transcribe(
            args.audio,
            model=args.model,
            language=args.language,
            output_dir=args.output_dir,
            diarise=args.diarise,
            hf_token=args.hf_token,
            device=args.device,
        )
    else:
        if not args.audio_dir.is_dir():
            print(f"ERROR: Audio directory not found: {args.audio_dir.name}", file=sys.stderr)
            sys.exit(1)
        wav_files = sorted(args.audio_dir.glob("*.wav"))
        if not wav_files:
            print(f"No WAV files found in {args.audio_dir.name}", file=sys.stderr)
            sys.exit(1)
        for wav in wav_files:
            transcribe(
                wav,
                model=args.model,
                language=args.language,
                output_dir=args.output_dir,
                diarise=args.diarise,
                hf_token=args.hf_token,
                device=args.device,
            )


if __name__ == "__main__":
    main()
