#!/usr/bin/env python3
"""
transcribe.py — CLI Whisper transcription wrapper for GP consult recordings.

Transcribes a local WAV file (or all WAVs in a directory) using OpenAI Whisper,
then saves the resulting text to the transcripts/ directory.

All audio and transcripts are processed and stored locally — no data leaves
this machine.

Usage:
    python3 src/transcribe.py --audio audio/20250225_143022.wav
    python3 src/transcribe.py --audio-dir audio/
    python3 src/transcribe.py --audio audio/20250225_143022.wav --model large
    python3 src/transcribe.py --audio audio/20250225_143022.wav --model medium --language en
"""

import argparse
import subprocess
import sys
from pathlib import Path

TRANSCRIPT_DIR = Path(__file__).parent.parent / "transcripts"


def _transcribe_with_library(audio_path: Path, model: str, language: str) -> str:
    """Transcribe using the openai-whisper Python library."""
    import whisper  # type: ignore

    print(f"  Loading Whisper model '{model}'…", file=sys.stderr)
    wmodel = whisper.load_model(model)
    print(f"  Transcribing {audio_path.name}…", file=sys.stderr)
    result = wmodel.transcribe(str(audio_path), language=language)
    return result["text"].strip()


def _transcribe_with_cli(
    audio_path: Path, model: str, language: str, output_dir: Path
) -> str:
    """Fallback: transcribe by shelling out to the whisper CLI."""
    print(f"  (Using whisper CLI fallback for {audio_path.name}…)", file=sys.stderr)
    cmd = [
        "whisper",
        str(audio_path),
        "--model", model,
        "--language", language,
        "--output_dir", str(output_dir),
        "--output_format", "txt",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"whisper CLI failed with exit code {result.returncode}")

    cli_out = output_dir / (audio_path.stem + ".txt")
    if cli_out.exists():
        text = cli_out.read_text(encoding="utf-8").strip()
        cli_out.unlink()
        return text
    raise FileNotFoundError(f"Expected CLI output not found: {cli_out}")


def transcribe(
    audio_path: Path,
    model: str = "medium",
    language: str = "en",
    output_dir: Path = TRANSCRIPT_DIR,
) -> Path:
    """
    Transcribe *audio_path* and save the result as a .txt file in *output_dir*.

    Returns the path of the saved transcript.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (audio_path.stem + ".txt")

    try:
        text = _transcribe_with_library(audio_path, model, language)
    except ModuleNotFoundError:
        try:
            text = _transcribe_with_cli(audio_path, model, language, output_dir)
        except (FileNotFoundError, RuntimeError):
            print(
                "ERROR: Neither the openai-whisper Python package nor the whisper CLI\n"
                "       could be found. Install one of:\n"
                "         pip3 install openai-whisper torch --break-system-packages\n"
                "         sudo apt-get install -y ffmpeg && pip3 install openai-whisper",
                file=sys.stderr,
            )
            sys.exit(1)

    out_path.write_text(text, encoding="utf-8")
    print(f"Saved transcript: {out_path}")
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
    args = parser.parse_args()

    if args.audio:
        if not args.audio.exists():
            print(f"ERROR: Audio file not found: {args.audio}", file=sys.stderr)
            sys.exit(1)
        transcribe(args.audio, model=args.model, language=args.language, output_dir=args.output_dir)
    else:
        wav_files = sorted(args.audio_dir.glob("*.wav"))
        if not wav_files:
            print(f"No WAV files found in {args.audio_dir}", file=sys.stderr)
            sys.exit(1)
        for wav in wav_files:
            transcribe(wav, model=args.model, language=args.language, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
