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
"""

import argparse
import subprocess
import sys
from pathlib import Path

TRANSCRIPT_DIR = Path(__file__).parent.parent / "transcripts"


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
        use_auth_token=hf_token,
    )
    print(f"  Running diarisation on {audio_path.name}…", file=sys.stderr)
    diarization = pipeline(str(audio_path))

    # Build sorted list of (start, end, speaker_label)
    speaker_segments = sorted(
        (turn.start, turn.end, speaker)
        for turn, _, speaker in diarization.itertracks(yield_label=True)
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
) -> Path:
    """
    Transcribe *audio_path* and save the result as a .txt file in *output_dir*.

    When *diarise* is ``True`` and a *hf_token* is supplied, speaker diarisation
    is attempted via pyannote.audio. Each line of the saved transcript will be
    prefixed with the speaker label, e.g. ``[SPEAKER_00]: …``. If pyannote.audio
    is not installed, a warning is printed and the plain transcript is saved
    instead.

    Returns the path of the saved transcript.
    """
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / (audio_path.stem + ".txt")

    try:
        text, segments = _transcribe_with_library(audio_path, model, language)
    except ModuleNotFoundError:
        try:
            text, segments = _transcribe_with_cli(audio_path, model, language, output_dir)
        except FileNotFoundError:
            print(
                "ERROR: Neither the openai-whisper Python package nor the whisper CLI\n"
                "       could be found. Install one of:\n"
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
            )


if __name__ == "__main__":
    main()
