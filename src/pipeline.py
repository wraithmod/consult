#!/usr/bin/env python3
"""pipeline.py — end-to-end GP consult pipeline runner.

By default, runs each step in-process by importing the source modules directly,
which avoids subprocess overhead and gives tighter error propagation.

Pass ``--subprocess`` to restore the original behaviour of launching each step
as a separate Python subprocess (useful for debugging or isolation).

Steps (sequential):
    record → transcribe → summarise → review
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Legacy subprocess helpers (used when --subprocess flag is set)
# ---------------------------------------------------------------------------

PATH_PATTERN = re.compile(r"([^\s\"']+\.(?:wav|txt|md))")


def _run_subprocess(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess, mirror stdout/stderr to the terminal, and capture stdout."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=None,
        stdin=None,
        text=True,
        bufsize=1,
        env=env,
    )

    captured_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        captured_lines.append(line)

    process.stdout.close()
    returncode = process.wait()
    return returncode, "".join(captured_lines)


def _extract_output_path(stdout_text: str, expected_suffix: str) -> Path | None:
    matches = PATH_PATTERN.findall(stdout_text)
    for candidate in reversed(matches):
        if candidate.lower().endswith(expected_suffix.lower()):
            return Path(candidate)
    return None


def _extract_review_status(stdout_text: str) -> str | None:
    for line in reversed(stdout_text.splitlines()):
        lower = line.strip().lower()
        if not lower:
            continue
        if "approved" in lower:
            return "Approved"
        if "rejected" in lower:
            return "Rejected"
        if "quit" in lower:
            return "Quit"
    return None


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path)


def _fail(step_name: str, returncode: int) -> None:
    print(
        f"ERROR: {step_name} failed with exit code {returncode}.",
        file=sys.stderr,
    )
    sys.exit(returncode)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GP consult pipeline (record, transcribe, summarise, review)."
    )
    parser.add_argument(
        "--device",
        default=None,
        metavar="N",
        help="audio input device index (passed to record) or device string for faster-whisper",
    )
    parser.add_argument(
        "--model-whisper",
        default="medium",
        metavar="MODEL",
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--model-llm",
        default="llama3",
        metavar="MODEL",
        help="Ollama model name (default: llama3)",
    )
    parser.add_argument(
        "--skip-record",
        type=Path,
        default=None,
        metavar="FILE",
        help="skip recording, use existing WAV file instead",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        dest="use_subprocess",
        help="use subprocess execution (legacy behaviour) instead of in-process calls",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# In-process pipeline steps
# ---------------------------------------------------------------------------

def _step_record_inprocess(device_arg: str | None) -> Path:
    """Record a consultation in-process and return the saved WAV path."""
    from src import record as record_mod  # type: ignore

    # record.record() accepts an optional integer device index.
    device_int: int | None = None
    if device_arg is not None:
        try:
            device_int = int(device_arg)
        except ValueError:
            pass  # non-integer device string — leave as None for default

    return record_mod.record(device=device_int)


def _step_transcribe_inprocess(
    audio_path: Path,
    model: str,
    device_arg: str | None,
) -> Path:
    """Transcribe *audio_path* in-process and return the saved transcript path."""
    from src import transcribe as transcribe_mod  # type: ignore

    # For transcription, --device is a string ("auto", "cuda", "cpu").
    # If the caller passed an integer (audio device index), fall back to "auto".
    device_str = "auto"
    if device_arg is not None and device_arg not in ("auto", "cuda", "cpu"):
        # Likely an audio device index — not relevant to Whisper device selection.
        device_str = "auto"
    elif device_arg is not None:
        device_str = device_arg

    return transcribe_mod.transcribe(
        audio_path=audio_path,
        model=model,
        language="en",
        device=device_str,
    )


def _step_summarise_inprocess(transcript_path: Path, model: str) -> Path:
    """Summarise *transcript_path* in-process and return the saved note path."""
    from src import summarise as summarise_mod  # type: ignore

    transcript_text = summarise_mod.read_transcript(str(transcript_path))
    prompt = summarise_mod.build_prompt(transcript_text)
    llm_text = summarise_mod.call_ollama(model, prompt)
    note_markdown = summarise_mod.format_soap_markdown(llm_text)

    # Resolve output directory the same way summarise.py does.
    src_dir = Path(__file__).resolve().parent
    output_dir = (src_dir / "../notes/").resolve()
    return summarise_mod.save_note(str(transcript_path), output_dir, note_markdown)


def _step_review_inprocess(note_path: Path, transcript_path: Path) -> tuple[str, int]:
    """Open the review UI in-process and return (status, exit_code)."""
    from src import review as review_mod  # type: ignore

    status, exit_code = review_mod.run_review_ui(note_path, transcript_path)
    if status == "Approved":
        review_mod.write_approved_marker(note_path)
    elif status == "Rejected":
        note_path.unlink(missing_ok=True)
    return status, exit_code


# ---------------------------------------------------------------------------
# Subprocess pipeline steps (legacy)
# ---------------------------------------------------------------------------

def _step_record_subprocess(src_dir: Path, device_arg: str | None) -> Path:
    record_cmd = ["python3", str(src_dir / "record.py")]
    if device_arg is not None:
        try:
            int(device_arg)
            record_cmd.extend(["--device", str(device_arg)])
        except ValueError:
            pass
    code, record_stdout = _run_subprocess(record_cmd)
    if code != 0:
        _fail("Recording", code)
    audio_path = _extract_output_path(record_stdout, ".wav")
    if audio_path is None:
        print(
            "ERROR: Recording completed but no WAV path was found in stdout.",
            file=sys.stderr,
        )
        sys.exit(1)
    return audio_path


def _step_transcribe_subprocess(
    src_dir: Path,
    audio_path: Path,
    model: str,
) -> Path:
    transcribe_cmd = [
        "python3",
        str(src_dir / "transcribe.py"),
        "--audio",
        str(audio_path),
        "--model",
        model,
    ]
    code, transcribe_stdout = _run_subprocess(transcribe_cmd)
    if code != 0:
        _fail("Transcribing", code)
    transcript_path = _extract_output_path(transcribe_stdout, ".txt")
    if transcript_path is None:
        print(
            "ERROR: Transcribing completed but no transcript path was found in stdout.",
            file=sys.stderr,
        )
        sys.exit(1)
    return transcript_path


def _step_summarise_subprocess(
    src_dir: Path,
    transcript_path: Path,
    model: str,
) -> Path:
    summarise_cmd = [
        "python3",
        str(src_dir / "summarise.py"),
        "--transcript",
        str(transcript_path),
        "--model",
        model,
        "--no-review",
    ]
    code, summarise_stdout = _run_subprocess(summarise_cmd)
    if code != 0:
        _fail("Summarising", code)
    note_path = _extract_output_path(summarise_stdout, ".md")
    if note_path is None:
        print(
            "ERROR: Summarising completed but no note path was found in stdout.",
            file=sys.stderr,
        )
        sys.exit(1)
    return note_path


def _step_review_subprocess(
    src_dir: Path,
    note_path: Path,
    transcript_path: Path,
) -> str:
    review_cmd = [
        "python3",
        str(src_dir / "review.py"),
        "--note",
        str(note_path),
        "--transcript",
        str(transcript_path),
    ]
    code, review_stdout = _run_subprocess(review_cmd)
    if code not in (0, 1, 2):
        _fail("Review", code)
    return _extract_review_status(review_stdout) or "Unknown"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"

    device_arg: str | None = str(args.device) if args.device is not None else None

    if args.skip_record is not None and not args.skip_record.exists():
        print(f"ERROR: Audio file not found: {args.skip_record}", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 1 — Record
    # ------------------------------------------------------------------
    if args.skip_record is None:
        print("[1/4] Recording...")
        if args.use_subprocess:
            audio_path = _step_record_subprocess(src_dir, device_arg)
        else:
            audio_path = _step_record_inprocess(device_arg)
    else:
        print("[1/4] Recording... (skipped)")
        audio_path = args.skip_record

    # ------------------------------------------------------------------
    # Step 2 — Transcribe
    # ------------------------------------------------------------------
    print("[2/4] Transcribing...")
    if args.use_subprocess:
        transcript_path = _step_transcribe_subprocess(src_dir, audio_path, args.model_whisper)
    else:
        transcript_path = _step_transcribe_inprocess(audio_path, args.model_whisper, device_arg)

    # ------------------------------------------------------------------
    # Step 3 — Summarise
    # ------------------------------------------------------------------
    print("[3/4] Summarising...")
    if args.use_subprocess:
        note_path = _step_summarise_subprocess(src_dir, transcript_path, args.model_llm)
    else:
        note_path = _step_summarise_inprocess(transcript_path, args.model_llm)

    # ------------------------------------------------------------------
    # Step 4 — Review
    # ------------------------------------------------------------------
    print("[4/4] Reviewing...")
    if args.use_subprocess:
        status = _step_review_subprocess(src_dir, note_path, transcript_path)
    else:
        status, _code = _step_review_inprocess(note_path, transcript_path)

    print()
    print(f"Audio:      {_display_path(audio_path, repo_root)}")
    print(f"Transcript: {_display_path(transcript_path, repo_root)}")
    print(f"Note:       {_display_path(note_path, repo_root)}")
    print(f"Status:     {status}")


if __name__ == "__main__":
    main()
