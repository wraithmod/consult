#!/usr/bin/env python3
"""pipeline.py — end-to-end GP consult pipeline runner.

Runs the local consult workflow as subprocesses:
record -> transcribe -> summarise -> review
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PATH_PATTERN = re.compile(r"([^\s\"']+\.(?:wav|txt|md))")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the GP consult pipeline (record, transcribe, summarise, review)."
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        metavar="N",
        help="audio input device (passed to record.py logic)",
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
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"

    if args.skip_record is not None and not args.skip_record.exists():
        print(f"ERROR: Audio file not found: {args.skip_record}", file=sys.stderr)
        sys.exit(1)

    if args.skip_record is None:
        print("[1/4] Recording...")
        record_cmd = ["python3", str(src_dir / "record.py")]
        if args.device is not None:
            record_cmd.extend(["--device", str(args.device)])
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
    else:
        print("[1/4] Recording... (skipped)")
        audio_path = args.skip_record

    print("[2/4] Transcribing...")
    transcribe_cmd = [
        "python3",
        str(src_dir / "transcribe.py"),
        "--audio",
        str(audio_path),
        "--model",
        args.model_whisper,
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

    print("[3/4] Summarising...")
    summarise_cmd = [
        "python3",
        str(src_dir / "summarise.py"),
        "--transcript",
        str(transcript_path),
        "--model",
        args.model_llm,
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

    print("[4/4] Reviewing...")
    review_cmd = [
        "python3",
        str(src_dir / "review.py"),
        "--note",
        str(note_path),
        "--transcript",
        str(transcript_path),
    ]
    code, review_stdout = _run_subprocess(review_cmd)
    if code != 0:
        _fail("Review", code)
    status = _extract_review_status(review_stdout) or "Unknown"

    print()
    print(f"Audio:      {_display_path(audio_path, repo_root)}")
    print(f"Transcript: {_display_path(transcript_path, repo_root)}")
    print(f"Note:       {_display_path(note_path, repo_root)}")
    print(f"Status:     {status}")


if __name__ == "__main__":
    main()
