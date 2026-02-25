#!/usr/bin/env python3
"""
metadata.py — JSON sidecar metadata tracker for GP consult sessions.

Stores consult session metadata locally as JSON sidecars alongside notes,
supporting workflows aligned with the Australian Privacy Act 1988.

Usage:
    python3 src/metadata.py list
    python3 src/metadata.py list --notes-dir notes/
"""

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4


DEFAULT_NOTES_DIR = Path(__file__).resolve().parent.parent / "notes"


@dataclass
class ConsultMetadata:
    session_id: str
    recorded_at: str
    audio_path: str
    transcript_path: str | None
    note_path: str | None
    whisper_model: str
    llm_model: str
    approved: bool
    approved_at: str | None
    mbs_item: str | None


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sidecar_path(meta: ConsultMetadata, output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    stem_source = meta.note_path if meta.note_path else meta.audio_path
    stem = Path(stem_source).stem
    return output_dir / f"{stem}.json"


def new_session(audio_path, whisper_model, llm_model) -> ConsultMetadata:
    return ConsultMetadata(
        session_id=str(uuid4()),
        recorded_at=_now_iso(),
        audio_path=str(audio_path),
        transcript_path=None,
        note_path=None,
        whisper_model=str(whisper_model),
        llm_model=str(llm_model),
        approved=False,
        approved_at=None,
        mbs_item=None,
    )


def save(meta, output_dir) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = _sidecar_path(meta, output_dir)
    json_path.write_text(json.dumps(asdict(meta), indent=2) + "\n", encoding="utf-8")
    return json_path


def load(json_path) -> ConsultMetadata:
    path = Path(json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return ConsultMetadata(**data)


def mark_approved(meta) -> ConsultMetadata:
    meta.approved = True
    meta.approved_at = _now_iso()
    return meta


def find_sessions(notes_dir) -> list[ConsultMetadata]:
    notes_path = Path(notes_dir)
    if not notes_path.exists():
        return []
    sessions: list[ConsultMetadata] = []
    for json_path in sorted(notes_path.glob("*.json")):
        sessions.append(load(json_path))
    return sessions


def _print_sessions(sessions: list[ConsultMetadata]) -> None:
    if not sessions:
        print("No consult metadata sidecars found.")
        return

    for meta in sessions:
        status = "approved" if meta.approved else "pending approval"
        note_name = Path(meta.note_path).name if meta.note_path else "-"
        transcript_name = Path(meta.transcript_path).name if meta.transcript_path else "-"
        mbs_item = meta.mbs_item if meta.mbs_item else "-"
        print(f"Session: {meta.session_id}")
        print(f"  Recorded: {meta.recorded_at}")
        print(f"  Audio: {Path(meta.audio_path).name}")
        print(f"  Transcript: {transcript_name}")
        print(f"  Note: {note_name}")
        print(f"  Whisper model: {meta.whisper_model}")
        print(f"  LLM model: {meta.llm_model}")
        print(f"  Approval: {status}")
        if meta.approved_at:
            print(f"  Approved at: {meta.approved_at}")
        print(f"  MBS item: {mbs_item}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage local JSON sidecar metadata for GP consult sessions."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list",
        help="Pretty-print all consult metadata sidecars.",
    )
    list_parser.add_argument(
        "--notes-dir",
        default=DEFAULT_NOTES_DIR,
        type=Path,
        help=f"Directory containing note sidecars (default: {DEFAULT_NOTES_DIR}).",
    )

    args = parser.parse_args()

    if args.command == "list":
        _print_sessions(find_sessions(args.notes_dir))


if __name__ == "__main__":
    main()
