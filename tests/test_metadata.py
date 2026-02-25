from datetime import datetime
from pathlib import Path
from uuid import UUID

from unittest import mock

from src import metadata


def _make_session(i: int) -> metadata.ConsultMetadata:
    meta = metadata.new_session(
        audio_path=f"/tmp/audio_{i}.wav",
        whisper_model="base",
        llm_model="llama3",
    )
    meta.transcript_path = f"/tmp/audio_{i}.txt"
    meta.note_path = f"/tmp/audio_{i}_soap.md"
    meta.mbs_item = "36"
    return meta


def test_new_session():
    meta = metadata.new_session("/tmp/consult.wav", "small", "llama3")

    assert isinstance(meta, metadata.ConsultMetadata)
    assert meta.audio_path == "/tmp/consult.wav"
    assert meta.whisper_model == "small"
    assert meta.llm_model == "llama3"
    assert meta.transcript_path is None
    assert meta.note_path is None
    assert meta.approved is False
    assert meta.approved_at is None
    UUID(meta.session_id)
    datetime.fromisoformat(meta.recorded_at)


def test_save_and_load_roundtrip(tmp_path):
    meta = _make_session(1)
    meta.approved = True
    meta.approved_at = metadata._now_iso()

    json_path = metadata.save(meta, tmp_path)
    loaded = metadata.load(json_path)

    assert loaded == meta


def test_mark_approved():
    meta = metadata.new_session("/tmp/consult.wav", "small", "llama3")

    updated = metadata.mark_approved(meta)

    assert updated is meta
    assert updated.approved is True
    assert isinstance(updated.approved_at, str)
    datetime.fromisoformat(updated.approved_at)


def test_find_sessions_empty_dir(tmp_path):
    assert metadata.find_sessions(tmp_path) == []


def test_find_sessions_multiple(tmp_path):
    created = []
    for i in range(3):
        meta = _make_session(i)
        created.append(meta)
        metadata.save(meta, tmp_path)

    found = metadata.find_sessions(tmp_path)

    assert len(found) == 3
    assert {m.session_id for m in found} == {m.session_id for m in created}


def test_cli_list(tmp_path):
    meta = _make_session(1)
    metadata.save(meta, tmp_path)

    with mock.patch("sys.argv", ["metadata.py", "list", "--notes-dir", str(tmp_path)]):
        metadata.main()
