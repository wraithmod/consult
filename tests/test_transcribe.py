import builtins
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

from src import transcribe


def transcript_fixture_text() -> str:
    return Path("tests/synthetic/transcript_01.txt").read_text(encoding="utf-8").strip()


def fail_whisper_import(monkeypatch):
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "whisper":
            raise ModuleNotFoundError("No module named 'whisper'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def test_transcribe_python_library(monkeypatch, tmp_path):
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = transcript_fixture_text()

    fake_model = mock.Mock()
    fake_model.transcribe.return_value = {"text": f"  {expected}\n"}
    fake_whisper = types.SimpleNamespace(load_model=mock.Mock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    out_path = transcribe.transcribe(audio_path, model="small", language="en", output_dir=out_dir)

    assert out_path == out_dir / "consult.txt"
    assert out_path.read_text(encoding="utf-8") == expected
    fake_whisper.load_model.assert_called_once_with("small")
    fake_model.transcribe.assert_called_once_with(str(audio_path), language="en")


def test_transcribe_cli_fallback(monkeypatch, tmp_path):
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = transcript_fixture_text()
    fail_whisper_import(monkeypatch)

    def fake_run(cmd, capture_output, text):
        cli_txt = out_dir / "consult.txt"
        cli_txt.parent.mkdir(parents=True, exist_ok=True)
        cli_txt.write_text(expected, encoding="utf-8")
        return mock.Mock(returncode=0)

    run_mock = mock.Mock(side_effect=fake_run)
    monkeypatch.setattr(transcribe.subprocess, "run", run_mock)

    out_path = transcribe.transcribe(audio_path, output_dir=out_dir)

    assert out_path == out_dir / "consult.txt"
    assert out_path.read_text(encoding="utf-8") == expected
    run_mock.assert_called_once()


def test_transcribe_both_unavailable_exits(monkeypatch, tmp_path):
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    fail_whisper_import(monkeypatch)
    monkeypatch.setattr(transcribe.subprocess, "run", mock.Mock(side_effect=FileNotFoundError()))

    with pytest.raises(SystemExit) as excinfo:
        transcribe.transcribe(audio_path, output_dir=tmp_path / "out")

    assert excinfo.value.code == 1


def test_transcribe_audio_dir(monkeypatch, tmp_path):
    audio_dir = tmp_path / "audio"
    out_dir = tmp_path / "out"
    audio_dir.mkdir()
    (audio_dir / "a.wav").write_bytes(b"RIFFa")
    (audio_dir / "b.wav").write_bytes(b"RIFFb")
    expected = transcript_fixture_text()

    fake_model = mock.Mock()
    fake_model.transcribe.return_value = {"text": expected}
    fake_whisper = types.SimpleNamespace(load_model=mock.Mock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)
    monkeypatch.setattr(sys, "argv", ["transcribe.py", "--audio-dir", str(audio_dir), "--output-dir", str(out_dir)])

    transcribe.main()

    produced = sorted(p.name for p in out_dir.glob("*.txt"))
    assert produced == ["a.txt", "b.txt"]
    assert (out_dir / "a.txt").read_text(encoding="utf-8") == expected
    assert (out_dir / "b.txt").read_text(encoding="utf-8") == expected


def test_transcribe_missing_audio_exits(tmp_path):
    missing = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        transcribe.transcribe(missing, output_dir=tmp_path / "out")
