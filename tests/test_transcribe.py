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


def fail_faster_whisper_import(monkeypatch):
    """Make 'import faster_whisper' raise ImportError."""
    original_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("No module named 'faster_whisper'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)


def _make_faster_whisper_mock(text: str, segments: list | None = None) -> types.SimpleNamespace:
    """Return a fake faster_whisper module with a mock WhisperModel."""
    if segments is None:
        segments = []

    # faster-whisper returns named tuple-like objects; use SimpleNamespace to mimic them.
    fake_segs = [
        types.SimpleNamespace(start=s.get("start", 0.0), end=s.get("end", 1.0), text=s["text"])
        for s in segments
    ]

    fake_model = mock.Mock()
    # transcribe returns (segments_iterable, info).
    # Use side_effect to produce a fresh iterator on every call so that
    # multi-file tests (audio-dir) don't exhaust a single shared iterator.
    fake_model.transcribe.side_effect = lambda *a, **kw: (iter(fake_segs), mock.Mock())

    fake_cls = mock.Mock(return_value=fake_model)
    return types.SimpleNamespace(WhisperModel=fake_cls), fake_model


def test_transcribe_faster_whisper_preferred(monkeypatch, tmp_path):
    """faster-whisper is used as the first-preference backend."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = transcript_fixture_text()

    # Build fake segments whose joined text equals the fixture text.
    segs = [{"start": 0.0, "end": 1.0, "text": expected}]
    fake_fw_mod, fake_model = _make_faster_whisper_mock(expected, segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    out_path = transcribe.transcribe(audio_path, model="small", language="en", output_dir=out_dir)

    assert out_path == out_dir / "consult.txt"
    assert out_path.read_text(encoding="utf-8") == expected
    fake_fw_mod.WhisperModel.assert_called_once()
    fake_model.transcribe.assert_called_once()


def test_transcribe_faster_whisper_device_auto_cpu(monkeypatch, tmp_path):
    """device='auto' resolves to 'cpu' when CUDA is unavailable, uses int8."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = "Hello."

    segs = [{"start": 0.0, "end": 1.0, "text": expected}]
    fake_fw_mod, _fake_model = _make_faster_whisper_mock(expected, segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    # Ensure torch reports no CUDA.
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    transcribe.transcribe(audio_path, model="small", language="en", output_dir=out_dir, device="auto")

    call_kwargs = fake_fw_mod.WhisperModel.call_args
    assert call_kwargs[1].get("device") == "cpu" or call_kwargs[0][1] == "cpu"
    assert call_kwargs[1].get("compute_type") == "int8" or call_kwargs[0][2] == "int8"


def test_transcribe_faster_whisper_device_cuda(monkeypatch, tmp_path):
    """device='cuda' explicit — uses float16 compute type."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = "Hello."

    segs = [{"start": 0.0, "end": 1.0, "text": expected}]
    fake_fw_mod, _fake_model = _make_faster_whisper_mock(expected, segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    transcribe.transcribe(audio_path, model="small", language="en", output_dir=out_dir, device="cuda")

    call_kwargs = fake_fw_mod.WhisperModel.call_args
    assert call_kwargs[1].get("compute_type") == "float16" or call_kwargs[0][2] == "float16"


def test_transcribe_falls_back_to_openai_whisper_when_faster_missing(monkeypatch, tmp_path):
    """Falls back to openai-whisper when faster-whisper is not installed."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = transcript_fixture_text()

    fail_faster_whisper_import(monkeypatch)

    fake_model = mock.Mock()
    fake_model.transcribe.return_value = {"text": f"  {expected}\n", "segments": []}
    fake_whisper = types.SimpleNamespace(load_model=mock.Mock(return_value=fake_model))
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper)

    out_path = transcribe.transcribe(audio_path, model="small", language="en", output_dir=out_dir)

    assert out_path == out_dir / "consult.txt"
    assert out_path.read_text(encoding="utf-8") == expected
    fake_whisper.load_model.assert_called_once_with("small")


def test_transcribe_python_library(monkeypatch, tmp_path):
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = transcript_fixture_text()

    # Disable faster-whisper so openai-whisper is the first successful backend.
    fail_faster_whisper_import(monkeypatch)

    fake_model = mock.Mock()
    fake_model.transcribe.return_value = {"text": f"  {expected}\n", "segments": []}
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
    fail_faster_whisper_import(monkeypatch)
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
    fail_faster_whisper_import(monkeypatch)
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

    # Use faster-whisper mock (first-preference backend).
    segs = [{"start": 0.0, "end": 1.0, "text": expected}]
    fake_fw_mod, _fake_model = _make_faster_whisper_mock(expected, segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

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


def test_transcribe_diarise_labels_speakers(monkeypatch, tmp_path):
    """Diarised transcript contains [SPEAKER_XX]: prefixes."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"

    segments = [
        {"start": 0.0, "end": 2.0, "text": " Hello there."},
        {"start": 2.5, "end": 4.0, "text": " Hi doctor."},
    ]

    # Use faster-whisper mock.
    fake_fw_mod, _fake_model = _make_faster_whisper_mock("Hello there. Hi doctor.", segments)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    # Mock pyannote pipeline — itertracks must return an iterable list directly.
    mock_turn_0 = mock.Mock()
    mock_turn_0.start = 0.0
    mock_turn_0.end = 2.2

    mock_turn_1 = mock.Mock()
    mock_turn_1.start = 2.3
    mock_turn_1.end = 4.5

    mock_diarization = mock.Mock()
    mock_diarization.itertracks.return_value = [
        (mock_turn_0, None, "SPEAKER_00"),
        (mock_turn_1, None, "SPEAKER_01"),
    ]
    # No speaker_diarization attribute — use the mock directly as the annotation.
    del mock_diarization.speaker_diarization  # ensure hasattr returns False

    mock_pipeline_instance = mock.Mock(return_value=mock_diarization)
    mock_pipeline_cls = mock.Mock()
    mock_pipeline_cls.from_pretrained.return_value = mock_pipeline_instance

    fake_pyannote_audio = types.SimpleNamespace(Pipeline=mock_pipeline_cls)
    fake_pyannote = types.SimpleNamespace(audio=fake_pyannote_audio)
    monkeypatch.setitem(sys.modules, "pyannote", fake_pyannote)
    monkeypatch.setitem(sys.modules, "pyannote.audio", fake_pyannote_audio)

    out_path = transcribe.transcribe(
        audio_path, model="small", language="en",
        output_dir=out_dir, diarise=True, hf_token="fake-token"
    )

    content = out_path.read_text(encoding="utf-8")
    assert "[SPEAKER_00]:" in content
    assert "[SPEAKER_01]:" in content
    assert "Hello there." in content
    assert "Hi doctor." in content


def test_transcribe_diarise_fallback_when_pyannote_missing(monkeypatch, tmp_path):
    """Falls back to plain transcript with a warning when pyannote is not installed."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"
    expected = "Hello there. Hi doctor."

    segs = [{"start": 0.0, "end": 2.0, "text": " Hello there."}]
    fake_fw_mod, _fake_model = _make_faster_whisper_mock(expected, segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    # Simulate pyannote not installed.
    monkeypatch.setitem(sys.modules, "pyannote", None)
    monkeypatch.setitem(sys.modules, "pyannote.audio", None)

    out_path = transcribe.transcribe(
        audio_path, model="small", language="en",
        output_dir=out_dir, diarise=True, hf_token="fake-token"
    )

    content = out_path.read_text(encoding="utf-8")
    # Should fall back to plain text without crashing.
    assert "Hello there." in content


def test_transcribe_device_arg_passed_to_faster_whisper(monkeypatch, tmp_path):
    """--device argument is forwarded to faster-whisper."""
    audio_path = tmp_path / "consult.wav"
    audio_path.write_bytes(b"RIFFfake")
    out_dir = tmp_path / "out"

    segs = [{"start": 0.0, "end": 1.0, "text": "text"}]
    fake_fw_mod, _fake_model = _make_faster_whisper_mock("text", segs)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw_mod)

    monkeypatch.setattr(
        sys, "argv",
        ["transcribe.py", "--audio", str(audio_path), "--output-dir", str(out_dir), "--device", "cpu"],
    )

    transcribe.main()

    call_kwargs = fake_fw_mod.WhisperModel.call_args
    # Positional: WhisperModel(model, device=..., compute_type=...)
    assert call_kwargs[1].get("device") == "cpu" or call_kwargs[0][1] == "cpu"
