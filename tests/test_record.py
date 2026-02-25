import builtins
import importlib
import sys
import types
import wave
from pathlib import Path
from unittest import mock

import numpy as np
import pytest


def load_record_module(monkeypatch):
    fake_sd = types.SimpleNamespace(
        query_devices=lambda: [],
        default=types.SimpleNamespace(device=(0, None)),
        InputStream=None,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    sys.modules.pop("src.record", None)
    module = importlib.import_module("src.record")
    return importlib.reload(module)


def test_list_devices(monkeypatch, capsys):
    record = load_record_module(monkeypatch)
    devices = [
        {"name": "Speaker", "max_input_channels": 0},
        {"name": "Mic A", "max_input_channels": 1},
        {"name": "Loopback", "max_input_channels": 2},
    ]
    monkeypatch.setattr(record.sd, "query_devices", lambda: devices)
    monkeypatch.setattr(record.sd, "default", types.SimpleNamespace(device=(2, None)))

    record.list_devices()

    out = capsys.readouterr().out
    assert "Available input devices:" in out
    assert "Speaker" not in out
    assert "[1] Mic A" in out
    assert "[2] Loopback (default)" in out


def test_record_saves_wav(monkeypatch, tmp_path):
    record = load_record_module(monkeypatch)
    monkeypatch.setattr(record, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(builtins, "input", mock.Mock(side_effect=["", ""]))

    class FakeTimer:
        def start(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(record.threading, "Thread", lambda *args, **kwargs: FakeTimer())

    frame_a = np.array([[0.25], [-0.25]], dtype=np.float32)
    frame_b = np.array([[0.5], [0.0]], dtype=np.float32)
    captured_kwargs = {}

    class FakeInputStream:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def __enter__(self):
            cb = captured_kwargs["callback"]
            cb(frame_a, len(frame_a), None, None)
            cb(frame_b, len(frame_b), None, None)
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(record.sd, "InputStream", FakeInputStream)

    out_path = record.record()

    assert out_path.exists()
    assert out_path.parent == tmp_path
    assert out_path.suffix == ".wav"
    assert captured_kwargs["samplerate"] == record.SAMPLE_RATE
    assert captured_kwargs["channels"] == record.CHANNELS
    assert captured_kwargs["dtype"] == record.DTYPE

    with wave.open(str(out_path), "rb") as wf:
        assert wf.getframerate() == record.SAMPLE_RATE
        assert wf.getnchannels() == record.CHANNELS
        assert wf.getsampwidth() == 2
        raw = wf.readframes(wf.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16)
    expected = (np.concatenate([frame_a, frame_b], axis=0) * 32767).astype(np.int16).reshape(-1)
    assert samples.dtype == np.int16
    assert np.array_equal(samples, expected)


def test_record_no_frames_exits(monkeypatch, tmp_path):
    record = load_record_module(monkeypatch)
    monkeypatch.setattr(record, "AUDIO_DIR", tmp_path)
    monkeypatch.setattr(builtins, "input", mock.Mock(side_effect=["", ""]))

    class FakeTimer:
        def start(self):
            return None

        def join(self):
            return None

    monkeypatch.setattr(record.threading, "Thread", lambda *args, **kwargs: FakeTimer())

    class EmptyInputStream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(record.sd, "InputStream", EmptyInputStream)

    with pytest.raises(SystemExit) as excinfo:
        record.record()

    assert excinfo.value.code == 1


def test_main_list_devices_flag(monkeypatch):
    record = load_record_module(monkeypatch)
    list_mock = mock.Mock()
    monkeypatch.setattr(record, "list_devices", list_mock)
    monkeypatch.setattr(record, "record", mock.Mock())
    monkeypatch.setattr(sys, "argv", ["record.py", "--list-devices"])

    record.main()

    list_mock.assert_called_once_with()
    record.record.assert_not_called()


def test_main_record_flag(monkeypatch):
    record = load_record_module(monkeypatch)
    rec_mock = mock.Mock(return_value=Path("dummy.wav"))
    monkeypatch.setattr(record, "record", rec_mock)
    monkeypatch.setattr(record, "list_devices", mock.Mock())
    monkeypatch.setattr(sys, "argv", ["record.py", "--device", "0"])

    record.main()

    rec_mock.assert_called_once_with(device=0)
    record.list_devices.assert_not_called()
