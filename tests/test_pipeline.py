"""Tests for pipeline.py.

The existing subprocess-path tests use --subprocess mode (legacy) and mock
``subprocess.Popen``.  New in-process path tests mock the imported source
modules directly.
"""

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import pytest

from src import pipeline


@dataclass
class ProcSpec:
    stdout: str
    returncode: int = 0


class FakeProcess:
    def __init__(self, spec: ProcSpec):
        self.stdout = io.StringIO(spec.stdout)
        self._returncode = spec.returncode

    def wait(self):
        return self._returncode


class PopenSequence:
    def __init__(self, specs):
        self._specs = list(specs)
        self.calls = []

    def __call__(self, cmd, **kwargs):
        self.calls.append((cmd, kwargs))
        if not self._specs:
            raise AssertionError(f"Unexpected subprocess.Popen call: {cmd}")
        return FakeProcess(self._specs.pop(0))

    def assert_all_used(self):
        assert not self._specs, f"Unused fake process specs: {self._specs!r}"


def default_args(**overrides):
    """Return a Namespace with the pipeline's default values.

    Sets ``use_subprocess=True`` by default so legacy subprocess tests still
    work with the Popen mock without triggering the in-process path.
    """
    args = argparse.Namespace(
        device=None,
        model_whisper="medium",
        model_llm="llama3",
        skip_record=None,
        use_subprocess=True,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def src_dir() -> Path:
    return Path(pipeline.__file__).resolve().parent


def run_pipeline_with_specs(specs, args=None):
    popen_seq = PopenSequence(specs)
    if args is None:
        args = default_args()

    with mock.patch.object(pipeline, "parse_args", return_value=args), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        pipeline.main()

    popen_seq.assert_all_used()
    return popen_seq.calls


# ---------------------------------------------------------------------------
# Subprocess path tests (legacy behaviour, --subprocess mode)
# ---------------------------------------------------------------------------

def test_full_pipeline_approved(capsys):
    calls = run_pipeline_with_specs(
        [
            ProcSpec("recorded to /tmp/consult.wav\n", 0),
            ProcSpec("transcript saved /tmp/consult.txt\n", 0),
            ProcSpec("note written /tmp/consult.md\n", 0),
            ProcSpec("Approved\n", 0),
        ]
    )

    out = capsys.readouterr().out
    assert "Status:     Approved" in out
    assert "Audio:      /tmp/consult.wav" in out
    assert "Transcript: /tmp/consult.txt" in out
    assert "Note:       /tmp/consult.md" in out

    expected = src_dir()
    assert calls[0][0] == ["python3", str(expected / "record.py")]
    assert calls[1][0] == [
        "python3",
        str(expected / "transcribe.py"),
        "--audio",
        "/tmp/consult.wav",
        "--model",
        "medium",
    ]
    assert calls[2][0] == [
        "python3",
        str(expected / "summarise.py"),
        "--transcript",
        "/tmp/consult.txt",
        "--model",
        "llama3",
        "--no-review",
    ]
    assert calls[3][0] == [
        "python3",
        str(expected / "review.py"),
        "--note",
        "/tmp/consult.md",
        "--transcript",
        "/tmp/consult.txt",
    ]


def test_full_pipeline_rejected(capsys):
    popen_seq = PopenSequence(
        [
            ProcSpec("recorded /tmp/consult.wav\n", 0),
            ProcSpec("saved /tmp/consult.txt\n", 0),
            ProcSpec("saved /tmp/consult.md\n", 0),
            ProcSpec("Rejected\n", 1),
        ]
    )

    with mock.patch.object(pipeline, "parse_args", return_value=default_args()), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        pipeline.main()

    out = capsys.readouterr().out
    assert "Status:     Rejected" in out


def test_pipeline_aborts_on_record_failure(capsys):
    popen_seq = PopenSequence([ProcSpec("record failed\n", 7)])

    with mock.patch.object(pipeline, "parse_args", return_value=default_args()), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        with pytest.raises(SystemExit) as excinfo:
            pipeline.main()

    assert excinfo.value.code == 7
    assert len(popen_seq.calls) == 1
    assert popen_seq.calls[0][0][1].endswith("record.py")
    err = capsys.readouterr().err
    assert "Recording failed with exit code 7" in err


def test_pipeline_aborts_on_transcribe_failure(capsys):
    popen_seq = PopenSequence(
        [
            ProcSpec("saved /tmp/consult.wav\n", 0),
            ProcSpec("transcribe failed\n", 9),
        ]
    )

    with mock.patch.object(pipeline, "parse_args", return_value=default_args()), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        with pytest.raises(SystemExit) as excinfo:
            pipeline.main()

    assert excinfo.value.code == 9
    assert len(popen_seq.calls) == 2
    assert popen_seq.calls[1][0][1].endswith("transcribe.py")
    err = capsys.readouterr().err
    assert "Transcribing failed with exit code 9" in err


def test_pipeline_aborts_on_summarise_failure(capsys):
    popen_seq = PopenSequence(
        [
            ProcSpec("saved /tmp/consult.wav\n", 0),
            ProcSpec("saved /tmp/consult.txt\n", 0),
            ProcSpec("summarise failed\n", 11),
        ]
    )

    with mock.patch.object(pipeline, "parse_args", return_value=default_args()), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        with pytest.raises(SystemExit) as excinfo:
            pipeline.main()

    assert excinfo.value.code == 11
    assert len(popen_seq.calls) == 3
    assert popen_seq.calls[2][0][1].endswith("summarise.py")
    err = capsys.readouterr().err
    assert "Summarising failed with exit code 11" in err


def test_skip_record_flag(tmp_path, capsys):
    wav_path = tmp_path / "fake.wav"
    wav_path.write_text("", encoding="utf-8")
    argv = ["pipeline.py", "--skip-record", str(wav_path), "--subprocess"]
    popen_seq = PopenSequence(
        [
            ProcSpec(f"saved {tmp_path / 'consult.txt'}\\n", 0),
            ProcSpec(f"saved {tmp_path / 'consult.md'}\\n", 0),
            ProcSpec("Approved\\n", 0),
        ]
    )

    with mock.patch.object(sys, "argv", argv), mock.patch.object(
        pipeline.subprocess, "Popen", side_effect=popen_seq
    ):
        pipeline.main()

    assert len(popen_seq.calls) == 3
    assert all(not call[0][1].endswith("record.py") for call in popen_seq.calls)
    transcribe_cmd = popen_seq.calls[0][0]
    assert transcribe_cmd[1].endswith("transcribe.py")
    assert transcribe_cmd[transcribe_cmd.index("--audio") + 1] == str(wav_path)

    out = capsys.readouterr().out
    assert "[1/4] Recording... (skipped)" in out


def test_progress_output(capsys):
    run_pipeline_with_specs(
        [
            ProcSpec("saved /tmp/a.wav\n", 0),
            ProcSpec("saved /tmp/a.txt\n", 0),
            ProcSpec("saved /tmp/a.md\n", 0),
            ProcSpec("Approved\n", 0),
        ]
    )

    out = capsys.readouterr().out
    assert "[1/4] Recording..." in out
    assert "[2/4] Transcribing..." in out
    assert "[3/4] Summarising..." in out
    assert "[4/4] Reviewing..." in out


# ---------------------------------------------------------------------------
# In-process path tests
# ---------------------------------------------------------------------------

def _inprocess_args(**overrides):
    """Args for the in-process (non-subprocess) pipeline path."""
    args = argparse.Namespace(
        device=None,
        model_whisper="medium",
        model_llm="llama3",
        skip_record=None,
        use_subprocess=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_inprocess_full_pipeline_approved(tmp_path, capsys):
    """In-process path calls module functions directly instead of subprocesses."""
    wav_path = tmp_path / "consult.wav"
    wav_path.write_bytes(b"RIFFfake")
    transcript_path = tmp_path / "consult.txt"
    transcript_path.write_text("transcript text", encoding="utf-8")
    note_path = tmp_path / "consult_soap.md"
    note_path.write_text("# Note", encoding="utf-8")

    args = _inprocess_args(skip_record=wav_path)

    with mock.patch.object(pipeline, "parse_args", return_value=args), \
         mock.patch.object(pipeline, "_step_transcribe_inprocess", return_value=transcript_path) as mock_tx, \
         mock.patch.object(pipeline, "_step_summarise_inprocess", return_value=note_path) as mock_sum, \
         mock.patch.object(pipeline, "_step_review_inprocess", return_value=("Approved", 0)) as mock_rev:
        pipeline.main()

    mock_tx.assert_called_once_with(wav_path, "medium", None)
    mock_sum.assert_called_once_with(transcript_path, "llama3")
    mock_rev.assert_called_once_with(note_path, transcript_path)

    out = capsys.readouterr().out
    assert "Status:     Approved" in out
    assert "[1/4] Recording... (skipped)" in out
    assert "[2/4] Transcribing..." in out
    assert "[3/4] Summarising..." in out
    assert "[4/4] Reviewing..." in out


def test_inprocess_record_step_called(tmp_path, capsys):
    """In-process path calls _step_record_inprocess when not skipping."""
    wav_path = tmp_path / "consult.wav"
    wav_path.write_bytes(b"RIFFfake")
    transcript_path = tmp_path / "consult.txt"
    transcript_path.write_text("transcript", encoding="utf-8")
    note_path = tmp_path / "consult_soap.md"
    note_path.write_text("# Note", encoding="utf-8")

    args = _inprocess_args()

    with mock.patch.object(pipeline, "parse_args", return_value=args), \
         mock.patch.object(pipeline, "_step_record_inprocess", return_value=wav_path) as mock_rec, \
         mock.patch.object(pipeline, "_step_transcribe_inprocess", return_value=transcript_path), \
         mock.patch.object(pipeline, "_step_summarise_inprocess", return_value=note_path), \
         mock.patch.object(pipeline, "_step_review_inprocess", return_value=("Approved", 0)):
        pipeline.main()

    mock_rec.assert_called_once_with(None)

    out = capsys.readouterr().out
    assert "[1/4] Recording..." in out
    assert "skipped" not in out.splitlines()[0]


def test_inprocess_no_subprocess_calls(tmp_path):
    """In-process mode never calls subprocess.Popen."""
    wav_path = tmp_path / "consult.wav"
    wav_path.write_bytes(b"RIFFfake")
    transcript_path = tmp_path / "consult.txt"
    transcript_path.write_text("t", encoding="utf-8")
    note_path = tmp_path / "note.md"
    note_path.write_text("#", encoding="utf-8")

    args = _inprocess_args(skip_record=wav_path)

    with mock.patch.object(pipeline, "parse_args", return_value=args), \
         mock.patch.object(pipeline, "_step_transcribe_inprocess", return_value=transcript_path), \
         mock.patch.object(pipeline, "_step_summarise_inprocess", return_value=note_path), \
         mock.patch.object(pipeline, "_step_review_inprocess", return_value=("Approved", 0)), \
         mock.patch.object(pipeline.subprocess, "Popen") as mock_popen:
        pipeline.main()

    mock_popen.assert_not_called()


def test_subprocess_flag_restores_legacy_behaviour(capsys):
    """--subprocess flag routes through the Popen-based path."""
    popen_seq = PopenSequence(
        [
            ProcSpec("recorded /tmp/consult.wav\n", 0),
            ProcSpec("saved /tmp/consult.txt\n", 0),
            ProcSpec("saved /tmp/consult.md\n", 0),
            ProcSpec("Approved\n", 0),
        ]
    )
    args = default_args(use_subprocess=True)

    with mock.patch.object(pipeline, "parse_args", return_value=args), \
         mock.patch.object(pipeline.subprocess, "Popen", side_effect=popen_seq):
        pipeline.main()

    popen_seq.assert_all_used()
    out = capsys.readouterr().out
    assert "Status:     Approved" in out
