import argparse
import json
import socket
from pathlib import Path
from unittest import mock
import urllib.error

import pytest

from src import summarise


SYNTHETIC_TRANSCRIPT = Path(__file__).parent / "synthetic" / "transcript_01.txt"


def _urlopen_cm_with_bytes(payload: bytes):
    response = mock.Mock()
    response.read.return_value = payload
    cm = mock.MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def _run_main_with_args(tmp_path: Path, transcript_path: Path, *, urlopen_side_effect=None, urlopen_bytes=None):
    args = argparse.Namespace(
        transcript=str(transcript_path),
        model="llama3",
        output_dir=str(tmp_path / "notes"),
        no_review=True,
    )
    patches = [mock.patch.object(summarise, "parse_args", return_value=args)]

    if urlopen_side_effect is not None:
        patches.append(
            mock.patch.object(summarise.urllib.request, "urlopen", side_effect=urlopen_side_effect)
        )
    elif urlopen_bytes is not None:
        patches.append(
            mock.patch.object(
                summarise.urllib.request,
                "urlopen",
                return_value=_urlopen_cm_with_bytes(urlopen_bytes),
            )
        )

    with patches[0]:
        if len(patches) == 1:
            summarise.main()
            return
        with patches[1]:
            summarise.main()


def test_build_prompt_contains_australian_english():
    prompt = summarise.build_prompt("Doctor: Hello\nPatient: Example transcript")

    assert "behaviour" in prompt
    assert "PBS" in prompt
    assert "MBS" in prompt
    assert "GP Management Plan item 721" in prompt
    assert "Transcript:" in prompt


def test_call_ollama_success():
    payload = json.dumps({"response": "SOAP note text"}).encode("utf-8")

    with mock.patch.object(
        summarise.urllib.request,
        "urlopen",
        return_value=_urlopen_cm_with_bytes(payload),
    ):
        result = summarise.call_ollama("llama3", "prompt text")

    assert result == "SOAP note text"


def test_call_ollama_connection_error(tmp_path):
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main_with_args(
            tmp_path,
            transcript_path,
            urlopen_side_effect=urllib.error.URLError("connection refused"),
        )

    assert excinfo.value.code == 1


@pytest.mark.parametrize("timeout_exc", [TimeoutError("timeout"), socket.timeout("timeout")])
def test_call_ollama_timeout(tmp_path, timeout_exc):
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main_with_args(tmp_path, transcript_path, urlopen_side_effect=timeout_exc)

    assert excinfo.value.code == 1


def test_call_ollama_bad_json(tmp_path):
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main_with_args(tmp_path, transcript_path, urlopen_bytes=b"{not-json")

    assert excinfo.value.code == 1


def test_format_soap_markdown():
    llm_output = """
Subjective: 52-year-old male with mild recurrent headaches and elevated pharmacy BP readings. No focal neuro symptoms.
Objective: BP 158/94 sitting, repeat elevated. Weight 92.4 kg. No red flag features reported.
Assessment: Likely essential hypertension with tension-type headaches; no acute neurological red flags.
Plan: Start amlodipine 5 mg daily (PBS listed if applicable), lifestyle advice, home BP log, review in 2-4 weeks.
Suggested MBS Item: 36 - Level C consultation, approximately 25 minutes.
""".strip()

    formatted = summarise.format_soap_markdown(llm_output)

    assert "## SOAP Note" in formatted
    assert "**Subjective:**" in formatted
    assert "mild recurrent headaches" in formatted
    assert "**Objective:**" in formatted
    assert "BP 158/94" in formatted
    assert "**Assessment:**" in formatted
    assert "essential hypertension" in formatted
    assert "**Plan:**" in formatted
    assert "amlodipine 5 mg daily" in formatted
    assert "**Suggested MBS Item:** 36" in formatted


def test_summarise_saves_file(tmp_path):
    transcript_path = tmp_path / "transcript_01.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "notes"
    fake_llm_output = (
        "Subjective: Headaches and elevated BP.\n"
        "Objective: BP elevated in clinic.\n"
        "Assessment: Probable hypertension.\n"
        "Plan: Start treatment and review.\n"
        "Suggested MBS Item: 36 - Level C consultation.\n"
    )
    args = argparse.Namespace(
        transcript=str(transcript_path),
        model="llama3",
        output_dir=str(output_dir),
        no_review=True,
    )

    with mock.patch.object(summarise, "parse_args", return_value=args), mock.patch.object(
        summarise, "call_ollama", return_value=fake_llm_output
    ):
        summarise.main()

    note_path = output_dir / "transcript_01_soap.md"
    assert note_path.exists()
    assert note_path.name == "transcript_01_soap.md"
    saved = note_path.read_text(encoding="utf-8")
    assert "## SOAP Note" in saved
    assert "Probable hypertension" in saved


def test_summarise_empty_transcript_exits(tmp_path):
    transcript_path = tmp_path / "empty.txt"
    transcript_path.write_text("", encoding="utf-8")
    args = argparse.Namespace(
        transcript=str(transcript_path),
        model="llama3",
        output_dir=str(tmp_path / "notes"),
        no_review=True,
    )
    fake_llm_output = (
        "Subjective: Example.\nObjective: Example.\nAssessment: Example.\nPlan: Example.\nSuggested MBS Item: 23 - Example."
    )

    with mock.patch.object(summarise, "parse_args", return_value=args), mock.patch.object(
        summarise, "call_ollama", return_value=fake_llm_output
    ):
        with pytest.raises(SystemExit) as excinfo:
            summarise.main()

    assert excinfo.value.code == 1


def test_summarise_missing_transcript_exits(tmp_path):
    missing_path = tmp_path / "missing.txt"
    args = argparse.Namespace(
        transcript=str(missing_path),
        model="llama3",
        output_dir=str(tmp_path / "notes"),
        no_review=True,
    )

    with mock.patch.object(summarise, "parse_args", return_value=args):
        with pytest.raises(SystemExit) as excinfo:
            summarise.main()

    assert excinfo.value.code == 1
