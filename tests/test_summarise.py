import argparse
import json
import socket
from pathlib import Path
from unittest import mock
import urllib.error

import pytest

from src import summarise


SYNTHETIC_TRANSCRIPT = Path(__file__).parent / "synthetic" / "transcript_01.txt"


def _urlopen_streaming_cm(ndjson_lines: list[bytes]):
    """Return a mock context manager whose body iterates over *ndjson_lines*."""
    response = mock.MagicMock()
    response.__iter__ = mock.Mock(return_value=iter(ndjson_lines))
    cm = mock.MagicMock()
    cm.__enter__.return_value = response
    cm.__exit__.return_value = False
    return cm


def _single_chunk_cm(response_text: str) -> object:
    """Streaming CM that returns a single done chunk containing *response_text*."""
    line = json.dumps({"response": response_text, "done": True}).encode("utf-8") + b"\n"
    return _urlopen_streaming_cm([line])


def _bad_json_cm() -> object:
    """Streaming CM that returns an unparseable line."""
    return _urlopen_streaming_cm([b"{not-json\n"])


def _run_main_with_args(tmp_path: Path, transcript_path: Path, *, urlopen_side_effect=None, urlopen_cm=None):
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
    elif urlopen_cm is not None:
        patches.append(
            mock.patch.object(
                summarise.urllib.request,
                "urlopen",
                return_value=urlopen_cm,
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
    assert "Consultation Summary" in prompt


def test_call_ollama_success():
    """call_ollama assembles streamed NDJSON tokens into the full response."""
    lines = [
        json.dumps({"response": "SOAP ", "done": False}).encode() + b"\n",
        json.dumps({"response": "note ", "done": False}).encode() + b"\n",
        json.dumps({"response": "text", "done": True}).encode() + b"\n",
    ]
    cm = _urlopen_streaming_cm(lines)

    with mock.patch.object(summarise.urllib.request, "urlopen", return_value=cm):
        result = summarise.call_ollama("llama3", "prompt text")

    assert result == "SOAP note text"


def test_call_ollama_single_chunk():
    """call_ollama works when the model returns everything in one done chunk."""
    cm = _single_chunk_cm("SOAP note text")

    with mock.patch.object(summarise.urllib.request, "urlopen", return_value=cm):
        result = summarise.call_ollama("llama3", "prompt text")

    assert result == "SOAP note text"


def test_call_ollama_streaming_prints_tokens(capsys):
    """Tokens are printed to stderr as they arrive."""
    lines = [
        json.dumps({"response": "Hello", "done": False}).encode() + b"\n",
        json.dumps({"response": " world", "done": True}).encode() + b"\n",
    ]
    cm = _urlopen_streaming_cm(lines)

    with mock.patch.object(summarise.urllib.request, "urlopen", return_value=cm):
        summarise.call_ollama("llama3", "prompt")

    err = capsys.readouterr().err
    assert "Hello" in err
    assert " world" in err


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
    """A non-JSON line in the stream is silently skipped; empty response raises RuntimeError."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run_main_with_args(tmp_path, transcript_path, urlopen_cm=_bad_json_cm())

    assert excinfo.value.code == 1


def test_call_ollama_ollama_error_in_stream(tmp_path):
    """An 'error' field in a streaming chunk raises RuntimeError."""
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")

    error_line = json.dumps({"error": "model not found"}).encode() + b"\n"
    cm = _urlopen_streaming_cm([error_line])

    with pytest.raises(SystemExit) as excinfo:
        _run_main_with_args(tmp_path, transcript_path, urlopen_cm=cm)

    assert excinfo.value.code == 1


def test_format_soap_markdown():
    llm_output = (
        "## Consultation Summary\n"
        "Patient presented with ankle pain and a history of hypertension.\n"
        "Subjective: 52-year-old male with mild recurrent headaches and elevated pharmacy BP readings. No focal neuro symptoms.\n"
        "Objective: BP 158/94 sitting, repeat elevated. Weight 92.4 kg. No red flag features reported.\n"
        "Assessment: Likely essential hypertension with tension-type headaches; no acute neurological red flags.\n"
        "Plan: Start amlodipine 5 mg daily (PBS listed if applicable), lifestyle advice, home BP log, review in 2-4 weeks.\n"
        "Suggested MBS Item: 36 - Level C consultation, approximately 25 minutes.\n"
    ).strip()

    formatted = summarise.format_soap_markdown(llm_output)

    assert "## Consultation Summary" in formatted
    assert "ankle pain" in formatted
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
    assert formatted.count(")") == 0 or "36 —" in formatted  # no stray trailing )


def test_mbs_trailing_punctuation_stripped():
    """Trailing ) . , from LLM output must be stripped from MBS justification."""
    llm_output = "Suggested MBS Item: 36 — Level C consultation (approximately 25 minutes)"
    result = summarise._extract_mbs_line(llm_output)
    assert result == "36 — Level C consultation (approximately 25 minutes"  # outer ) stripped
    assert not result.endswith(")")

    llm_output2 = "Suggested MBS Item: 36 - 39 minutes)"
    result2 = summarise._extract_mbs_line(llm_output2)
    assert not result2.endswith(")")
    assert "39 minutes" in result2


def test_summarise_saves_file(tmp_path):
    transcript_path = tmp_path / "transcript_01.txt"
    transcript_path.write_text(SYNTHETIC_TRANSCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
    output_dir = tmp_path / "notes"
    fake_llm_output = (
        "## Consultation Summary\nPatient presented with headaches.\n"
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
    assert "Patient presented with headaches" in saved


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
