"""Generate an Australian GP SOAP note from a local transcript using Ollama.

Usage:
    python src/summarise.py --transcript transcripts/example.txt
    python src/summarise.py --transcript transcripts/example.txt --model llama3.1
    python src/summarise.py --transcript transcripts/example.txt --no-review

Privacy and compliance:
    - PHI remains local to your machine.
    - This tool sends transcript text only to a locally hosted Ollama instance
      at http://localhost:11434 and does not call external APIs.
    - Designed to support workflows aligned with the Australian Privacy Act 1988.
"""

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from datetime import date
from pathlib import Path
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an Australian GP SOAP note from a transcript using local Ollama."
    )
    parser.add_argument(
        "--transcript",
        required=True,
        help="path to a .txt transcript file",
    )
    parser.add_argument(
        "--model",
        default="llama3",
        help="Ollama LLM model name (default: llama3)",
    )
    parser.add_argument(
        "--output-dir",
        default="../notes/",
        help="notes output directory, default: ../notes/ relative to src/",
    )
    parser.add_argument(
        "--no-review",
        action="store_true",
        help="skip opening in editor after generation",
    )
    return parser.parse_args()


def resolve_output_dir(output_dir_arg: str) -> Path:
    output_dir = Path(output_dir_arg)
    if output_dir.is_absolute():
        return output_dir
    src_dir = Path(__file__).resolve().parent
    return (src_dir / output_dir).resolve()


def read_transcript(path_str: str) -> str:
    transcript_path = Path(path_str)
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path.name}")
    if transcript_path.suffix.lower() != ".txt":
        raise ValueError("Transcript file must be a .txt file")
    transcript_text = transcript_path.read_text(encoding="utf-8")
    if not transcript_text.strip():
        print("Error: Transcript file is empty.", file=sys.stderr)
        sys.exit(1)
    return transcript_text


def build_prompt(transcript_text: str) -> str:
    system_instruction = (
        "You are an Australian GP clinical documentation assistant. "
        "Use Australian English spelling throughout (e.g., behaviour, anaesthesia, "
        "haematology, paediatric, organise, recognise). "
        "Follow Australian GP clinical conventions. "
        "Where prescriptions are relevant, include PBS item codes. "
        "Suggest an MBS item number using these rules only: "
        "23 = Level B 6-19 min; 36 = Level C 20-39 min; 44 = Level D 40+ min. "
        "Reference GP Management Plan item 721 and Mental Health Treatment Plan items "
        "2710/2712 where clinically applicable. "
        "Return only a SOAP note and suggested MBS item in plain text with clear labels for "
        "Subjective, Objective, Assessment, Plan, and Suggested MBS Item. "
        "Do not include disclaimers or mention AI."
    )

    return (
        f"{system_instruction}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n"
    )


def call_ollama(model: str, full_prompt: str) -> str:
    payload = json.dumps(
        {"model": model, "prompt": full_prompt, "stream": False}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Ollama API returned HTTP {exc.code}. Ensure the model '{model}' is available locally."
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Could not reach the local Ollama API at "
            f"{OLLAMA_URL}. Ensure Ollama is running and the model is available."
        ) from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Ollama returned invalid JSON.") from exc

    error_text = parsed.get("error")
    if isinstance(error_text, str) and error_text.strip():
        raise RuntimeError(f"Ollama error: {error_text.strip()}")

    response_text = parsed.get("response")
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Ollama response did not contain a usable 'response' field.")
    return response_text.strip()


def _extract_section(text: str, label: str, next_labels: list[str]) -> str:
    lower_text = text.lower()
    markers = [f"**{label.lower()}:**", f"{label.lower()}:", f"## {label.lower()}"]
    start = -1
    start_len = 0
    for marker in markers:
        idx = lower_text.find(marker)
        if idx != -1:
            start = idx
            start_len = len(marker)
            break
    if start == -1:
        return ""

    content_start = start + start_len
    content_end = len(text)
    for next_label in next_labels:
        for marker in (
            f"**{next_label.lower()}:**",
            f"{next_label.lower()}:",
            f"## {next_label.lower()}",
        ):
            idx = lower_text.find(marker, content_start)
            if idx != -1 and idx < content_end:
                content_end = idx

    return text[content_start:content_end].strip(" \n\r*-")


def _extract_mbs_line(text: str) -> str:
    lower_text = text.lower()
    anchors = ["suggested mbs item", "mbs item", "mbs"]
    segment = text
    for anchor in anchors:
        idx = lower_text.find(anchor)
        if idx != -1:
            segment = text[idx: idx + 300]
            break

    match = re.search(r"\b(23|36|44)\b", segment)
    chosen = match.group(1) if match else "23"

    justification = ""
    if "—" in segment:
        justification = segment.split("—", 1)[1].splitlines()[0].strip()
    elif "-" in segment:
        justification = segment.split("-", 1)[1].splitlines()[0].strip()
    elif ":" in segment:
        tail = segment.split(":", 1)[1]
        justification = tail.splitlines()[0].strip()

    if not justification:
        justification = "Based on the documented consultation complexity and duration estimate."
    return f"{chosen} — {justification}"


def format_soap_markdown(llm_text: str) -> str:
    subjective = _extract_section(llm_text, "Subjective", ["Objective", "Assessment", "Plan", "Suggested MBS Item"])
    objective = _extract_section(llm_text, "Objective", ["Assessment", "Plan", "Suggested MBS Item"])
    assessment = _extract_section(llm_text, "Assessment", ["Plan", "Suggested MBS Item"])
    plan = _extract_section(llm_text, "Plan", ["Suggested MBS Item"])
    suggested_mbs = _extract_mbs_line(llm_text)

    if not subjective:
        subjective = "Unable to reliably extract Subjective section from model output. Review source transcript and regenerate."
    if not objective:
        objective = "Unable to reliably extract Objective section from model output."
    if not assessment:
        assessment = "Unable to reliably extract Assessment section from model output."
    if not plan:
        plan = "Unable to reliably extract Plan section from model output. Include PBS codes for prescriptions where applicable."

    today = date.today().isoformat()
    return (
        f"## SOAP Note — {today}\n"
        f"**Subjective:** {subjective}\n"
        f"**Objective:** {objective}\n"
        f"**Assessment:** {assessment}\n"
        f"**Plan:** {plan}\n"
        "---\n"
        f"**Suggested MBS Item:** {suggested_mbs}\n"
        "*(23 = Level B 6–19 min; 36 = Level C 20–39 min; 44 = Level D 40+ min)*\n"
    )


def save_note(transcript_path_str: str, output_dir: Path, content: str) -> Path:
    transcript_path = Path(transcript_path_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    note_path = output_dir / f"{transcript_path.stem}_soap.md"
    note_path.write_text(content, encoding="utf-8")
    return note_path


def maybe_open_in_editor(note_path: Path) -> None:
    editor = os.environ.get("EDITOR", "nano")
    try:
        subprocess.run([editor, str(note_path)], check=False)
    except FileNotFoundError:
        print(
            f"Editor '{editor}' was not found. Note saved as: {note_path.name}",
            file=sys.stderr,
        )


def main() -> None:
    args = parse_args()

    try:
        transcript_text = read_transcript(args.transcript)
        prompt = build_prompt(transcript_text)
        llm_text = call_ollama(args.model, prompt)
        note_markdown = format_soap_markdown(llm_text)
        output_dir = resolve_output_dir(args.output_dir)
        note_path = save_note(args.transcript, output_dir, note_markdown)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        sys.exit(130)

    print(note_path.name)

    if not args.no_review:
        maybe_open_in_editor(note_path)


if __name__ == "__main__":
    main()
