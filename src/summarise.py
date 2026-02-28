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
import time
from datetime import date
from pathlib import Path
import urllib.error
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"

# Overall generation timeout (seconds).
_OVERALL_TIMEOUT = 120
# Per-chunk read timeout (seconds) — used when streaming NDJSON responses.
_CHUNK_TIMEOUT = 10


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
        "Return your response using EXACTLY these section headers on their own line, "
        "in this order:\n"
        "## Consultation Summary\n"
        "## Subjective\n"
        "## Objective\n"
        "## Assessment\n"
        "## Plan\n"
        "## Suggested MBS Item\n"
        "Do not use any other header format. Do not use bold for section headers. "
        "Do not include disclaimers or mention AI."
    )

    return (
        f"{system_instruction}\n\n"
        "Transcript:\n"
        f"{transcript_text}\n"
    )


def call_ollama(model: str, full_prompt: str) -> str:
    """Send *full_prompt* to the local Ollama API and return the complete response.

    Uses streaming (``"stream": true``) so tokens are printed to stderr as they
    arrive, giving the GP immediate visual feedback.  Each chunk is expected
    within *_CHUNK_TIMEOUT* seconds; the overall call must complete within
    *_OVERALL_TIMEOUT* seconds.
    """
    payload = json.dumps(
        {"model": model, "prompt": full_prompt, "stream": True}
    ).encode("utf-8")
    request = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    deadline = time.monotonic() + _OVERALL_TIMEOUT
    accumulated: list[str] = []

    try:
        with urllib.request.urlopen(request, timeout=_CHUNK_TIMEOUT) as response:
            for raw_line in response:
                # Enforce the overall wall-clock timeout.
                if time.monotonic() > deadline:
                    raise TimeoutError(
                        f"Ollama did not finish within {_OVERALL_TIMEOUT} seconds."
                    )

                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    # Ignore non-JSON lines (e.g. blank lines or keep-alives).
                    continue

                error_text = chunk.get("error")
                if isinstance(error_text, str) and error_text.strip():
                    raise RuntimeError(f"Ollama error: {error_text.strip()}")

                token = chunk.get("response", "")
                if token:
                    accumulated.append(token)
                    print(token, end="", flush=True, file=sys.stderr)

                if chunk.get("done"):
                    break

    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Ollama API returned HTTP {exc.code}. Ensure the model '{model}' is available locally."
        ) from exc
    except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            "Could not reach the local Ollama API at "
            f"{OLLAMA_URL}. Ensure Ollama is running and the model is available."
        ) from exc

    # Emit a newline after the streamed tokens so the next log line is clean.
    print(file=sys.stderr)

    full_response = "".join(accumulated).strip()
    if not full_response:
        raise RuntimeError("Ollama response did not contain a usable 'response' field.")
    return full_response


def _extract_section(text: str, label: str, next_labels: list[str]) -> str:
    # Build a pattern that matches the label under any of these header formats:
    #   ## Label        (markdown h2, with or without trailing colon)
    #   ### Label       (markdown h3, with or without trailing colon)
    #   **Label:**      (markdown bold with colon inside)
    #   **Label**:      (markdown bold with colon outside)
    #   Label:          (plain text, any capitalisation)
    escaped = re.escape(label)
    header_pattern = (
        r"(?:"
        r"#{2,3}\s+" + escaped + r":?"   # ## Label  or  ### Label  (optional colon)
        r"|"
        r"\*\*" + escaped + r":\*\*"     # **Label:**
        r"|"
        r"\*\*" + escaped + r"\*\*:"     # **Label**:
        r"|"
        r"^" + escaped + r":"            # Label:  at start of line
        r")"
    )

    match = re.search(header_pattern, text, re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""

    content_start = match.end()
    content_end = len(text)

    for next_label in next_labels:
        next_escaped = re.escape(next_label)
        next_pattern = (
            r"(?:"
            r"#{2,3}\s+" + next_escaped + r":?"
            r"|"
            r"\*\*" + next_escaped + r":\*\*"
            r"|"
            r"\*\*" + next_escaped + r"\*\*:"
            r"|"
            r"^" + next_escaped + r":"
            r")"
        )
        next_match = re.search(next_pattern, text[content_start:], re.IGNORECASE | re.MULTILINE)
        if next_match:
            candidate_end = content_start + next_match.start()
            if candidate_end < content_end:
                content_end = candidate_end

    return text[content_start:content_end].strip(" \n\r*-")


def _extract_consultation_summary(text: str) -> str:
    return _extract_section(text, "Consultation Summary", ["Subjective", "Objective", "Assessment", "Plan", "Suggested MBS Item"])


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

    justification = justification.strip(" \t.,;:()[]")

    if not justification:
        justification = "Based on the documented consultation complexity and duration estimate."
    return f"{chosen} — {justification}"


def format_soap_markdown(llm_text: str) -> str:
    consultation_summary = _extract_consultation_summary(llm_text)
    subjective = _extract_section(llm_text, "Subjective", ["Objective", "Assessment", "Plan", "Suggested MBS Item"])
    objective = _extract_section(llm_text, "Objective", ["Assessment", "Plan", "Suggested MBS Item"])
    assessment = _extract_section(llm_text, "Assessment", ["Plan", "Suggested MBS Item"])
    plan = _extract_section(llm_text, "Plan", ["Suggested MBS Item"])
    suggested_mbs = _extract_mbs_line(llm_text)

    if not consultation_summary:
        consultation_summary = "Unable to extract consultation summary. Review source transcript."
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
        f"## Consultation Summary — {today}\n"
        f"{consultation_summary}\n"
        "\n"
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
