import os
import re
import sys
import shutil
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consult-server")

# Add the consult/src directory to sys.path to import modules
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.append(str(SRC_DIR))

try:
    import transcribe
    import summarise
except ImportError as e:
    logger.error(f"Failed to import consult modules: {e}")
    logger.error(f"SRC_DIR: {SRC_DIR}")
    sys.exit(1)

app = FastAPI(title="Consult Audio Processing Server")
DEFAULT_LLM_MODEL = os.environ.get("CONSULT_DEFAULT_LLM_MODEL", "gemma3:1b")
DEFAULT_WHISPER_MODEL = os.environ.get("CONSULT_DEFAULT_WHISPER_MODEL", "medium")
ALLOWED_AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".mp4", ".webm", ".ogg", ".flac", ".aac"
}
CONTENT_TYPE_TO_SUFFIX = {
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/aac": ".aac",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
    "video/webm": ".webm",
    "audio/flac": ".flac",
}

# Ensure audio directory exists
AUDIO_DIR = REPO_ROOT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
WEB_UI_PATH = REPO_ROOT / "consult-server" / "web" / "index.html"
TRANSCRIPTS_DIR = REPO_ROOT / "transcripts"
NOTES_DIR = REPO_ROOT / "notes"
TIMESTAMP_PATTERN = re.compile(r"^(\d{8}_\d{6}_\d{6})")
NOTE_FILE_PATTERN = re.compile(r"^(?P<case>.+)_soap(?:_v(?P<ver>\d+))?\.md$")


@app.on_event("startup")
async def preload_llm_model():
    """Warm Ollama model on server startup to reduce first-request latency."""
    try:
        logger.info(f"Warming Ollama model {DEFAULT_LLM_MODEL}...")
        summarise.warmup_ollama_model(DEFAULT_LLM_MODEL)
        logger.info("Ollama model warmup complete.")
        _log_ollama_runtime(DEFAULT_LLM_MODEL)
    except Exception:
        # Non-fatal: requests can still trigger model load on demand.
        logger.exception("Ollama warmup failed; continuing without preloaded model.")


def _safe_audio_suffix(filename: str, content_type: str | None) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in ALLOWED_AUDIO_SUFFIXES:
        return suffix
    inferred = CONTENT_TYPE_TO_SUFFIX.get((content_type or "").lower())
    if inferred:
        return inferred
    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported audio format. Use one of: "
            f"{', '.join(sorted(ALLOWED_AUDIO_SUFFIXES))}"
        ),
    )


def _build_saved_audio_path(original_filename: str, content_type: str | None) -> Path:
    suffix = _safe_audio_suffix(original_filename, content_type)
    stem = Path(original_filename).stem or "upload"
    safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_")) or "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return AUDIO_DIR / f"{timestamp}_{safe_stem}{suffix}"


def _parse_case_timestamp(case_id: str, fallback_path: Path) -> datetime:
    match = TIMESTAMP_PATTERN.match(case_id)
    if match:
        try:
            return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S_%f")
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_path.stat().st_mtime)


def _safe_file_path(base_dir: Path, filename: str) -> Path:
    candidate = (base_dir / filename).resolve()
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    if base_dir.resolve() not in candidate.parents:
        raise HTTPException(status_code=400, detail="Invalid file path.")
    return candidate


def _parse_note_version(filename: str, fallback_path: Path | None = None) -> tuple[int, datetime]:
    match = NOTE_FILE_PATTERN.match(filename)
    version = 1
    if match and match.group("ver"):
        try:
            version = int(match.group("ver"))
        except ValueError:
            version = 1

    if fallback_path and fallback_path.exists():
        ts = datetime.fromtimestamp(fallback_path.stat().st_mtime)
    else:
        ts = datetime.min
    return version, ts


def _build_note_versions(note_files: list[str]) -> list[dict]:
    enriched: list[tuple[int, datetime, str]] = []
    seen: set[str] = set()
    for filename in note_files:
        if filename in seen:
            continue
        seen.add(filename)
        note_path = NOTES_DIR / filename
        version, ts = _parse_note_version(filename, note_path if note_path.exists() else None)
        enriched.append((version, ts, filename))

    enriched.sort(key=lambda item: (item[0], item[1], item[2]))
    versions: list[dict] = []
    for version, _ts, filename in enriched:
        versions.append(
            {
                "file": filename,
                "url": f"/artifacts/note/{filename}",
                "version": version,
                "label": f"v{version}",
            }
        )
    return versions


def _next_note_version_path(consult_id: str) -> Path:
    existing_versions = [1]
    for note_path in NOTES_DIR.glob(f"{consult_id}_soap*.md"):
        version, _ = _parse_note_version(note_path.name, note_path)
        existing_versions.append(version)
    next_version = max(existing_versions) + 1
    return NOTES_DIR / f"{consult_id}_soap_v{next_version}.md"


def _consult_or_404(consult_id: str) -> dict:
    for consult in _build_consult_index():
        if consult["consult_id"] == consult_id:
            return consult
    raise HTTPException(status_code=404, detail="Consult not found.")


def _log_ollama_runtime(model: str) -> None:
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        logger.warning("Could not inspect Ollama runtime: %s", exc)
        return

    if result.returncode != 0:
        logger.warning("`ollama ps` failed with code %s", result.returncode)
        return

    for line in result.stdout.splitlines():
        if line.strip().startswith(model):
            compact = " ".join(line.split())
            logger.info("Ollama runtime for %s: %s", model, compact)
            ratio = re.search(r"(\\d+)%/(\\d+)%\\s+CPU/GPU", line)
            if ratio:
                gpu_pct = int(ratio.group(2))
                if gpu_pct == 0:
                    logger.warning(
                        "Ollama is using CPU only for %s. Enable GPU acceleration for best latency.",
                        model,
                    )
            return

    logger.warning("Model %s not visible in `ollama ps` output after warmup.", model)


def _build_consult_index() -> list[dict]:
    records: dict[str, dict] = {}

    def ensure_record(case_id: str, path: Path) -> dict:
        record = records.get(case_id)
        if record is None:
            created_at = _parse_case_timestamp(case_id, path)
            record = {
                "consult_id": case_id,
                "created_at": created_at.isoformat(),
                "created_ts": created_at,
                "audio_file": None,
                "transcript_file": None,
                "note_files": [],
            }
            records[case_id] = record
        return record

    if AUDIO_DIR.exists():
        for audio_path in sorted(AUDIO_DIR.iterdir()):
            if not audio_path.is_file():
                continue
            case_id = audio_path.stem
            record = ensure_record(case_id, audio_path)
            record["audio_file"] = audio_path.name

    if TRANSCRIPTS_DIR.exists():
        for transcript_path in sorted(TRANSCRIPTS_DIR.glob("*.txt")):
            case_id = transcript_path.stem
            record = ensure_record(case_id, transcript_path)
            record["transcript_file"] = transcript_path.name

    if NOTES_DIR.exists():
        for note_path in sorted(NOTES_DIR.glob("*.md")):
            match = NOTE_FILE_PATTERN.match(note_path.name)
            if not match:
                continue
            case_id = match.group("case")
            record = ensure_record(case_id, note_path)
            record["note_files"].append(note_path.name)

    consults = []
    for record in records.values():
        note_versions = _build_note_versions(record["note_files"])
        latest_note = note_versions[-1] if note_versions else None
        consult = {
            "consult_id": record["consult_id"],
            "created_at": record["created_at"],
            "audio_file": record["audio_file"],
            "transcript_file": record["transcript_file"],
            "note_file": latest_note["file"] if latest_note else None,
            "audio_url": (
                f"/artifacts/audio/{record['audio_file']}" if record["audio_file"] else None
            ),
            "transcript_url": (
                f"/artifacts/transcript/{record['transcript_file']}"
                if record["transcript_file"]
                else None
            ),
            "note_url": latest_note["url"] if latest_note else None,
            "note_versions": note_versions,
            "note_version_count": len(note_versions),
            "is_complete": bool(
                record["audio_file"] and record["transcript_file"] and latest_note
            ),
        }
        consults.append((record["created_ts"], consult))

    consults.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in consults]


def _process_saved_audio(file_path: Path, model_whisper: str, model_llm: str) -> dict:
    # Step 1: Transcribe
    logger.info(f"Starting transcription with model {model_whisper}...")
    transcript_path = transcribe.transcribe(
        audio_path=file_path,
        model=model_whisper,
        language="en",
    )
    logger.info(f"Transcription saved to {transcript_path}")

    # Step 2: Summarise
    logger.info(f"Starting summarisation with model {model_llm}...")
    try:
        transcript_text = summarise.read_transcript(str(transcript_path))
    except SystemExit as exc:
        # summarise.read_transcript() exits on empty input in CLI mode; convert to API error.
        raise HTTPException(
            status_code=422,
            detail="Transcript was empty. Please record clearer speech and try again.",
        ) from exc
    prompt = summarise.build_prompt(transcript_text)
    llm_text = summarise.call_ollama(model_llm, prompt)
    note_markdown = summarise.format_soap_markdown(llm_text)

    # Save note
    notes_dir = NOTES_DIR
    note_path = summarise.save_note(str(transcript_path), notes_dir, note_markdown)
    logger.info(f"Note saved to {note_path}")

    consult_id = file_path.stem
    return {
        "status": "success",
        "consult_id": consult_id,
        "audio_file": file_path.name,
        "transcript_file": transcript_path.name,
        "note_file": note_path.name,
        "audio_url": f"/artifacts/audio/{file_path.name}",
        "transcript_url": f"/artifacts/transcript/{transcript_path.name}",
        "note_url": f"/artifacts/note/{note_path.name}",
        "note": note_markdown,
    }


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_ollama_tokens(model: str, prompt: str):
    payload = json.dumps(summarise._build_generate_payload(model, prompt, stream=True)).encode("utf-8")
    request = urllib.request.Request(
        summarise.OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    deadline = time.monotonic() + getattr(summarise, "_OVERALL_TIMEOUT", 240)
    read_timeout = getattr(summarise, "_READ_TIMEOUT", 180)

    with urllib.request.urlopen(request, timeout=read_timeout) as response:
        for raw_line in response:
            if time.monotonic() > deadline:
                raise RuntimeError("Ollama generation exceeded overall timeout.")

            line = raw_line.decode("utf-8").strip()
            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue

            error_text = chunk.get("error")
            if isinstance(error_text, str) and error_text.strip():
                raise RuntimeError(f"Ollama error: {error_text.strip()}")

            token = chunk.get("response", "")
            if token:
                yield token

            if chunk.get("done"):
                break


@app.get("/")
async def web_ui():
    if not WEB_UI_PATH.is_file():
        raise HTTPException(status_code=404, detail="Web UI file not found.")
    return FileResponse(WEB_UI_PATH)


@app.get("/web")
async def web_ui_alias():
    return await web_ui()


@app.get("/consults")
async def list_consults(limit: int = 200):
    if limit <= 0 or limit > 1000:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 1000.")
    consults = _build_consult_index()
    return {"consults": consults[:limit]}


@app.get("/consults/{consult_id}")
async def get_consult(consult_id: str):
    return _consult_or_404(consult_id)


@app.post("/consults/{consult_id}/regenerate")
async def regenerate_consult_summary(
    consult_id: str,
    model_llm: str = Form(DEFAULT_LLM_MODEL),
):
    consult = _consult_or_404(consult_id)
    transcript_file = consult.get("transcript_file")
    if not transcript_file:
        raise HTTPException(status_code=422, detail="Consult has no transcript to summarise.")

    transcript_path = _safe_file_path(TRANSCRIPTS_DIR, transcript_file)
    logger.info("Regenerating summary for %s with model %s", consult_id, model_llm)

    transcript_text = summarise.read_transcript(str(transcript_path))
    prompt = summarise.build_prompt(transcript_text)
    llm_text = summarise.call_ollama(model_llm, prompt)
    note_markdown = summarise.format_soap_markdown(llm_text)

    note_path = _next_note_version_path(consult_id)
    note_path.write_text(note_markdown, encoding="utf-8")

    refreshed = _consult_or_404(consult_id)
    return {
        "status": "success",
        "consult_id": consult_id,
        "model_llm": model_llm,
        "note_file": note_path.name,
        "note_url": f"/artifacts/note/{note_path.name}",
        "note": note_markdown,
        "consult": refreshed,
    }


@app.get("/artifacts/audio/{filename}")
async def get_audio_artifact(filename: str):
    path = _safe_file_path(AUDIO_DIR, filename)
    return FileResponse(path)


@app.get("/artifacts/transcript/{filename}")
async def get_transcript_artifact(filename: str):
    path = _safe_file_path(TRANSCRIPTS_DIR, filename)
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/artifacts/note/{filename}")
async def get_note_artifact(filename: str):
    path = _safe_file_path(NOTES_DIR, filename)
    return FileResponse(path, media_type="text/markdown; charset=utf-8")


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    model_whisper: str = Form(DEFAULT_WHISPER_MODEL),
    model_llm: str = Form(DEFAULT_LLM_MODEL),
):
    """
    Receives an audio file, transcribes it, and generates a SOAP note.
    Backward compatible with CLI WAV uploads and browser MediaRecorder blobs.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    # Save the uploaded file
    file_path = _build_saved_audio_path(file.filename, file.content_type)
    logger.info(f"Saving uploaded file to {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        return _process_saved_audio(file_path, model_whisper, model_llm)
    except HTTPException:
        raise
    except SystemExit as exc:
        logger.exception("Audio processing exited early")
        raise HTTPException(
            status_code=422,
            detail=(
                "Audio could not be transcribed. "
                "Please check microphone input and try a longer, clearer recording."
            ),
        ) from exc
    except Exception as e:
        logger.exception("Error processing audio")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_stream")
async def process_audio_stream(
    file: UploadFile = File(...),
    model_whisper: str = Form(DEFAULT_WHISPER_MODEL),
    model_llm: str = Form(DEFAULT_LLM_MODEL),
):
    """Stream consult processing progress and summary tokens as Server-Sent Events."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing uploaded filename.")

    file_path = _build_saved_audio_path(file.filename, file.content_type)
    logger.info(f"Saving uploaded file to {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    def event_stream():
        try:
            yield _sse("stage", {"name": "transcription_started", "model_whisper": model_whisper})
            transcript_path = transcribe.transcribe(
                audio_path=file_path,
                model=model_whisper,
                language="en",
            )
            yield _sse(
                "stage",
                {
                    "name": "transcription_done",
                    "transcript_file": transcript_path.name,
                    "transcript_url": f"/artifacts/transcript/{transcript_path.name}",
                },
            )

            transcript_text = summarise.read_transcript(str(transcript_path))
            prompt = summarise.build_prompt(transcript_text)
            yield _sse("stage", {"name": "summarisation_started", "model_llm": model_llm})

            token_count = 0
            llm_parts: list[str] = []
            for token in _stream_ollama_tokens(model_llm, prompt):
                llm_parts.append(token)
                token_count += 1
                yield _sse("token", {"text": token})

            llm_text = "".join(llm_parts).strip()
            if not llm_text:
                raise RuntimeError("Ollama returned an empty summary response.")

            note_markdown = summarise.format_soap_markdown(llm_text)
            note_path = summarise.save_note(str(transcript_path), NOTES_DIR, note_markdown)

            consult_id = file_path.stem
            payload = {
                "status": "success",
                "consult_id": consult_id,
                "audio_file": file_path.name,
                "transcript_file": transcript_path.name,
                "note_file": note_path.name,
                "audio_url": f"/artifacts/audio/{file_path.name}",
                "transcript_url": f"/artifacts/transcript/{transcript_path.name}",
                "note_url": f"/artifacts/note/{note_path.name}",
                "note": note_markdown,
                "token_count": token_count,
            }
            yield _sse("done", payload)
        except HTTPException as exc:
            yield _sse("error", {"detail": exc.detail})
        except SystemExit:
            yield _sse(
                "error",
                {
                    "detail": (
                        "Audio could not be transcribed. "
                        "Please check microphone input and try a longer, clearer recording."
                    )
                },
            )
        except (TimeoutError, socket.timeout, urllib.error.URLError) as exc:
            yield _sse("error", {"detail": f"Timed out while waiting for Ollama: {exc}"})
        except Exception as exc:
            logger.exception("Error in streamed processing")
            yield _sse("error", {"detail": str(exc)})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
