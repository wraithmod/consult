import os
import sys
import shutil
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
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
DEFAULT_LLM_MODEL = os.environ.get("CONSULT_DEFAULT_LLM_MODEL", "llama3")
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


@app.on_event("startup")
async def preload_llm_model():
    """Warm Ollama model on server startup to reduce first-request latency."""
    try:
        logger.info(f"Warming Ollama model {DEFAULT_LLM_MODEL}...")
        summarise.warmup_ollama_model(DEFAULT_LLM_MODEL)
        logger.info("Ollama model warmup complete.")
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
    transcript_text = summarise.read_transcript(str(transcript_path))
    prompt = summarise.build_prompt(transcript_text)
    llm_text = summarise.call_ollama(model_llm, prompt)
    note_markdown = summarise.format_soap_markdown(llm_text)

    # Save note
    notes_dir = REPO_ROOT / "notes"
    note_path = summarise.save_note(str(transcript_path), notes_dir, note_markdown)
    logger.info(f"Note saved to {note_path}")

    return {
        "status": "success",
        "audio_file": file_path.name,
        "transcript_file": transcript_path.name,
        "note_file": note_path.name,
        "note": note_markdown,
    }


@app.get("/")
async def web_ui():
    if not WEB_UI_PATH.is_file():
        raise HTTPException(status_code=404, detail="Web UI file not found.")
    return FileResponse(WEB_UI_PATH)


@app.get("/web")
async def web_ui_alias():
    return await web_ui()


@app.post("/process")
async def process_audio(
    file: UploadFile = File(...),
    model_whisper: str = DEFAULT_WHISPER_MODEL,
    model_llm: str = DEFAULT_LLM_MODEL,
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
    except Exception as e:
        logger.exception("Error processing audio")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
