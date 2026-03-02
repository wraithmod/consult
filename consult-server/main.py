import os
import sys
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
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

# Ensure audio directory exists
AUDIO_DIR = REPO_ROOT / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


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


@app.post("/process")
async def process_audio(file: UploadFile = File(...), model_whisper: str = "medium", model_llm: str = DEFAULT_LLM_MODEL):
    """
    Receives a WAV file, transcribes it, and generates a SOAP note.
    """
    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only WAV files are supported.")

    # Save the uploaded file
    file_path = AUDIO_DIR / file.filename
    logger.info(f"Saving uploaded file to {file_path}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Step 1: Transcribe
        logger.info(f"Starting transcription with model {model_whisper}...")
        transcript_path = transcribe.transcribe(
            audio_path=file_path,
            model=model_whisper,
            language="en"
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
            "audio_file": file.filename,
            "transcript_file": transcript_path.name,
            "note_file": note_path.name,
            "note": note_markdown
        }

    except Exception as e:
        logger.exception("Error processing audio")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
