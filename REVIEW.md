# REVIEW.md

## Scope
Reviewed:
- `src/record.py`
- `src/transcribe.py`
- `src/summarise.py`
- `requirements.txt`
- `CLAUDE.md`

Checks performed:
1. Correctness and edge cases
2. Privacy/security (PHI leakage risks)
3. Australian English spelling
4. Consistency with `CLAUDE.md` requirements (local-only, Whisper, Ollama, MBS/PBS)
5. Missing error handling

## What Was Fixed

### `src/record.py`
- Added error handling for audio device enumeration failures (`sounddevice` backend issues now return a clear error instead of a traceback).
- Added handling for `EOFError`/`KeyboardInterrupt` around start/stop prompts so the recorder exits cleanly in non-interactive or interrupted sessions.
- Added audio input stream startup error handling (invalid device, unavailable device, etc.).
- Reduced path exposure in normal output by printing the saved filename rather than the full filesystem path.

### `src/transcribe.py`
- Fixed Whisper CLI fallback error classification:
  - missing `whisper` executable is now handled separately;
  - CLI runtime failures are no longer incorrectly reported as “package/CLI not found”.
- Prevented misleading `FileNotFoundError` when CLI output file is missing after a successful CLI run (now reported as a runtime transcription failure).
- Added file/directory validation for `--audio` and `--audio-dir` (`is_file()` / `is_dir()`).
- Added top-level handling for Whisper library runtime failures to avoid raw tracebacks for common setup/runtime issues.
- Reduced path exposure in normal output and common error messages by printing filenames/directory names only.

### `src/summarise.py`
- Fixed brittle Ollama connectivity error handling by explicitly handling `urllib.error.HTTPError`, `urllib.error.URLError`, and timeouts.
- Added handling for Ollama JSON responses that contain an `error` field.
- Fixed MBS item extraction logic to choose the first valid item number in order of appearance (`23`, `36`, `44`) rather than always preferring `23` if multiple numbers appear in the model output.
- Reduced path exposure in normal output and editor-fallback messages by printing filenames only.
- Tightened transcript input validation to require a file (`is_file()`), not merely an existing path.

## What Is Fine

### `src/record.py`
- Records locally to `audio/` as WAV (local-only, aligned with `CLAUDE.md`).
- Uses 16 kHz mono, which is appropriate for Whisper.
- No temporary files or cloud calls observed.
- Australian English in user-facing text is acceptable.

### `src/transcribe.py`
- Uses local Whisper (`openai-whisper` Python library or local `whisper` CLI fallback), consistent with `CLAUDE.md`.
- Stores transcripts locally in `transcripts/`.
- No external API use observed.
- `requirements.txt` now includes `openai-whisper`, which matches the implementation.

### `src/summarise.py`
- Sends transcript text only to a local Ollama endpoint (`localhost`), consistent with local-only PHI handling.
- Prompt explicitly requests Australian English and references PBS/MBS/GPMP/MHTP context, aligned with project requirements.
- Output is written locally to `notes/`.

### `requirements.txt`
- Includes `sounddevice`, `numpy`, and `openai-whisper` required by current scripts.

## Needs Attention (Not Changed)

### `CLAUDE.md`
- `CLAUDE.md` is generally aligned with the current implementation and explicitly documents local-only processing, Whisper, Ollama, and Australian clinical/MBS/PBS context.
- Minor documentation drift remains possible because `CLAUDE.md` includes both planned and in-progress components (`src/review.py`, `src/pipeline.py`), so keep status notes current as implementation evolves.

### General / Residual Risks
- Filenames can still contain PHI if users manually name files with patient identifiers. The code now avoids printing full paths in normal output, but filename conventions should remain timestamp-based.
- `src/summarise.py` relies on prompt compliance for PBS/MBS inclusion and SOAP structure. The post-processing improves extraction but does not fully validate clinical completeness.
- `src/transcribe.py` loads Whisper models per run/file, which is correct but can be slow for batch transcription (performance issue, not correctness).

## Australian English Review
- Source files are generally consistent with Australian English where clinically relevant (especially `src/summarise.py` prompt text).
- No US spelling issues requiring code changes were identified in user-facing clinical instructions.
