You are an agent for claude. You are an excellent programmer and are great at solving complex programming requirements in clean and simple ways. You will maintain this file as a list of current abilities and jobs that you will perform as agent. You will be able to give yourself new abilities.

## Abilities
- Python implementation: clean, idiomatic, stdlib-first
- Performance optimisation: profiling, async, in-process refactoring
- Test-driven development: pytest, mocking, fixtures
- CLI tool design: argparse, subprocess management
- Local ML integration: Whisper, Ollama HTTP API
- Audio processing: sounddevice, numpy, WAV I/O

## Current Tasks

### TASK-1: Enable Ollama streaming for summarise.py
**Status:** assigned
**File:** `src/summarise.py`
**Detail:**
- Change `call_ollama()` to use `"stream": true` in the POST body.
- Read the NDJSON response line-by-line, accumulating `response` fields until `"done": true`.
- Print each token to stderr as it arrives so the GP sees progress.
- Keep the function signature identical; return the complete assembled string.
- Timeout per-chunk: 10 s; overall timeout: retain existing 120 s guard (track wall time).

### TASK-2: Add faster-whisper backend to transcribe.py
**Status:** assigned
**File:** `src/transcribe.py`
**Detail:**
- Add `_transcribe_with_faster_whisper(audio_path, model, language, device)` function.
  - Import `faster_whisper.WhisperModel` (guard with `ImportError`).
  - `device` defaults to `"cuda"` if torch.cuda.is_available() else `"cpu"`.
  - Use `compute_type="float16"` for CUDA, `"int8"` for CPU.
  - Return `(full_text, segments)` matching existing tuple contract.
  - Segments must have `start`, `end`, `text` keys (faster-whisper uses named tuples; adapt).
- Insert `faster-whisper` as first attempt in `transcribe()`, before the existing library path.
- Add `--device` argument (default: `"auto"`) to `main()` and pass through.
- Update `requirements.txt` with `faster-whisper>=1.0`.

### TASK-3: Refactor pipeline.py — in-process execution
**Status:** assigned
**File:** `src/pipeline.py`
**Detail:**
- Import `transcribe`, `summarise` etc. as modules and call their functions directly instead
  of launching Python subprocesses.
- Steps remain sequential (record → transcribe → summarise → review).
- Print the same `[1/4]…[4/4]` progress headers.
- Each step's output path is returned from the function, not parsed from stdout.
- Keep `--skip-record`, `--model-whisper`, `--model-llm`, `--device` args.
- Remove `_run_subprocess`, `_extract_output_path`, `PATH_PATTERN` once unused.
- Do NOT break the existing subprocess runner path for backward compat — add a
  `--subprocess` flag that restores the old behaviour.

### TASK-4: Update tests
**Status:** blocked on TASK-1, TASK-2, TASK-3
**File:** `tests/`
**Detail:**
- After the above tasks are implemented, update/add tests so all 38+ tests pass.
- Mock `faster_whisper.WhisperModel` in transcribe tests.
- Add a test for streaming Ollama path (mock chunked NDJSON responses).

## Completed Tasks
*(none yet)*
