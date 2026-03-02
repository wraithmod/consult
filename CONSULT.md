# CONSULT Code Summary

## What This Program Does

This repository is a local-first GP consultation workflow for:
1. Recording audio (`src/record.py`)
2. Transcribing speech to text (`src/transcribe.py`)
3. Generating a structured Australian GP SOAP note via local Ollama (`src/summarise.py`)
4. Reviewing/approving the note in a terminal UI (`src/review.py`)
5. Orchestrating all steps end-to-end (`src/pipeline.py`)

Design intent: keep PHI local, avoid third-party cloud APIs, and support Australian clinical documentation conventions (SOAP + MBS/PBS context).

## Core Modules

- `src/record.py`
  - Records 16 kHz mono audio with `sounddevice`.
  - Writes timestamped WAV files to `audio/`.

- `src/transcribe.py`
  - Backend order: `faster-whisper` -> `openai-whisper` library -> `whisper` CLI fallback.
  - Supports `--device` (`auto`, `cuda`, `cpu`) and optional diarisation via `pyannote.audio`.
  - Writes transcripts to `transcripts/`.

- `src/summarise.py`
  - Builds a constrained GP prompt for local Ollama (`/api/generate`).
  - Uses streaming NDJSON (`"stream": true`) with token-by-token stderr progress.
  - Parses/normalizes model output into consistent markdown sections and suggested MBS line.
  - Writes notes to `notes/`.

- `src/review.py`
  - Curses UI with note + optional transcript panes.
  - Actions: edit (`$EDITOR`), approve, reject, quit.
  - Approval writes `.approved` sidecar; rejection deletes note file.

- `src/pipeline.py`
  - Main runner for sequential flow: `record -> transcribe -> summarise -> review`.
  - Default mode: in-process module calls.
  - Legacy mode: `--subprocess` keeps subprocess execution path.
  - Preserves progress headers `[1/4]` through `[4/4]`.

- `src/metadata.py`
  - JSON sidecar metadata model and helpers (session IDs, model info, approval state, paths).

## Pipeline Behavior (Operational)

Default invocation:
```bash
python3 src/pipeline.py
```

Useful flags:
```bash
python3 src/pipeline.py --skip-record audio/<file>.wav
python3 src/pipeline.py --model-whisper medium --model-llm llama3
python3 src/pipeline.py --device auto
python3 src/pipeline.py --subprocess
```

Artifacts:
- Audio: `audio/*.wav`
- Transcript: `transcripts/*.txt`
- Note: `notes/*_soap.md`
- Approval marker: `notes/*.approved`

## Test/Quality Snapshot

- Test suite: `PYTHONPATH=. pytest -q tests`
- Current result in this workspace: `53 passed`
- Coverage focus includes:
  - Streaming Ollama response handling
  - faster-whisper integration behavior
  - In-process + subprocess pipeline flows

## Current Outstanding Tasks

### 1) Reconcile agent status documents
Repository guidance files are out of sync:
- `AGENTS.md` reports major tasks complete.
- `CODEX.md` still marks those same tasks as assigned/blocked.
- `CLAUDE.md` still states review/pipeline are in progress.

Action: pick one source of truth and update the others.

### 2) Decide whether to remove or keep legacy subprocess path long-term
`src/pipeline.py` currently supports both:
- Preferred in-process path (faster, tighter error propagation)
- `--subprocess` compatibility path

Action: either formally retain both paths (with tests/docs) or deprecate subprocess mode.

### 3) Validate Gemini CLI usability in this environment
Headless Gemini calls (`gemini -p ...`) were attempted multiple times (including parallel runs) but timed out with no stdout/stderr payload in this runtime.

Action: verify network/auth/runtime behavior for Gemini CLI if it is required for future automation.

## Practical Next Work (Suggested)

1. Sync `AGENTS.md`, `CODEX.md`, and `CLAUDE.md` to one consistent status model.
2. Add/update a single top-level README for onboarding (purpose, setup, pipeline, privacy constraints, commands).
3. If metadata sidecars are part of the desired flow, wire `src/metadata.py` directly into `src/pipeline.py`.
