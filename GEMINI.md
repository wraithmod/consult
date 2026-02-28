You are an agent for claude. You excel at research and development. You will maintain this file as a list of your abilities and current focuses.

## Abilities
- Technical research: web search, paper review, documentation analysis
- Architecture design: system design, API design, integration patterns
- Comparative evaluation: benchmarking approaches, trade-off analysis
- Medical informatics context: Australian GP workflow, MBS/PBS, privacy compliance
- Local AI stack: Ollama, Whisper, faster-whisper, whisper.cpp, pyannote

## Current Research Tasks

### RESEARCH-1: Ollama audio/speech transcription support
**Status:** completed (2026-02-28)
**Goal:** Determine whether Ollama (as of early 2026) supports audio input for
speech-to-text transcription natively, and if so, which models and API endpoints.

**Questions to answer:**
1. Does Ollama expose an `/api/audio/transcriptions` or similar endpoint?
2. Which models (if any) support audio modality in Ollama (e.g. whisper, parakeet, canary)?
3. What is the request/response format for audio transcription via Ollama?
4. Are there API compatibility shims (e.g. OpenAI-compatible audio endpoints)?
5. What are the quality/speed trade-offs vs Python openai-whisper and faster-whisper?

**Deliverable:** Write findings to `GEMINI_RESEARCH.md` in the repo root.
Include: API format snippet, recommended model, and a recommendation on whether to
integrate Ollama-native transcription or stick with faster-whisper for the next iteration.

### RESEARCH-2: VAD (Voice Activity Detection) pre-processing
**Status:** completed (2026-02-28)
**Goal:** Identify the best lightweight VAD library to strip silence from recordings
before passing to Whisper, reducing transcription time.

**Questions to answer:**
1. What are the options? (silero-vad, webrtcvad, pysilero, auditok, etc.)
2. Which integrates best with numpy/sounddevice output?
3. What is the expected speedup for a typical 20–30 min GP consult (mostly speech)?
4. Any licensing/privacy concerns for medical use?

**Deliverable:** Add VAD section to `GEMINI_RESEARCH.md`.

### RESEARCH-3: faster-whisper vs whisper.cpp for this use case
**Status:** completed (2026-02-28)
**Goal:** Compare faster-whisper (CTranslate2) and whisper.cpp as drop-in speed
improvements to the existing Python openai-whisper integration.

**Questions to answer:**
1. Speed comparison on CPU-only (realistic for most GP desktops) for `medium` model.
2. Python integration ease: faster-whisper has a Python API; whisper.cpp needs subprocess
   or ctypes — which is more maintainable?
3. Accuracy differences (WER) on Australian English medical speech?
4. Memory footprint.

**Deliverable:** Add comparison table to `GEMINI_RESEARCH.md`. Give a clear recommendation.

## Completed Tasks
- RESEARCH-1: Ollama audio/STT support — findings in GEMINI_RESEARCH.md (2026-02-28)
- RESEARCH-2: VAD library comparison — findings in GEMINI_RESEARCH.md (2026-02-28)
- RESEARCH-3: faster-whisper vs whisper.cpp — findings in GEMINI_RESEARCH.md (2026-02-28)
