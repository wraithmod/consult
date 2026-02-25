# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

AI-assisted medical consult transcription application for a GP registrar. Captures and processes clinical consultations, summarising them into structured medical documentation — all processed and stored **locally**. No patient health information (PHI) is transmitted to external services.

**Current status (25 February 2026):**
- `src/record.py`: implemented — WAV audio capture, 16 kHz mono, `sounddevice`
- `src/transcribe.py`: implemented — Whisper STT, Python library + CLI fallback
- `src/summarise.py`: implemented — Ollama SOAP note generator, MBS/PBS/GPMP context
- `src/review.py`: in progress
- `src/pipeline.py`: in progress

## Hard Requirements

- **All audio, transcriptions, and generated notes must be stored locally** — no cloud upload of patient data
- **No PHI to third-party APIs** — AI processing must run locally (Whisper for STT, local LLM for summarisation) or use fully anonymised/synthetic data for testing
- Compliance with the **Australian Privacy Act 1988** and **Health Records Acts** (state-level)
- Use **Australian English** spelling throughout (e.g. "anaesthesia", "haematology", "paediatric")

## Planned Architecture

```
consult/
├── audio/          # Raw audio recordings (never leave this machine)
├── transcripts/    # Raw transcribed text (never leave this machine)
├── notes/          # Generated SOAP notes and summaries
├── templates/      # Document templates (SOAP, referral, MHTP)
└── src/            # Application source code
```

**Processing pipeline:**
1. Audio capture → local STT via OpenAI Whisper (`whisper` CLI or Python library)
2. Transcript → structured note generation via local LLM (or Claude API using only anonymised/synthetic text)
3. Note → review UI for GP to edit and approve before saving

## Development Commands

```bash
# Record a consultation
python3 src/record.py                  # uses default input device
python3 src/record.py --device N       # select device by index
python3 src/record.py --list-devices   # list available input devices

# Transcribe with local Whisper (Python library, with CLI fallback if needed)
python3 src/transcribe.py --audio audio/<file>.wav
python3 src/transcribe.py --audio-dir audio/ --model medium --language en
python3 src/transcribe.py --audio audio/<file>.wav --output-dir transcripts/

# Summarise to SOAP note using local Ollama
python3 src/summarise.py --transcript transcripts/<file>.txt
python3 src/summarise.py --transcript transcripts/<file>.txt --model llama3
python3 src/summarise.py --transcript transcripts/<file>.txt --output-dir notes/ --no-review

# Install dependencies
sudo apt-get install -y libportaudio2  # system dep for sounddevice
pip3 install -r requirements.txt --break-system-packages
```

Install Whisper (requires ffmpeg; used by Python package and CLI workflows):
```bash
sudo apt-get install -y ffmpeg
pip3 install openai-whisper torch --break-system-packages
```

Run Ollama locally for summarisation:
```bash
ollama serve
ollama pull llama3
```

## Medical Documentation Standards

### SOAP Notes
- **S**ubjective — presenting complaint, history, symptoms in patient's own words
- **O**bjective — vital signs, examination findings, investigations ordered/reviewed
- **A**ssessment — diagnosis or differential diagnoses
- **P**lan — management, prescriptions, referrals, follow-up

### MBS Item Numbers (Australia)
Include a suggested MBS item in every generated note with a brief justification (estimated time + complexity markers). The GP makes the final determination.

| Item | Level   | Duration   | Requirements |
|------|---------|------------|--------------|
| 23   | Level B | 6–19 min   | One or more health problems; not a brief/afterthought consult |
| 36   | Level C | 20–39 min  | At least one complex issue; document complexity |
| 44   | Level D | 40+ min    | Long complex consult; document time and complexity in full |

### Other Document Types
- **Health Summary** — active problems, medications, allergies, immunisations
- **Referral letters** — recipient, reason, relevant history, current medications
- **Mental Health Treatment Plans** (MHTP) — item 2710/2712 where applicable

## Australian Clinical Context

- Prescriptions must reference **PBS** item codes where applicable
- Referrals to specialists should note whether a **GP Management Plan** (GPMP, item 721) is in place
- Mental health referrals: note whether a **Mental Health Treatment Plan** exists
- Bulk-billing vs private billing distinction affects MBS documentation requirements
