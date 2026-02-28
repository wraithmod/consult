# GEMINI Research Report
**Date:** 2026-02-28
**Author:** GEMINI research agent
**Scope:** Three research tasks covering Ollama STT, VAD pre-processing, and faster-whisper vs whisper.cpp for this GP consult transcription project.

---

## RESEARCH-1: Ollama Native Audio / Speech-to-Text Support

### Current State (February 2026)

Ollama **does not natively expose a speech-to-text endpoint**. As of early 2026 there is no `/api/audio/transcriptions`, `/v1/audio/transcriptions`, or any analogous endpoint built into the Ollama server. Ollama's design philosophy is focused on running LLM inference (text-in, text-out); audio modality is out of scope for the core project.

#### What the Ollama Hub does provide

The Ollama model hub lists community-contributed Whisper model cards (e.g. `karanchopda333/whisper`, `dimavz/whisper-tiny`). These appear to be model registry stubs. They do **not** cause Ollama to serve an audio API — they cannot be invoked with audio payloads via the standard `POST /api/generate` endpoint. Attempting to use them for audio transcription would require piping base64-encoded audio as a text prompt, which is neither documented nor supported.

#### The standard community architecture

Every production-grade open-source project that combines local Whisper STT with local LLM summarisation (Meetily, Ollama-Transcriber, ollama-voice, OWhisper) keeps Whisper and Ollama as **separate, independent processes**:

```
sounddevice → WAV file → [faster-whisper Python process] → transcript text
                                                              ↓
                                            POST http://localhost:11434/api/generate
                                            {"model": "llama3", "prompt": transcript_text}
                                                              ↓
                                                         SOAP note text
```

The existing `src/transcribe.py` and `src/summarise.py` already implement this exact split correctly.

#### OpenAI-compatible audio shim (OWhisper)

OWhisper (released August 2025, by HyprNote) attempts to provide an "Ollama for real-time STT" experience — i.e., a locally running server that serves `/v1/audio/transcriptions` and manages Whisper models similar to how Ollama manages LLMs. It is early-stage and adds operational complexity (a second daemon to manage) without meaningful benefit over calling `faster-whisper` directly in Python.

#### Roadmap

There is a documented community request for Ollama to support audio transcription natively. No official roadmap date exists. This feature is not expected to land in Ollama in the near term.

### API Format (if Ollama ever supports it — illustrative)

The hypothetical future request format, based on OpenAI's specification, would look like:

```http
POST http://localhost:11434/v1/audio/transcriptions
Content-Type: multipart/form-data

file=<wav_bytes>
model=whisper-medium
language=en
response_format=text
```

This is **not currently supported**. Do not rely on it.

### Quality/Speed Comparison with faster-whisper

| Factor | Ollama-native STT (hypothetical) | faster-whisper (current) |
|---|---|---|
| Availability | Not available | Available now |
| CPU int8 speed | Unknown | ~2-4x faster than openai-whisper |
| Python integration | Would need HTTP client | Native Python API |
| Dependency overhead | Single process | One additional pip package |
| Accuracy control | Unknown | Same Whisper weights, identical WER |

### Recommendation

**Do not integrate Ollama-native transcription.** It does not exist. Continue using `faster-whisper` (already the primary backend in `src/transcribe.py`) as a standalone Python library for STT. Ollama's role in this project remains exactly what it is: LLM text inference for SOAP note generation from the transcript string. The architectural separation is correct and should be maintained.

---

## RESEARCH-2: VAD (Voice Activity Detection) Pre-processing

### Why VAD Matters for This Use Case

A 20–30 minute GP consultation contains significant silence: patient pauses, typing, note-taking, and between-question gaps. Whisper processes audio in 30-second windows; silence-heavy windows still consume full computation time. Stripping silence before transcription can reduce the effective audio duration by 20–40% in a typical consult, proportionally reducing transcription time.

### Options Compared

#### Silero-VAD

- **Project:** `snakers4/silero-vad` (MIT licence)
- **Current version:** 6.2.1 (released 24 February 2026 — ONNX Runtime made optional)
- **Model size:** ~2 MB (v5+), processes 30+ ms audio chunks in under 1 ms per chunk on a single CPU thread
- **Sample rates:** 8000 Hz and 16000 Hz (matches project's 16 kHz recording format exactly)
- **Languages:** trained on 6000+ languages — no English-specific tuning needed
- **Accuracy (benchmark vs WebRTC):** At 5% False Positive Rate, Silero achieves 87.7% True Positive Rate vs WebRTC VAD's 50% TPR — approximately 4x fewer missed speech frames
- **Licence:** MIT — no restrictions for medical software
- **Integration pattern:** accepts PyTorch tensors; numpy arrays convert trivially via `torch.from_numpy()`

**numpy/sounddevice integration snippet:**

```python
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()

def strip_silence(audio_np: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Remove silence from a float32 numpy array using Silero VAD.
    audio_np: float32 array in range [-1.0, 1.0], shape (N,), at 16 kHz
    Returns: concatenated speech segments as float32 numpy array
    """
    wav = torch.from_numpy(audio_np)
    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=sample_rate,
        threshold=0.5,          # speech probability threshold
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        return_seconds=False,   # return sample indices
    )
    if not speech_timestamps:
        return audio_np  # no speech detected — return original

    segments = [audio_np[ts["start"]:ts["end"]] for ts in speech_timestamps]
    return np.concatenate(segments)
```

Usage with sounddevice output (which already produces float32 at 16 kHz):

```python
import sounddevice as sd
import numpy as np

# Record (matches src/record.py output format)
raw_audio = sd.rec(int(duration * 16000), samplerate=16000,
                   channels=1, dtype='float32')
sd.wait()
audio_mono = raw_audio.squeeze()  # shape: (N,)

# Strip silence before passing to faster-whisper
speech_only = strip_silence(audio_mono, sample_rate=16000)

# Pass directly to WhisperModel.transcribe() as numpy array
segments, info = whisper_model.transcribe(speech_only, language="en")
```

Note: `sounddevice` records float32 in [-1.0, 1.0] by default at the specified rate — no int16-to-float32 conversion is required. This is a direct fit with Silero-VAD's expected input format.

#### WebRTC VAD (`py-webrtcvad`)

- Based on Google's legacy WebRTC VAD (Gaussian Mixture Model, 2016)
- Requires int16 PCM at 8/16/32/48 kHz with specific frame sizes (10/20/30 ms)
- TPR at 5% FPR: ~50% (half the accuracy of Silero-VAD)
- Licence: BSD/WebRTC-compatible — suitable for medical use
- Integration: needs int16 conversion from sounddevice's float32 output; fiddlier API
- **Not recommended** — inferior accuracy and more integration friction

#### auditok

- Pure-Python energy-based VAD (amplitude threshold, no ML)
- Works on raw bytes or numpy arrays
- No deep-learning dependency (no PyTorch required)
- Simpler but less accurate on low-volume speech or overlapping background noise (clinic ambient noise)
- Suitable only if PyTorch is unavailable
- MIT licence

#### TEN-VAD (TEN-framework, HuggingFace)

- Very new (2025), limited documentation
- Not yet suitable for production use in this context

### Expected Speedup for a 20–30 Minute Consult

A GP consultation at 20–30 minutes of recorded audio will typically contain:
- 40–60% active speech (combined GP and patient turns)
- 10–20% brief pauses within turns
- 20–30% significant silence (note-taking, thinking, examination)

Stripping silence at `min_silence_duration_ms=500` would reduce effective audio to roughly 15–20 minutes. faster-whisper's CPU processing time scales approximately linearly with audio duration, so expected reduction is 25–40% of transcription wall-clock time.

For a 25-minute consult on a mid-range Intel desktop (i5/i7), this could reduce transcription from roughly 8–12 minutes to 5–8 minutes with the `medium` model.

### Licensing Note

Silero-VAD (MIT), auditok (MIT), and webrtcvad (BSD/WebRTC) all permit free use in medical software without royalty or registration obligations. None perform any network communication. All comply with the Australian Privacy Act 1988 requirement that no PHI leave the machine.

### Recommendation

**Use Silero-VAD v6.x.** It is the clear accuracy leader (87.7% TPR vs 50% for WebRTC), is MIT-licensed, integrates cleanly with the project's existing float32/16 kHz numpy arrays, and is a dependency that faster-whisper itself already uses internally (faster-whisper's `vad.py` module wraps Silero-VAD). The PyTorch dependency is already present (required by openai-whisper and faster-whisper), so there is no additional installation cost.

VAD should be applied as a pre-processing step in `src/transcribe.py` before calling `WhisperModel.transcribe()`, using sample-index timestamps to reassemble only the speech frames. The `strip_silence()` helper above can be dropped directly into the existing codebase.

---

## RESEARCH-3: faster-whisper vs whisper.cpp — CPU-Only Comparison

### Context

The project currently has `faster-whisper` as its primary STT backend (already implemented in `src/transcribe.py`). This research validates that choice and provides benchmark context for the GP hardware assumption (standard Intel desktop, no GPU).

### Speed Comparison (CPU Only)

| Model | Backend | CPU Benchmark | Notes |
|---|---|---|---|
| small | faster-whisper (int8) | 1m 42s on i7-12700K (8 threads) | From faster-whisper README |
| small.en | faster-whisper | 14s | Comparative benchmark |
| small.en | whisper.cpp | 46s | Same hardware |
| medium | faster-whisper (int8) | ~3–5 min for 25 min audio* | Extrapolated from small benchmarks |
| medium | whisper.cpp (GGML q5) | ~6–10 min for 25 min audio* | Extrapolated |
| medium | openai-whisper (fp32) | ~10–15 min for 25 min audio* | Baseline reference |

*Extrapolated estimates — no directly published 25-minute audio benchmark for the exact medium model on a typical GP desktop.

Key finding: **faster-whisper is approximately 2–5x faster than whisper.cpp on CPU** in published benchmarks. The gap is most pronounced when faster-whisper uses int8 quantisation with Intel MKL/AVX512 acceleration.

### Memory Footprint

| Backend | Medium Model | Notes |
|---|---|---|
| openai-whisper (fp32) | ~2.5 GB RAM | PyTorch tensors, full precision |
| faster-whisper (int8) | ~800 MB – 1.2 GB RAM | CTranslate2 int8; significant reduction |
| whisper.cpp (GGML q5_0) | ~2.1 GB RAM | GGML format; stays flat during inference |

faster-whisper with int8 quantisation offers the smallest memory footprint — important if the GP desktop runs other applications concurrently. whisper.cpp's memory usage is stable (doesn't grow with audio length), but starts higher than faster-whisper int8.

### Accuracy (WER)

Both backends use the identical Whisper model weights. With the same decoding parameters:
- WER differences are negligible (< 0.5% relative difference in controlled tests)
- faster-whisper uses float32 intermediate computations in attention layers even in int8 mode, preserving numerical precision where it matters
- For Australian English medical speech (accented vocabulary, clinical terms), model choice (medium vs large) matters far more than the backend choice

Whisper medium is well-suited to Australian English — it was trained on a diverse multilingual corpus including Australian English content. Clinical and pharmaceutical vocabulary (PBS drug names, anatomical terms) may have slightly higher WER in all Whisper variants; the large-v3 model shows measurable improvement here if CPU resources permit.

### Python Integration Ease

#### faster-whisper

- Pure Python API via `pip install faster-whisper`
- Direct numpy float32 array input: `model.transcribe(audio_np_float32, language="en")`
- Returns a generator of segment objects with `.start`, `.end`, `.text` attributes
- VAD parameter passthrough: `model.transcribe(..., vad_filter=True)` (uses built-in Silero-VAD)
- Already integrated in `src/transcribe.py` — zero additional integration work

#### whisper.cpp

- C/C++ binary — must be compiled or installed separately
- Python bindings available (`pywhispercpp` 1.4.1, December 2025; `whispercpp`; `whisper-cpp-python`) — all are third-party, not from the whisper.cpp authors
- No binding has the same API stability guarantee as faster-whisper
- The existing codebase already has a CLI subprocess fallback pattern in `src/transcribe.py`; whisper.cpp would slot into that same subprocess approach if the Python bindings are avoided
- Subprocess approach requires the compiled `whisper-cpp` binary to be present on the GP's machine — a deployment/packaging burden

### Comparison Table

| Criterion | faster-whisper | whisper.cpp |
|---|---|---|
| CPU speed (medium model) | Best — 2–5x vs whisper.cpp | Good — faster than openai-whisper |
| RAM (medium, int8/q5) | ~800 MB – 1.2 GB | ~2.1 GB |
| Python API quality | First-class, stable | Third-party bindings, variable quality |
| numpy array input | Native float32 support | Requires file or subprocess |
| Deployment complexity | `pip install faster-whisper` | Compile or ship binary + pip install binding |
| VAD built-in | Yes (Silero-VAD via `vad_filter=True`) | No |
| Diarisation integration | Works (segment timestamps available) | Limited via subprocess |
| Accuracy vs openai-whisper | Identical (same weights) | Identical (same weights) |
| Licence | MIT | MIT |
| Maintenance status | Active (SYSTRAN, 2025–2026) | Active (ggml-org, 2025–2026) |

### Recommendation

**Stay with faster-whisper. It is already the correct choice and is already implemented.**

The combination of:
- Best CPU speed on int8 quantisation (2–5x faster than whisper.cpp)
- Lowest memory footprint (~1 GB for medium int8)
- First-class Python API with native numpy float32 input
- Built-in Silero-VAD passthrough (`vad_filter=True`)
- Existing integration in `src/transcribe.py`

...makes faster-whisper unambiguously the right backend for this project. whisper.cpp offers no advantage in this context: it is slower on CPU, uses more memory, has no stable Python API, and would add deployment complexity.

If the GP machine has a GPU in future, faster-whisper's `compute_type="float16"` on CUDA would yield further dramatic speedups without any code change beyond the `--device cuda` flag that already exists.

---

## Overall: Next Iteration Architecture

Based on all three research findings, the recommended architecture for the next iteration is an evolution of the current implementation — not a replacement. No major structural changes are needed.

### Recommended Stack

```
sounddevice (16 kHz, float32 mono)
    │
    ▼
Silero-VAD 6.x (MIT)
    silence stripping, speech timestamp extraction
    expected: 25–40% reduction in audio fed to Whisper
    │
    ▼
faster-whisper (CTranslate2, int8, medium model)
    CPU-only, ~800 MB–1.2 GB RAM
    Australian English, segment timestamps for diarisation
    │
    ▼
pyannote.audio (optional, --diarise flag)
    speaker diarisation using segment timestamps
    │
    ▼
plain text transcript (saved to transcripts/)
    │
    ▼
Ollama local LLM (llama3 or similar)
    POST http://localhost:11434/api/generate
    SOAP note generation with MBS/PBS/GPMP context
    │
    ▼
curses review UI (src/review.py)
    GP approves, edits, saves to notes/
```

### What Changes in the Next Iteration

1. **Add Silero-VAD pre-processing step in `src/transcribe.py`**
   - Strip silence before passing audio to `WhisperModel.transcribe()`
   - OR use faster-whisper's built-in `vad_filter=True` parameter (wraps Silero internally) — this is the zero-friction path
   - Reduces transcription time by 25–40% for typical consults

2. **Confirm faster-whisper `vad_filter=True` as default**
   - This activates Silero-VAD inside faster-whisper transparently
   - No separate Silero-VAD installation is then required
   - Adjust `vad_parameters` dict for clinic noise (e.g. `min_silence_duration_ms=600`)

3. **No changes to Ollama integration**
   - Ollama remains the LLM backend for summarisation at `http://localhost:11434/api/generate`
   - No audio endpoints available or needed
   - `src/summarise.py` is correct as-is

4. **No changes to backend selection logic**
   - faster-whisper → openai-whisper → whisper CLI fallback chain in `src/transcribe.py` is correct
   - whisper.cpp is not recommended and should not be added

### Dependency Summary for Next Iteration

| Package | Version | Purpose | Already present? |
|---|---|---|---|
| `faster-whisper` | >=1.0 | STT with CTranslate2 | Yes (primary backend) |
| `silero-vad` | >=6.2 | VAD (or use vad_filter=True in faster-whisper) | No — optional |
| `torch` | >=2.0 | Runtime for Whisper and Silero-VAD | Yes (openai-whisper dep) |
| `sounddevice` | any | Audio capture | Yes |
| `numpy` | any | Audio array handling | Yes |
| `pyannote.audio` | >=3.1 | Speaker diarisation (optional) | Optional (already flagged) |

The simplest path for VAD in the next iteration is to add `vad_filter=True` to the `WhisperModel.transcribe()` call in `_transcribe_with_faster_whisper()` — this requires no new package installations since faster-whisper bundles its own Silero-VAD weights.

### Privacy Compliance Note

All components in this architecture run entirely on the GP's local machine. No audio, transcript, or generated note is transmitted to any external service. This complies with the Australian Privacy Act 1988 and state-level Health Records Acts. Silero-VAD, faster-whisper, and Ollama all operate fully offline after their model files are downloaded once.

---

*Sources consulted:*
- [GitHub — SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [GitHub — snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- [GitHub — ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [Picovoice — Best VAD in 2026](https://picovoice.ai/blog/best-voice-activity-detection-vad/)
- [Modal — Choosing Whisper Variants](https://modal.com/blog/choosing-whisper-variants)
- [Alibaba Insights — faster-whisper vs whisper.cpp](https://www.alibaba.com/product-insights/a-practical-guide-to-choosing-between-whisper-cpp-and-faster-whisper-for-offline-transcription.html)
- [faster-whisper — GitHub Issue #1127 comparison](https://github.com/ggml-org/whisper.cpp/issues/1127)
- [silero-vad PyPI](https://pypi.org/project/silero-vad/)
- [PyTorch Hub — Silero VAD](https://pytorch.org/hub/snakers4_silero-vad_vad/)
- [OWhisper / HyprNote — HN discussion](https://news.ycombinator.com/item?id=44901853)
- [Meetily — Local Meeting Notes with Whisper + Ollama](https://dev.to/zackriya/local-meeting-notes-with-whisper-transcription-ollama-summaries-gemma3n-llama-mistral--2i3n)
- [Northflank — Best open source STT in 2026](https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks)
