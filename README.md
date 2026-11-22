# Voace — Live Presentation Coach

Real-time, fully local speech recognition that coaches students while they rehearse a talk:
live captions, pace, filler words, pauses, timer, and a report after each run.

## Setup

```bash
# macOS (Apple Silicon)                      # Windows (PowerShell)
python3.12 -m venv .venv                     # py -3.12 -m venv .venv
source .venv/bin/activate                    # .\.venv\Scripts\Activate.ps1
pip install -r requirements-mac.txt          # pip install -r requirements-windows.txt

python scripts/download_models.py            # Nemotron 560 ms (~460 MB) + Silero VAD
```

macOS with the python.org installer: if the download fails with `CERTIFICATE_VERIFY_FAILED`,
run `/Applications/Python 3.12/Install Certificates.command` once.

## Run

```bash
uvicorn app.server:app --port 8000
```

Open http://localhost:8000 in Chrome, Edge or Safari (Firefox cannot resample the mic to 16 kHz).

## Evaluate

```bash
# Accuracy: fast replay, then score against references ("<id> <text>" per line)
python -m eval.replay data/raw/*.wav --out results/baseline
python -m eval.wer data/refs.txt results/baseline/hyps.txt              # fillers kept
python -m eval.wer data/refs.txt results/baseline/hyps.txt --drop-fillers --show

# Latency: same pipeline, fed at real speed
python -m eval.replay data/raw/*.wav --out results/baseline-rt --realtime

# E6 chunk size: download a variant, point the app at it
python scripts/download_models.py --chunk 160 1120
COACH_ASR_MODEL=models/nemotron-en-160ms python -m eval.replay ... --realtime
```

Unit tests: `pytest`

## Layout

```
app/        FastAPI server (WebSocket /ws) + browser UI (AudioWorklet mic capture)
coach/      events.py (shared format) · engines/ · vad.py · analyzers/ · session.py · report.py · store.py
eval/       replay.py (replay harness) · wer.py (WER/CER)
scripts/    download_models.py
```

## Critical design notes

**One event format for every engine.** Engines emit `Transcript(kind="partial"|"final", words=[Word(text, start, end, conf)])`.
Partials may still change; a final is fixed. Analysers and evaluation only see this format,
so the Whisper engine for Bahasa mode plugs in without touching them.

**Partial vs final.** Nemotron emits a partial whenever its hypothesis changes and a final
after 1 s of trailing silence (sherpa-onnx endpointing). On stop, 1.5 s of silence is fed
before flushing — without it the last word is cut off.

**The VAD does not gate Nemotron.** The cache-aware model needs the continuous stream for
correct word timestamps and endpointing, and it does not hallucinate on silence. Silero VAD
runs alongside it for pause detection. Gating silence away from the model is for the Whisper
engine, which does invent text during silence.

**Pauses** are gaps between VAD speech segments: 0.5–2 s good, 2–3 s long, over 3 s freeze.
A pause is counted the moment speech resumes; a freeze alert fires live after 3 s.

**Latency definitions** (`eval/replay.py`, measured against the audio clock):

| Metric | Meaning |
| --- | --- |
| word delay | first time a word is shown − when it was spoken (word end) |
| final delay | caption line finalised − end of its last word |
| lag | message ready − audio chunk arrival (processing backlog) |
| RTF | compute time / audio duration |

Reference run (Linux VM, 4 CPU threads, 560 ms model, sherpa-onnx test clips): WER 0.0% on
77 words, RTF 0.08 (fast) to 0.13 (real time), word delay p50 ≈ 590 ms, lag p50 ≈ 11 ms.

## Configuration

| Variable | Default |
| --- | --- |
| `COACH_ASR_MODEL` | `models/nemotron-en-560ms` |
| `COACH_VAD_MODEL` | `models/silero_vad.onnx` |
| `COACH_THREADS` | half the CPU cores |
| `COACH_PROVIDER` | `cpu` (`coreml` optional on Mac) |
| `COACH_DB` | `data/sessions.db` |

## Not built yet

Bahasa campur mode (Whisper engine), script follower, hotwords, clarity heatmap, progress
dashboard. Known limit: ASR models tend to drop "um"/"uh" — measured in experiment E5.
