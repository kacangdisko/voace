"""Replay harness: stream WAV files through the exact live pipeline and log latency.

    python -m eval.replay data/raw/*.wav --out results/baseline --realtime

--realtime feeds 100 ms chunks at real speed; latency numbers are only valid with it.
Without it, audio is fed as fast as possible (quick WER runs; RTF still valid).
Writes <out>/hyps.txt (for eval.wer) and <out>/latency.json.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from coach.config import ASR_MODEL_DIR, SAMPLE_RATE
from coach.engines.nemotron import NemotronEngine
from coach.events import norm
from coach.session import Session
from coach.vad import VAD


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != SAMPLE_RATE:
        t = np.arange(int(len(audio) * SAMPLE_RATE / sr)) / SAMPLE_RATE
        audio = np.interp(t, np.arange(len(audio)) / sr, audio).astype(np.float32)
    return audio


def replay(engine: NemotronEngine, path: Path, realtime: bool, chunk_s: float = 0.1) -> dict:
    """Latency definitions (all against the audio clock, t0 = stream start):
    word delay  = first time a word is shown - (t0 + word end)
    final delay = caption line finalised   - (t0 + its last word end)
    lag         = message ready            - chunk arrival (processing backlog)
    """
    audio = load_audio(path)
    session = Session(engine.stream(), VAD())
    n = int(SAMPLE_RATE * chunk_s)
    first_seen, final_delays, lags, report = {}, [], [], None
    t0 = time.time()

    def collect(msgs, natural_end=True):
        nonlocal report
        for m in msgs:
            if m["type"] == "transcript":
                for w in m["words"]:
                    first_seen.setdefault((w["start"], norm(w["text"])), m["emitted_at"])
                if m["kind"] == "final" and natural_end:
                    final_delays.append(m["emitted_at"] - (t0 + m["words"][-1]["end"]))
            elif m["type"] == "metrics":
                lags.append(m["lag_ms"])
            elif m["type"] == "report":
                report = m

    for i in range(0, len(audio), n):
        chunk = audio[i:i + n]
        due = t0 + (i + len(chunk)) / SAMPLE_RATE  # when the chunk's last sample was "spoken"
        if realtime:
            time.sleep(max(0.0, due - time.time()))
        collect(session.feed(chunk, arrived_at=due if realtime else time.time()))
    collect(session.finish(), natural_end=False)  # end of file forces the last line

    result = {"file": path.name, "duration": report["duration"], "rtf": report["rtf"],
              "hyp": " ".join(w["text"] for w in report["transcript"])}
    if realtime:
        delays = [first_seen[(w["start"], norm(w["text"]))] - (t0 + w["end"])
                  for w in report["transcript"] if (w["start"], norm(w["text"])) in first_seen]
        result |= {"word_delay_ms": _stats(delays), "final_delay_ms": _stats(final_delays),
                   "lag_ms": _stats([x / 1000 for x in lags])}
    return result


def _stats(xs: list[float]) -> dict | None:
    if not xs:
        return None
    xs = sorted(xs)
    return {"mean": round(statistics.mean(xs) * 1000), "p50": round(xs[len(xs) // 2] * 1000),
            "p90": round(xs[int(len(xs) * 0.9)] * 1000), "n": len(xs)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wavs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--model", type=Path, default=ASR_MODEL_DIR)
    a = ap.parse_args()

    engine = NemotronEngine(a.model)
    a.out.mkdir(parents=True, exist_ok=True)
    results = []
    for wav in a.wavs:
        r = replay(engine, wav, a.realtime)
        results.append(r)
        extra = f" | word delay p50 {r['word_delay_ms']['p50']} ms" if r.get("word_delay_ms") else ""
        print(f"{wav.name}: {r['duration']}s, RTF {r['rtf']}{extra}\n  {r['hyp']}")

    (a.out / "hyps.txt").write_text("".join(f"{r['file']}\t{r['hyp']}\n" for r in results), encoding="utf-8")
    (a.out / "latency.json").write_text(json.dumps({"model": str(a.model), "realtime": a.realtime,
                                                    "files": results}, indent=2))
    print(f"Wrote {a.out}/hyps.txt and latency.json")
