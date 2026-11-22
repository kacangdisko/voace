import time
from dataclasses import asdict

import numpy as np

from .analyzers import fillers, pace, pauses
from .config import SAMPLE_RATE
from .engines.base import ASRStream
from .events import Transcript, Word
from .report import build_report
from .vad import VAD


class Session:
    """One rehearsal: audio in, JSON-ready messages out (transcripts, metrics, nudges, report)."""

    def __init__(self, stream: ASRStream, vad: VAD, lang: str = "en", filler_list: list[str] | None = None):
        self.stream, self.vad, self.lang = stream, vad, lang
        self.fillers = filler_list or fillers.DEFAULT_FILLERS[lang]
        self.pace = pace.Pace(lang)
        self.pauses = pauses.Pauses()
        self.finals: list[Word] = []
        self.partial: list[Word] = []
        self.audio_t = 0.0     # seconds of audio received
        self.compute_s = 0.0   # seconds spent in ASR + VAD

    @property
    def words(self) -> list[Word]:
        return self.finals + self.partial

    def feed(self, samples: np.ndarray, arrived_at: float | None = None) -> list[dict]:
        arrived_at = arrived_at or time.time()
        t0 = time.perf_counter()
        events = self.stream.accept(samples)
        segments = self.vad.accept(samples)
        self.compute_s += time.perf_counter() - t0
        self.audio_t += len(samples) / SAMPLE_RATE
        return self._messages(events, segments, arrived_at)

    def finish(self) -> list[dict]:
        msgs = self._messages(self.stream.finish(), self.vad.finish(), time.time())
        msgs.append({"type": "report", **build_report(self)})
        return msgs

    def _messages(self, events: list[Transcript], segments: list, arrived_at: float) -> list[dict]:
        msgs = []
        for ev in events:
            if ev.kind == "final":
                self.finals += ev.words
                self.partial = []
            else:
                self.partial = ev.words
            msgs.append({"type": "transcript", "kind": ev.kind, "emitted_at": ev.emitted_at,
                         "words": [asdict(w) for w in ev.words]})

        self.pauses.add(segments, self.vad.ongoing_start)
        now, words = self.audio_t, self.words
        p = self.pace.measure(words, now)
        if nudge := self.pace.nudge(p["wpm"], now):
            msgs.append({"type": "nudge", "message": nudge})
        if self.pauses.freeze_now(now, self.vad.speaking):
            msgs.append({"type": "nudge", "message": "Long pause - pick up where you left off"})

        msgs.append({
            "type": "metrics",
            "elapsed": round(now, 2),
            **p,
            "fillers": fillers.summary(words, self.fillers, now / 60),
            "pauses": self.pauses.counts(),
            "speaking": self.vad.speaking,
            "lag_ms": round((time.time() - arrived_at) * 1000),   # chunk arrival -> message ready
            "rtf": round(self.compute_s / now, 3) if now else None,
        })
        return msgs
