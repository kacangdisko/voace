from collections import Counter

MIN_PAUSE, GOOD_MAX, FREEZE = 0.5, 2.0, 3.0  # seconds


def classify(gap: float) -> str | None:
    if gap < MIN_PAUSE:
        return None
    if gap <= GOOD_MAX:
        return "good"
    return "long" if gap <= FREEZE else "freeze"


class Pauses:
    """Silences between VAD speech segments: 0.5-2 s good, 2-3 s long, >3 s freeze."""

    def __init__(self):
        self.items: list[dict] = []
        self.last_end: float | None = None
        self.last_start: float | None = None
        self._live_freeze = False

    def add(self, segments: list[tuple[float, float]], ongoing_start: float | None = None) -> None:
        """segments: finished speech; ongoing_start: start of speech still in progress.
        A pause is counted as soon as speech resumes, not when that speech ends."""
        for start, end in segments:
            self._speech_started(start)
            self.last_end = end
        if ongoing_start is not None:
            self._speech_started(ongoing_start)

    def _speech_started(self, start: float) -> None:
        if self.last_start is not None and start <= self.last_start + 0.01:
            return  # already counted while it was ongoing
        if self.last_end is not None and (kind := classify(start - self.last_end)):
            self.items.append({"start": round(self.last_end, 2), "end": round(start, 2), "kind": kind})
        self.last_start = start
        self._live_freeze = False

    def freeze_now(self, now: float, speaking: bool) -> bool:
        """True once per silence, as soon as it passes the freeze threshold."""
        if speaking or self.last_end is None or self._live_freeze or now - self.last_end <= FREEZE:
            return False
        self._live_freeze = True
        return True

    def counts(self) -> dict:
        c = Counter(p["kind"] for p in self.items)
        return {k: c.get(k, 0) for k in ("good", "long", "freeze")}
