import re

from ..config import PACE_BAND
from ..events import Word, norm

_EN = re.compile(r"[aeiouy]+")
_ID = re.compile(r"ai|au|oi|ei|[aeiou]")  # Indonesian spelling is phonetic: one vowel (or diphthong) = one syllable


def syllables(word: str, lang: str = "en") -> int:
    w = norm(word)
    if not w:
        return 0
    if lang == "id":
        return max(1, len(_ID.findall(w)))
    n = len(_EN.findall(w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and n > 1:
        n -= 1  # silent final e: "make", "time"
    return max(1, n)


def rate(words: list[Word], start: float, end: float, lang: str = "en") -> tuple[float, float]:
    """Words per minute and syllables per second for words starting in [start, end)."""
    span = end - start
    inside = [w for w in words if start <= w.start < end]
    return len(inside) * 60 / span, sum(syllables(w.text, lang) for w in inside) / span


class Pace:
    """Live pace over a sliding window, with a 'slow down' nudge."""

    def __init__(self, lang: str = "en", window: float = 20.0, band: tuple = PACE_BAND, nudge_after: float = 10.0):
        self.lang, self.window, self.band, self.nudge_after = lang, window, band, nudge_after
        self._fast_since: float | None = None
        self._nudged = False

    def measure(self, words: list[Word], now: float) -> dict:
        if not words or now - words[0].start < 5:
            return {"wpm": None, "sps": None}
        start = max(words[0].start, now - self.window)
        wpm, sps = rate(words, start, now, self.lang)
        return {"wpm": round(wpm), "sps": round(sps, 1)}

    def nudge(self, wpm: float | None, now: float) -> str | None:
        if wpm is None or wpm <= self.band[1]:
            self._fast_since, self._nudged = None, False
            return None
        if self._fast_since is None:
            self._fast_since = now
        if not self._nudged and now - self._fast_since >= self.nudge_after:
            self._nudged = True
            return "Slow down a little"
        return None


def timeline(words: list[Word], duration: float, lang: str = "en", window: float = 20.0, step: float = 5.0) -> list[dict]:
    """Pace over time for the report chart."""
    if not words:
        return []
    points, t = [], words[0].start + min(window, 10.0)
    while t <= duration + step / 2:
        start = max(words[0].start, t - window)
        wpm, sps = rate(words, start, t, lang)
        points.append({"t": round(t, 1), "wpm": round(wpm), "sps": round(sps, 1)})
        t += step
    return points
