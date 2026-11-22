"""Shared event format: every ASR engine emits these, every analyser consumes them."""
import re
from dataclasses import dataclass

_PUNCT = re.compile(r"[^\w\s'-]")


def norm(text: str) -> str:
    """Lowercase and strip punctuation, keeping apostrophes and inner hyphens."""
    return _PUNCT.sub("", text.lower()).strip("-' ")


@dataclass
class Word:
    text: str                  # as displayed (may carry punctuation)
    start: float               # seconds since session start, audio clock
    end: float
    conf: float | None = None  # 0..1, from token probabilities


@dataclass
class Transcript:
    kind: str                  # "partial" (may still change) | "final" (fixed)
    words: list[Word]
    emitted_at: float          # wall clock, time.time()

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)
