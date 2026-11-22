from typing import Protocol

import numpy as np

from ..events import Transcript


class ASRStream(Protocol):
    """One live audio stream. Engines differ inside; the events they emit do not."""

    def accept(self, samples: np.ndarray) -> list[Transcript]:
        """Feed float32 mono 16 kHz audio; return any new partial/final transcripts."""

    def finish(self) -> list[Transcript]:
        """Flush remaining audio at the end of a session."""
