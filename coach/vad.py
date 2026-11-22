from pathlib import Path

import numpy as np
import sherpa_onnx

from .config import SAMPLE_RATE, VAD_MODEL


class VAD:
    """Silero VAD. Returns finished speech segments as (start, end) seconds."""

    def __init__(self, model: Path = VAD_MODEL, min_silence: float = 0.3):
        if not Path(model).exists():
            raise FileNotFoundError(f"No VAD model at {model}. Run: python scripts/download_models.py")
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = 0.2
        cfg.sample_rate = SAMPLE_RATE
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    @property
    def speaking(self) -> bool:
        return self.vad.is_speech_detected()

    @property
    def ongoing_start(self) -> float | None:
        """Start of the speech segment in progress (finished segments arrive later)."""
        return self.vad.current_segment.start / SAMPLE_RATE if self.speaking else None

    def accept(self, samples: np.ndarray) -> list[tuple[float, float]]:
        self.vad.accept_waveform(samples)
        return self._drain()

    def finish(self) -> list[tuple[float, float]]:
        self.vad.flush()
        return self._drain()

    def _drain(self) -> list[tuple[float, float]]:
        segments = []
        while not self.vad.empty():
            seg = self.vad.front
            segments.append((seg.start / SAMPLE_RATE, (seg.start + len(seg.samples)) / SAMPLE_RATE))
            self.vad.pop()
        return segments
