import math
import time
from pathlib import Path

import numpy as np
import sherpa_onnx

from ..config import NUM_THREADS, PROVIDER, SAMPLE_RATE
from ..events import Transcript, Word

FRAME = 0.08         # one encoder frame: 10 ms features x 8 subsampling
TAIL_PADDING = 1.5   # seconds; covers the largest (1120 ms) chunk variant


class NemotronEngine:
    """Nemotron Speech Streaming (cache-aware FastConformer RNN-T) via sherpa-onnx.

    Load once; call .stream() per session. Works with any sherpa-onnx streaming
    transducer folder (encoder/decoder/joiner .onnx + tokens.txt).
    """

    def __init__(self, model_dir: Path, num_threads: int = NUM_THREADS, provider: str = PROVIDER):
        d = Path(model_dir)
        if not (d / "tokens.txt").exists():
            raise FileNotFoundError(f"No ASR model in {d}. Run: python scripts/download_models.py")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=str(d / "tokens.txt"),
            encoder=str(_pick(d, "encoder")),
            decoder=str(_pick(d, "decoder")),
            joiner=str(_pick(d, "joiner")),
            num_threads=num_threads,
            provider=provider,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule2_min_trailing_silence=1.0,  # finalise a caption line after 1 s of silence
        )

    def stream(self) -> "NemotronStream":
        return NemotronStream(self.recognizer)


class NemotronStream:
    def __init__(self, recognizer):
        self.r = recognizer
        self.s = recognizer.create_stream()
        self._last = ""

    def accept(self, samples: np.ndarray) -> list[Transcript]:
        self.s.accept_waveform(SAMPLE_RATE, samples)
        return self._decode(final=False)

    def finish(self) -> list[Transcript]:
        # The encoder needs right context to commit the last word; pad with silence first.
        self.s.accept_waveform(SAMPLE_RATE, np.zeros(int(TAIL_PADDING * SAMPLE_RATE), np.float32))
        self.s.input_finished()
        return self._decode(final=True)

    def _decode(self, final: bool) -> list[Transcript]:
        while self.r.is_ready(self.s):
            self.r.decode_stream(self.s)
        res = self.r.get_result_all(self.s)

        if final or self.r.is_endpoint(self.s):
            self.r.reset(self.s)
            self._last = ""
            return [Transcript("final", to_words(res), time.time())] if res.text.strip() else []
        if res.text != self._last:
            self._last = res.text
            return [Transcript("partial", to_words(res), time.time())]
        return []


def to_words(res) -> list[Word]:
    """Merge BPE tokens into words. A leading space marks a new word; timestamps are
    relative to the current segment, so add res.start_time."""
    probs = list(res.ys_probs)
    if len(probs) != len(res.tokens):
        probs = [None] * len(res.tokens)

    words: list[list] = []
    for tok, ts, lp in zip(res.tokens, res.timestamps, probs):
        t = res.start_time + ts
        if tok.startswith(" ") or not words:
            words.append([tok.strip(), t, t + FRAME, [lp]])
        else:
            words[-1][0] += tok
            words[-1][2] = t + FRAME
            words[-1][3].append(lp)
    return [Word(text, round(s, 2), round(e, 2), _conf(lps)) for text, s, e, lps in words if text]


def _conf(logprobs: list) -> float | None:
    if None in logprobs:
        return None
    return round(math.exp(sum(logprobs) / len(logprobs)), 3)


def _pick(d: Path, part: str) -> Path:
    files = sorted(d.glob(f"{part}*.onnx"), key=lambda p: "int8" not in p.name)
    if not files:
        raise FileNotFoundError(f"No {part}*.onnx in {d}")
    return files[0]
