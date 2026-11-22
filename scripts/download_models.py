"""Download the ASR and VAD models into models/ (from sherpa-onnx GitHub releases).

    python scripts/download_models.py              # Nemotron, 560 ms chunks (default)
    python scripts/download_models.py --chunk 160 1120   # extra variants for experiment E6

Use a variant with: COACH_ASR_MODEL=models/nemotron-en-160ms
"""
import argparse
import tarfile
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "models"
BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
NEMOTRON = "sherpa-onnx-nemotron-speech-streaming-en-0.6b-{ms}ms-int8-2026-04-25"
CHUNKS = [80, 160, 560, 1120]


def fetch(url: str, dest: Path) -> None:
    def progress(blocks, block_size, total):
        print(f"\r  {dest.name}: {min(blocks * block_size, total) / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
    urllib.request.urlretrieve(url, dest, progress)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, nargs="+", default=[560], choices=CHUNKS)
    chunks = ap.parse_args().chunk
    MODELS.mkdir(exist_ok=True)

    vad = MODELS / "silero_vad.onnx"
    if not vad.exists():
        fetch(BASE + "silero_vad.onnx", vad)

    for ms in chunks:
        target = MODELS / f"nemotron-en-{ms}ms"
        if target.exists():
            print(f"  {target.name}: already there")
            continue
        name = NEMOTRON.format(ms=ms)
        archive = MODELS / f"{name}.tar.bz2"
        fetch(f"{BASE}{name}.tar.bz2", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(MODELS, filter="data")
        (MODELS / name).rename(target)
        archive.unlink()
    print(f"Models ready in {MODELS}")


if __name__ == "__main__":
    main()
