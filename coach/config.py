import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"

SAMPLE_RATE = 16000
ASR_MODEL_DIR = Path(os.getenv("COACH_ASR_MODEL", MODELS / "nemotron-en-560ms"))
VAD_MODEL = Path(os.getenv("COACH_VAD_MODEL", MODELS / "silero_vad.onnx"))
DB_PATH = Path(os.getenv("COACH_DB", ROOT / "data" / "sessions.db"))

NUM_THREADS = int(os.getenv("COACH_THREADS", 0)) or max(1, (os.cpu_count() or 2) // 2)
PROVIDER = os.getenv("COACH_PROVIDER", "cpu")  # "coreml" on Mac is optional

PACE_BAND = (130, 160)  # target words per minute
