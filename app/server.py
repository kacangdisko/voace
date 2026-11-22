import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from coach import store
from coach.analyzers.fillers import DEFAULT_FILLERS
from coach.config import ASR_MODEL_DIR, PACE_BAND
from coach.engines.nemotron import NemotronEngine
from coach.report import to_markdown
from coach.session import Session
from coach.vad import VAD

engine: NemotronEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = NemotronEngine(ASR_MODEL_DIR)  # load once, ~1-2 s
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/api/defaults")
def defaults():
    return {"fillers": DEFAULT_FILLERS, "pace_band": PACE_BAND}


@app.get("/api/sessions/{session_id}.md", response_class=PlainTextResponse)
def report_markdown(session_id: int):
    report = store.load(session_id)
    if report is None:
        raise HTTPException(404)
    return PlainTextResponse(to_markdown(report), headers={
        "Content-Disposition": f'attachment; filename="rehearsal-{session_id}.md"'})


@app.websocket("/ws")
async def rehearse(ws: WebSocket):
    """Protocol: JSON {"type":"start", lang, fillers} -> binary int16 16 kHz chunks -> JSON {"type":"stop"}."""
    await ws.accept()
    start = await ws.receive_json()
    session = Session(engine.stream(), VAD(), start.get("lang", "en"), start.get("fillers"))
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                return
            if msg.get("bytes") is not None:
                arrived = time.time()
                samples = np.frombuffer(msg["bytes"], dtype="<i2").astype(np.float32) / 32768
                out = await asyncio.to_thread(session.feed, samples, arrived)
            else:  # {"type": "stop"}
                out = await asyncio.to_thread(session.finish)
                out[-1]["id"] = store.save(out[-1])
            for m in out:
                await ws.send_json(m)
            if out and out[-1]["type"] == "report":
                await ws.close()
                return
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
