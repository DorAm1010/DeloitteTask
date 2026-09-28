"""Web chat server.

    uvicorn airport_agent.server:app --reload     # then open http://localhost:8000

One AirportAgent (i.e. one conversation history) per browser session id.
Sessions live in memory, which is fine for a single-user demo; production
would persist them (Redis/DB) and add auth.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import AirportAgent
from .metrics import data_vintage

app = FastAPI(title="Airport Investment Intelligence Agent")
WEB_DIR = Path(__file__).parent / "web"
# JS libraries are vendored (not loaded from a CDN) so the UI works offline / behind corporate proxies.
app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")
_sessions: dict[str, AirportAgent] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


def _preview(output: str, limit: int = 1500) -> str:
    """Tool outputs can be long; the UI shows a trimmed, pretty version."""
    try:
        text = json.dumps(json.loads(output), indent=1)
    except ValueError:
        text = output
    return text if len(text) <= limit else text[:limit] + "\n..."


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {"data_vintage": data_vintage()}


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:  # sync def -> FastAPI runs it in a worker thread
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    session_id = req.session_id or str(uuid.uuid4())
    agent = _sessions.setdefault(session_id, AirportAgent())
    reply = agent.ask(req.message)
    return {
        "session_id": session_id,
        "answer": reply.text,
        "stop_reason": reply.stop_reason,
        "tool_calls": [{"name": c.name, "input": c.input, "is_error": c.is_error,
                        "output_preview": _preview(c.output)} for c in reply.tool_calls],
    }


@app.post("/api/reset")
def reset(req: ChatRequest) -> dict:
    if req.session_id in _sessions:
        _sessions[req.session_id].reset()
    return {"ok": True}
