"""Web chat server.

    uvicorn airport_agent.server:app --reload     # then open http://localhost:8000

One AirportAgent (i.e. one conversation history) per browser session id.
The web UI uses /api/chat/stream (server-sent events, so it can show each step
live); /api/chat returns the same answer in one JSON response.
Sessions live in memory, which is fine for a single-user demo; production
would persist them (Redis/DB) and add auth.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import uuid
from collections.abc import Iterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import AgentError, AirportAgent
from .metrics import data_vintage

app = FastAPI(title="Airport Investment Intelligence Agent")
WEB_DIR = Path(__file__).parent / "web"
# JS libraries are vendored (not loaded from a CDN) so the UI works offline / behind corporate proxies.
app.mount("/vendor", StaticFiles(directory=WEB_DIR / "vendor"), name="vendor")
_sessions: dict[str, AirportAgent] = {}
log = logging.getLogger(__name__)


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


def _charts(calls) -> list[dict]:
    """Chart specs from successful show_chart calls (computed by charts.py, not by the model)."""
    return [json.loads(c.output)["chart"] for c in calls if c.name == "show_chart" and not c.is_error]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    return {"data_vintage": data_vintage()}


def _session(req: ChatRequest) -> tuple[str, AirportAgent]:
    if not req.message.strip():
        raise HTTPException(400, "Empty message")
    session_id = req.session_id or str(uuid.uuid4())
    return session_id, _sessions.setdefault(session_id, AirportAgent())


def _payload(session_id: str, reply) -> dict:
    return {
        "session_id": session_id,
        "answer": reply.text,
        "stop_reason": reply.stop_reason,
        "usage": {**reply.usage.to_dict(), "cost_usd": reply.cost_usd, "latency_s": reply.latency_s},
        "charts": _charts(reply.tool_calls),
        "grounding": {"share": reply.grounding, "ungrounded": reply.ungrounded},
        "tool_calls": [{"name": c.name, "input": c.input, "is_error": c.is_error,
                        "output_preview": _preview(c.output)} for c in reply.tool_calls],
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:  # sync def -> FastAPI runs it in a worker thread
    session_id, agent = _session(req)
    try:
        reply = agent.ask(req.message)
    except AgentError as err:  # always JSON, so the UI can show a readable message
        raise HTTPException(err.status, err.message) from err
    return _payload(session_id, reply)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_events(agent: AirportAgent, message: str, session_id: str) -> Iterator[str]:
    """Run the agent in a worker thread and yield its progress as server-sent events.

    Events: "progress" (see AirportAgent.ask), then exactly one "answer" (same payload as /api/chat)
    or "error" ({message, status}). Every event carries the session id, so the browser keeps the
    same conversation even when the first question fails.
    """
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            reply = agent.ask(message, on_progress=lambda e: events.put(("progress", {**e, "session_id": session_id})))
            events.put(("answer", _payload(session_id, reply)))
        except AgentError as err:
            events.put(("error", {"message": err.message, "status": err.status, "session_id": session_id}))
        except Exception:  # noqa: BLE001 - never leave the browser waiting on a dead stream
            log.exception("Unexpected error while answering")
            events.put(("error", {"message": "Unexpected server error; see the server log.", "status": 500,
                                  "session_id": session_id}))
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (item := events.get()) is not None:
        yield _sse(*item)


@app.post("/api/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    session_id, agent = _session(req)
    return StreamingResponse(_stream_events(agent, req.message, session_id), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/reset")
def reset(req: ChatRequest) -> dict:
    if req.session_id in _sessions:
        _sessions[req.session_id].reset()
    return {"ok": True}
