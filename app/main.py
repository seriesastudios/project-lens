import asyncio
import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.engine import embeddings, views
from app.engine.brain import process_user_input_events
from app.database import models


@asynccontextmanager
async def lifespan(_app: FastAPI):
    models.init_db()
    # Index any nodes that don't have embeddings yet (no-op if server is down)
    indexed = embeddings.backfill()
    if indexed:
        print(f"Embeddings: indexed {indexed} nodes at startup")
    yield


app = FastAPI(title="Project Lens Runtime",
              description="Local Task Graph Management System",
              lifespan=lifespan)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        await self._send_state(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    @staticmethod
    def _state_payload() -> str:
        state = views.compute_view_cards()
        return json.dumps({"type": "STATE_UPDATE", "data": state["cards"], "view": state["view"]})

    async def _send_state(self, websocket: WebSocket):
        await websocket.send_text(self._state_payload())

    async def broadcast_state(self):
        payload = self._state_payload()
        dead = []
        for connection in self.active_connections:
            try:
                await connection.send_text(payload)
            except Exception:
                dead.append(connection)
        for connection in dead:
            self.disconnect(connection)


manager = ConnectionManager()


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Routes user text through the brain, streaming protocol events back as
    NDJSON. The turn runs in its own task so a client disconnect mid-stream
    can't abort half-finished database work; 'lens' events become WebSocket
    state pushes (cards update while the reply is still streaming)."""
    queue: asyncio.Queue = asyncio.Queue()

    async def run_turn():
        try:
            async for event in process_user_input_events(request.message):
                if event["type"] == "lens":
                    await manager.broadcast_state()
                else:
                    await queue.put(event)
        except Exception as exc:
            await queue.put({"type": "done", "reply": f"Something went wrong: {exc}"})
        finally:
            await manager.broadcast_state()  # covers view fallbacks and stragglers
            await queue.put(None)

    task = asyncio.create_task(run_turn())

    async def event_stream():
        while True:
            event = await queue.get()
            if event is None:
                break
            yield json.dumps(event) + "\n"
        await task

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


@app.post("/api/nodes/{node_id}/complete")
async def complete_node(node_id: int):
    """Deterministic completion for the card checkmark — no LLM round-trip."""
    completed = models.complete_nodes([node_id])
    if not completed:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
    await manager.broadcast_state()
    return {"success": True, "node_id": node_id}


@app.get("/api/lens")
async def get_lens():
    state = views.compute_view_cards()
    return {"data": state["cards"], "view": state["view"]}


class ViewRequest(BaseModel):
    mode: str
    path: list[int] | None = None        # for mode "node": breadcrumb trail of container ids
    project_id: int | None = None        # legacy: entering a single project


@app.post("/api/view")
async def set_view(request: ViewRequest):
    """Deterministic click navigation — entering a container, drilling into a
    subtask, or climbing the breadcrumb never needs an LLM round-trip."""
    if request.mode in ("node", "project"):
        # Accept either the new path form or the legacy single project_id.
        path = request.path if request.path else (
            [request.project_id] if request.project_id is not None else [])
        if not path:
            raise HTTPException(status_code=422, detail="mode 'node' requires a non-empty path")
        for nid in path:
            node = models.get_node(nid)
            if not node or node["status"] != "active":
                raise HTTPException(status_code=404, detail=f"No active node {nid} in path")
        views.set_view({"mode": "node", "path": path})
    elif request.mode in ("today", "projects", "loose"):
        views.set_view({"mode": request.mode})
    else:
        raise HTTPException(status_code=422, detail=f"Unknown mode {request.mode!r}")
    await manager.broadcast_state()
    return {"success": True}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Clients don't send over WS; this just detects disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
