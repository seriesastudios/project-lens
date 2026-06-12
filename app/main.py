import json
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.engine import embeddings, views
from app.engine.brain import process_user_input
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


class ChatResponse(BaseModel):
    reply: str


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Routes user text through the brain, then pushes the new lens state to all clients."""
    reply_text = await process_user_input(request.message)
    await manager.broadcast_state()
    return ChatResponse(reply=reply_text)


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
    project_id: int | None = None


@app.post("/api/view")
async def set_view(request: ViewRequest):
    """Deterministic click navigation — entering a project or going back never
    needs an LLM round-trip."""
    if request.mode == "project":
        if request.project_id is None:
            raise HTTPException(status_code=422, detail="mode 'project' requires project_id")
        node = models.get_node(request.project_id)
        if not node or node.get("node_type") != "project" or node["status"] != "active":
            raise HTTPException(status_code=404, detail=f"No active project {request.project_id}")
        views.set_view({"mode": "project", "project_id": request.project_id})
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
