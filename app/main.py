import json
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from app.engine.brain import process_user_input
from app.database.models import get_active_nodes, get_edges_for_node

app = FastAPI(title="Project Lens Runtime", description="Local Task Graph Management System")

# Set up CORS for frontend decoupling
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open for local HTML files
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send initial state upon connection
        await self.broadcast_state()

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast_state(self):
        nodes = get_active_nodes()
        payload = {"type": "STATE_UPDATE", "data": nodes}
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(payload))
            except Exception:
                pass

manager = ConnectionManager()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Takes user text, processes it through the SLM, and returns the response."""
    # Process through the brain (this blocks, but for a local engine it's usually acceptable; 
    # could be run in a thread pool for true async)
    reply_text = process_user_input(request.message)
    
    # Broadcast the new state to all connected frontends
    await manager.broadcast_state()
    
    return ChatResponse(reply=reply_text)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We don't expect messages from the client over WS, just listen for disconnects
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def serve_index():
    return FileResponse("index.html")

if __name__ == "__main__":
    import uvicorn
    # Optional: run standard Uvicorn setup if file is executed directly
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
