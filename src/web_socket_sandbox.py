

from typing import Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn


from src.LLM import create_llm
from src.config import Config

# WebSocket manager
class WebSocketManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        print(f"Session {session_id} connected.")

    def disconnect(self, session_id: str):
        self.active_connections.pop(session_id, None)
        print(f"Session {session_id} disconnected.")

    async def send_message(self, session_id: str, message: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_text(message)
            

class UserPromptData(BaseModel):
    prompt: str
    conversation_id: str

class UserPromptRequest(BaseModel):
    event: str
    data: UserPromptData

# Generic event request
class BaseEventRequest(BaseModel):
    event: str

ws_manager = WebSocketManager()

app = FastAPI()


        
        
active_websockets = {}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await ws_manager.connect(websocket, session_id)
    global active_websockets
    active_websockets[session_id] = websocket

    try:
        while True:
            data = await websocket.receive_json()
            event_type = data.get("event")
            print(f"Received event: {event_type}")
            ### User prompt
            if event_type == "user_prompt":
                req = UserPromptRequest(**data) # Unpacks message into UserPromptRequest
                model = "gemeni"
                language_model = create_llm(model)
                ai_response = await language_model.generate(req.data.prompt)

    except WebSocketDisconnect:
        ws_manager.disconnect(session_id)

#
# Server Startup
#
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)