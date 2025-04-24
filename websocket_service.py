from fastapi import WebSocket, WebSocketDisconnect
import secrets

# In-memory key store for simplicity (could use Redis or DB)
VALID_KEYS = {}

class WebSocketService:
    def __init__(self):
        pass

    def register_key(self, user_id):
        key = secrets.token_urlsafe(32)
        VALID_KEYS[key] = user_id
        return key

    def validate_key(self, key):
        return key in VALID_KEYS

    async def chat(self, websocket: WebSocket):
        await websocket.accept()
        try:
            data = await websocket.receive_text()
            if not self.validate_key(data):
                await websocket.close(code=1008)
                return
            await websocket.send_text("Connected to GPU chat service!")
            # More logic goes here
        except WebSocketDisconnect:
            print("WebSocket disconnected")