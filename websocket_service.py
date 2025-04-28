from fastapi import WebSocket, WebSocketDisconnect
from auth_middleware import AuthMiddleware
from rag_service import RAGService
import jwt
import os

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret")

class WebSocketService:
    def __init__(self):
        pass

    async def chat(self, websocket: WebSocket):
        await websocket.accept()
        try:
            token = websocket.headers.get("Authorization")
            if not token:
                await websocket.close(code=1008)
                return
            decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            user_id = decoded.get("sub")
            company_id = decoded.get("company_id")
            role = decoded.get("role")
            if not user_id or not company_id or not role:
                await websocket.close(code=1008)
                return

            rag_service = RAGService(company_id)
            async for message in websocket.iter_text():
                async for chunk in rag_service.query(message):
                    await websocket.send_text(chunk)
        except WebSocketDisconnect:
            print("WebSocket disconnected")
        except Exception as e:
            await websocket.close(code=1011)
            print(f"WebSocket error: {e}")