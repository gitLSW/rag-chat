import json
from fastapi import WebSocket, WebSocketDisconnect

class WebsocketService:
    """
    Handles WebSocket connections for streaming RAG responses token-by-token.
    """
    def __init__(self, company_mw):
        self.company_mw = company_mw

    async def chat(self, websocket: WebSocket, user_role):
        # Accept the WebSocket connection
        await websocket.accept()
        try:
            # Receive the initial payload (should contain question and optional search_depth)
            data = await websocket.receive_json()
            question = data.get("question")
            search_depth = data.get("search_depth", 10)

            # Stream tokens from the RAG pipeline
            async for token in self.company_mw.query_llm(question, user_role, search_depth):
                await websocket.send_text(token)
        except WebSocketDisconnect:
            # Client disconnected prematurely
            print("WebSocket client disconnected")
        except Exception as e:
            # Send error message back to client
            await websocket.send_text(json.dumps({"error": str(e)}))
        finally:
            await websocket.close()