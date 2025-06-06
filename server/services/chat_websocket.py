import asyncio
from pydantic import BaseModel, ValidationError
from typing import Optional, Literal, Union
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.llm_chat import LLMChat


class BaseChatAction(BaseModel):
    action: str
    chat_id: str

class StartChatAction(BaseChatAction):
    action: Literal["start"]

class MessageAction(BaseChatAction):
    action: Literal["message"]
    message: str
    use_db: Optional[bool] = False
    rag_search_depth: Optional[int] = None
    show_chat_history: Optional[bool] = False
    resuming: Optional[bool] = False

class PauseChatAction(BaseChatAction):
    action: Literal["pause"]

class DeleteChatAction(BaseChatAction):
    action: Literal["delete"]

class ResumeChatAction(BaseChatAction):
    action: Literal["resume"]

ChatAction = Union[
    StartChatAction,
    MessageAction,
    PauseChatAction,
    DeleteChatAction,
    ResumeChatAction,
]


router = APIRouter()
active_chats: dict[str, LLMChat] = {}


@router.websocket("/chat")
async def websocket_query(websocket: WebSocket):
    await websocket.accept()
    company_id = websocket.scope["state"].company_id
    user = websocket.scope["state"].user

    try:
        while True:
            raw_data = await websocket.receive_json()

            try:
                action = ChatAction.model_validate(raw_data)
            except ValidationError as ve:
                await websocket.send_text(f"[ERROR] Invalid message format: {ve}")
                continue

            # ------------------------
            # Start a New Chat
            # ------------------------
            if isinstance(action, StartChatAction):
                if action.chat_id in active_chats:
                    await websocket.send_text(f"[ERROR] Chat {action.chat_id} already exists")
                    continue
                active_chats[action.chat_id] = LLMChat(
                    userId=user.id,
                    company_id=company_id,
                    chat_id=action.chat_id,
                    user_access_roles=user.access_roles,
                )
                await websocket.send_text(f"[STARTED] Chat {action.chat_id}")

            # ------------------------
            # Send Message
            # ------------------------
            elif isinstance(action, MessageAction):
                chat = active_chats.get(action.chat_id)
                if not chat:
                    await websocket.send_text(f"[ERROR] Chat {action.chat_id} not found")
                    continue

                try:
                    async for chunk in chat.query(
                        message=action.message,
                        use_db=action.use_db,
                        rag_search_depth=action.rag_search_depth,
                        show_chat_history=action.show_chat_history,
                        resuming=action.resuming,
                    ):
                        await websocket.send_text(chunk)
                    await websocket.send_text("[DONE]")
                except asyncio.CancelledError:
                    await websocket.send_text("[CANCELLED]")

            # ------------------------
            # Pause Chat
            # ------------------------
            elif isinstance(action, PauseChatAction):
                chat = active_chats.get(action.chat_id)
                if chat:
                    await chat.pause()
                    await websocket.send_text(f"[PAUSED] Chat {action.chat_id}")
                else:
                    await websocket.send_text(f"[ERROR] Chat {action.chat_id} not found")

            # ------------------------
            # Resume Chat
            # ------------------------
            elif isinstance(action, ResumeChatAction):
                chat = active_chats.get(action.chat_id)
                if chat:
                    try:
                        async for chunk in chat.resume():
                            await websocket.send_text(chunk)
                        await websocket.send_text("[DONE]")
                    except asyncio.CancelledError:
                        await websocket.send_text("[CANCELLED]")
                else:
                    await websocket.send_text(f"[ERROR] Chat {action.chat_id} not found")

            # ------------------------
            # Delete Chat
            # ------------------------
            elif isinstance(action, DeleteChatAction):
                chat = active_chats.pop(action.chat_id, None)
                if chat:
                    await chat.abort()
                    del chat
                    await websocket.send_text(f"[DELETED] Chat {action.chat_id}")
                else:
                    await websocket.send_text(f"[ERROR] Chat {action.chat_id} not found")

    except WebSocketDisconnect:
        for chat in active_chats.values():
            await chat.pause()
        active_chats.clear()
    except Exception as e:
        await websocket.send_text(f"[ERROR] {str(e)}")
        for chat in active_chats.values():
            await chat.abort()
        active_chats.clear()