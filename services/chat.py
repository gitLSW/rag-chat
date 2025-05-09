import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional
from ..get_env_var import get_env_var

@dataclass
class ChatEntry:
    prompt: str
    question: Optional[str]
    timestamp: datetime
    answer: str
    answer_timestamp: Optional[datetime]
    completed: bool
    partial_answer: str = ""  # Used during streaming


class Chat:
    _llm_service = LLMService()  # Shared static LLM instance across sessions


    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[ChatEntry] = []
        self._current_task: Optional[asyncio.Task] = None
        self._current_stream_queue: Optional[asyncio.Queue] = None
        self._current_request_id: Optional[str] = None


    async def query(
        self,
        prompt: str,
        question: Optional[str] = None,
        show_chat_history: bool = False
    ) -> AsyncGenerator[str, None]:
        # Cancel any existing generation first
        self.stop()

        # Build final prompt with optional chat history
        final_prompt = self._build_history_prompt(prompt, show_chat_history)

        # Create new chat entry
        entry = ChatEntry(
            prompt=prompt,
            question=question,
            timestamp=datetime.now(),
            answer='',
            answer_timestamp=None,
            completed=False
        )
        self.history.append(entry)

        # Stream queue for chunk communication
        self._current_stream_queue = asyncio.Queue()

        # Kick off the streaming generation
        async def run():
            try:
                stream = self._llm_service.query(
                    prompt=final_prompt,
                    question=None,  # Already incorporated
                    stream=True
                )

                # Pull request ID from generator
                self._current_request_id = stream.request_id  # Assumes it's exposed here

                async for chunk in stream:
                    await self._current_stream_queue.put(chunk)
                    entry.answer += chunk
                    entry.partial_answer += chunk
                entry.completed = True
                entry.answer_timestamp = datetime.now()

            except asyncio.CancelledError:
                # If manually stopped
                self._llm_service.cancel(self._current_request_id)
            finally:
                await self._current_stream_queue.put(None)

        self._current_task = asyncio.create_task(run())

        # Yield chunks from the queue
        try:
            while True:
                chunk = await self._current_stream_queue.get()
                if chunk is None:
                    break
                yield chunk
        finally:
            # Cleanup after generation ends
            self._current_task = None
            self._current_stream_queue = None
            self._current_request_id = None


    def stop(self):
        """Stop the current generation safely without affecting other chats."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
        if self._current_request_id:
            self._llm_service.cancel(self._current_request_id)
        self._current_task = None
        self._current_request_id = None


    async def resume(self) -> AsyncGenerator[str, None]:
        """Resume from the last prompt and partial answer."""
        if not self.history or not self.history[-1].partial_answer:
            raise ValueError("No interrupted generation to resume from.")

        last_entry = self.history[-1]

        resume_prompt = self._build_history_prompt(
            prompt=last_entry.prompt,
            show_chat_history=True
        ) + last_entry.partial_answer

        return self.query(
            prompt=resume_prompt,
            question="[CONTINUE]",
            show_chat_history=False
        )


    def _build_history_prompt(self, prompt: str, show_chat_history: bool) -> str:
        """Prepend full chat history to the current prompt if requested."""
        if not show_chat_history:
            return prompt
    
        history_parts = []
        for entry in self.history:
            history_parts.append(f"User: {entry.prompt}")
            if entry.answer:
                history_parts.append(f"Assistant: {entry.answer}")
    
        return "\n\n".join(history_parts + [f"User: {prompt}"])


    @property
    def chat_history(self) -> List[ChatEntry]:
        """Return structured chat history."""
        return self.history