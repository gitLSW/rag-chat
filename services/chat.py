import os
import asyncio
from datetime import datetime
from typing import AsyncGenerator, List, Dict, Optional
from ..get_env_var import get_env_var

class Chat:
    _llm_service = LLMService()
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.history: List[Dict] = []
        self._current_task: Optional[asyncio.Task] = None
        self._current_stream_queue = None
        self._partial_answer = ""


    async def query(
        self,
        prompt: str,
        question: str = None,
        show_chat_history: bool = False
    ) -> AsyncGenerator[str, None]:
        # Cancel any existing generation
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

        # Create new history entry
        entry = {
            'prompt': prompt,
            'question': question,
            'timestamp': datetime.now(),
            'answer': '',
            'answer_timestamp': None,
            'completed': False
        }
        self.history.append(entry)

        # Build full prompt with history if needed
        final_prompt = self._build_full_prompt(prompt, question, show_chat_history)
        
        # Create communication queue
        self._current_stream_queue = asyncio.Queue()
        self._current_task = asyncio.create_task(
            self._execute_generation(final_prompt, entry)
        )

        # Stream results
        try:
            while True:
                chunk = await self._current_stream_queue.get()
                if chunk is None:  # End of generation
                    entry['completed'] = True
                    entry['answer_timestamp'] = datetime.now()
                    break
                entry['answer'] += chunk
                self._partial_answer += chunk
                yield chunk
        finally:
            self._current_task = None
            self._current_stream_queue = None
            if not entry['completed']:
                self._partial_answer = ""  # Reset if not completed

    async def _execute_generation(self, prompt: str, entry: Dict):
        """Run generation and put results into queue"""
        try:
            stream = self._llm_service.query(
                prompt=prompt,
                question=None,  # Already incorporated into prompt
                sampling_params=None,
                stream=True
            )
            
            async for chunk in stream:
                await self._current_stream_queue.put(chunk)
            await self._current_stream_queue.put(None)
        except asyncio.CancelledError:
            await self._current_stream_queue.put(None)
            entry['completed'] = False
            raise


    def stop(self):
        """Stop current generation"""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._partial_answer = ""


    async def resume(self) -> AsyncGenerator[str, None]:
        """Resume generation with full history context"""
        if not self._partial_answer:
            raise ValueError("Nothing to resume")

        # Rebuild prompt with history and partial answer
        last_entry = self.history[-1]
        prompt = self._build_full_prompt(
            prompt=last_entry['prompt'],
            question=last_entry['question'],
            show_chat_history=True
        ) + self._partial_answer

        # Create new entry for the continuation
        return self.query(
            prompt=prompt,
            question="[CONTINUE]",  # Marker for continuation
            show_chat_history=False
        )


    def _build_full_prompt(self, prompt: str, question: str, show_chat_history: bool) -> str:
        """Construct the final prompt with chat history"""
        if not show_chat_history:
            return f"{prompt}\n{question}" if question else prompt

        history = []
        for entry in self.history:
            # Add previous Q&A pairs
            history.append(
                f"User: {entry['prompt']}"
                + (f"\nQuestion: {entry['question']}" if entry['question'] else "")
            )
            if entry['answer']:
                history.append(f"Assistant: {entry['answer']}")

        # Add current prompt
        current = f"User: {prompt}"
        if question:
            current += f"\nQuestion: {question}"
            
        return "\n\n".join(history + [current])


    @property
    def chat_history(self) -> List[Dict]:
        """Get complete chat history with metadata"""
        return [{
            'prompt': e['prompt'],
            'question': e['question'],
            'timestamp': e['timestamp'],
            'answer': e['answer'],
            'answer_timestamp': e['answer_timestamp'],
            'completed': e['completed']
        } for e in self.history]