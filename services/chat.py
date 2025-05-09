import re
import json
import asyncio
import aiofiles
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict
from ..get_env_var import get_env_var
from vllm_service import LLMService
from rag_service import get_company_rag_service
from mongo_db_connector import MongoDBConnector

@dataclass
class ChatEntry:
    message: str
    timestamp: datetime
    answer: str
    answer_timestamp: Optional[datetime]
    completed: bool


class Chat:
    llm_service = LLMService()  # Shared static LLM instance across sessions

    def __init__(self, company_id, session_id, user_access_role):
        self.company_id = company_id
        self.session_id = session_id
        self.history: List[ChatEntry] = []
        self.user_access_role = user_access_role
        self.rag_service = get_company_rag_service(company_id)
        self.mongo_db_connector = None
        
        self._generation_task: Optional[asyncio.Task] = None
        self._current_stream_queue: Optional[asyncio.Queue] = None
        self._current_req_id: Optional[str] = None


    async def query(self,
                    message,
                    search_depth = None,
                    use_db = False,
                    show_chat_history = False,
                    is_reasoning_model = False) -> AsyncGenerator[str, None]:
        # Cancel any existing generation first
        self.stop()

        if search_depth and use_db:
            raise ValueError('Either search_depth or use_db are allowed, not both.')

        prompt = message
        doc_summary_context = None
        if use_db:
            db_query, db_response = self.llm_db_query(message, doc_summaries, is_reasoning_model)
            if db_response:
                prompt += f"""This MongoDB query and response might be relevant to the previous message:\n
                            MongoDB query: {json.dumps(db_query, indent=2)}\n\n
                            MongoDB response: {json.dumps(db_response, indent=2)}"""
        elif search_depth:
            docs_data = self.rag_service.find_docs(message, search_depth)
            doc_summaries, _, doc_sources_map = self._summarize_docs(docs_data, message)
            doc_summary_context = "\n\n".join(doc_summaries)
            prompt += f'These texts might be relevant to the previous message:'

        # Create new chat entry
        entry = ChatEntry(
            message=message,
            timestamp=datetime.now(),
            answer='',
            answer_timestamp=None,
            completed=False
        )
        self.history.append(entry)

        # Build final prompt with optional chat history
        history = None
        if show_chat_history:
            history = self._get_chat_history()

        # Stream queue for chunk communication
        self._current_stream_queue = asyncio.Queue()

        # Kick off the streaming generation
        async def run():
            try:
                self._current_req_id = f'{self.company_id}-{self.session_id}'
                stream = self.llm_service.query(
                    prompt=prompt,
                    context=doc_summary_context,
                    history=history,
                    req_id=self._current_req_id
                )

                entry.answer_timestamp = datetime.now()
                async for chunk in stream:
                    await self._current_stream_queue.put(chunk)
                    entry.answer += chunk
                entry.completed = True
            except asyncio.CancelledError:
                # If manually stopped
                self.llm_service.cancel(self._current_req_id)
            finally:
                await self._current_stream_queue.put(None)

        self._generation_task = asyncio.create_task(run())

        # Yield chunks from the queue
        try:
            while True:
                chunk = await self._current_stream_queue.get()
                if chunk is None:
                    break
                yield chunk

            if doc_sources_map:
                yield self._generate_source_references_str(doc_sources_map)
        finally:
            # Cleanup after generation ends
            self._generation_task = None
            self._current_stream_queue = None
            self._current_req_id = None


    def stop(self):
        """Stop the current generation safely without affecting other chats."""
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
        if self._current_req_id:
            self.llm_service.cancel(self._current_req_id)

        self._generation_task = None
        self._current_req_id = None


    async def resume(self) -> AsyncGenerator[str, None]:
        """Resume from the last prompt and partial answer."""
        if not self.history or not self.history[-1].answer:
            raise ValueError("No interrupted generation to resume from.")

        last_entry = self.history[-1]

        resume_prompt = self._build_history_prompt(
            prompt=last_entry.prompt,
            show_chat_history=True
        ) + last_entry.answer

        return self.query(
            prompt=resume_prompt,
            question="[CONTINUE]",
            show_chat_history=False
        )


    def _get_chat_history(self):
        history_parts = []
        for entry in self.history:
            history_parts.append(f"User: {entry.prompt}")
            if entry.answer:
                history_parts.append(f"Assistant: {entry.answer}")
    
        return "\n\n".join(history_parts)


    async def _summarize_docs(self, docs_data, message):
        """Process documents concurrently to generate summaries and collect metadata."""
        doc_types = set()
        doc_sources_map = defaultdict(set)
        summarize_tasks = []
        
        for doc_data in docs_data:
            doc_id = doc_data['docId']
            page_num = doc_data['pageNum']
            doc_type = doc_data.get('docType')
            if doc_type:
                doc_types.add(doc_type)
    
            if page_num:
                doc_sources_map[doc_id].add(page_num)
            else:
                doc_sources_map[doc_id] = None
            
            summarize_tasks.append(self._load_and_summarize_doc(doc_id, message))
    
        # Run all LLM summaries concurrently
        doc_summaries = await asyncio.gather(*summarize_tasks)
        return doc_summaries, doc_types, doc_sources_map
    
    
    async def _load_and_summarize_doc(self, doc_id, message):
        # Loads and summarizes a single document.
        txt_path = f"./companies/{self.company_id}/docs/{doc_id}.txt"
        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
            doc_text = await f.read()

        message = f'Summarize all the relevant information and facts needed to answer the following message from the text:\n\n{message}'
        
        summary = ''
        req_id = f"{self.company_id}-{self.session_id}-doc-{doc_id}"

        try:
            async for output in Chat.llm_service.query(message, doc_text, process_chunks_concurrently=True, req_id=req_id):
                summary += output
        except asyncio.CancelledError:
            Chat.llm_service.cancel(req_id)

        return re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)

    
    def _generate_source_references_str(self, doc_sources_map):
        """Generate the source references string."""
        sources_info = 'Consult these documents for more detail:\n'
        for doc_id, pages in doc_sources_map.items():
            try:
                doc_pseudo_path = self.rag_service.json_db.find_one({ '_id': doc_id }).get('path')
            finally:
                doc_pseudo_path = f"Document with ID {doc_id}"
                
            sources_info += doc_pseudo_path
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'
        return sources_info
    
    
    async def llm_db_query(self, message, is_reasoning_model=True):
        # Attach MongoDB schema information
        context += '\n\n\nYou have read-only access to the MongoDB `docs` collection. All the documents in it have a doc_type field. These are the JSON schemata for each doc_type:'
    
        for doc_type, doc_schema in self.rag_service.doc_schemata:
            context += f'\n\Doc_type {doc_type}: {json.dumps(doc_schema)}'
        
        context += '\n\nIf you need to query the MongoDB, write a JSON query in tags like so: ```mongo_json YOUR_QUERY ```.'
        if is_reasoning_model:
            context += '\nWrite your final mongoDB json command with its tags inside your think tags. Like so: <think> YOUR THOUGHTS ```mongo_json YOUR_QUERY ``` </think>.'
        
        req_id = f"{self.company_id}-{self.session_id}-mongo"

        answer_buffer = ''
        mongo_query = None
        try:
            async for chunk in self.llm_service.query(message, context, allow_chunking=False, req_id=req_id):
                answer_buffer += chunk

                if chunk.contains('`'):
                    # Check for complete mongo_json block
                    mongo_match = re.search(r"mongo_json\s*(.*?)\s*", answer_buffer, re.DOTALL)
                    if mongo_match:
                        mongo_query = mongo_match.group(1)
                        break  # Stop the first generation
        except asyncio.CancelledError:
            self.llm_service.cancel(req_id)

        # If we found a mongo query, execute it and return it
        if mongo_query:
            # Init MongoDBConnector
            if not self.mongo_db_connector:
                self.mongo_db_connector = MongoDBConnector(self.company_id)

            # Execute MongoDB query
            return self.mongo_db_connector.run(mongo_query, self.user_access_role)

        return None, None