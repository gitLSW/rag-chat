import re
import json
import asyncio
import aiofiles
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict, Set
from ..get_env_var import get_env_var
from vllm_service import LLMService
from rag_service import get_company_rag_service
from mongo_db_connector import MongoDBConnector

@dataclass
class ChatEntry:
    message: str
    timestamp: datetime

    use_db: Optional[bool]= None
    search_depth: Optional[int]= None
    show_chat_history: Optional[bool]= None
    is_reasoning_model: bool = False
    
    answer: str = ''
    answer_timestamp: Optional[datetime] = None
    completed: bool = False


class Chat:
    llm_service = LLMService()  # Shared static LLM instance across sessions

    def __init__(self, company_id, session_id, user_access_role):
        self.company_id = company_id
        self.session_id = session_id
        self.history: List[ChatEntry] = []
        self.user_access_role = user_access_role
        self.rag_service = get_company_rag_service(company_id)
        self.mongo_db_connector = None
        
        self._db_query_req_id: Optional[str] = None
        self._summarize_doc_req_ids: Optional[Dict[str, str]] = {}
        self._generation_req_id: Optional[str] = None


    async def query(self,
                    message,
                    use_db = False,
                    search_depth = None,
                    show_chat_history = False,
                    is_reasoning_model = False,
                    resuming = False) -> AsyncGenerator[str, None]:
        if search_depth and use_db:
            raise ValueError('Either search_depth or use_db are allowed, not both.')

        # Create new chat entry
        if resuming:
            self.pause()
            entry = self.history[-1]
        else:
            self.abort() # Stops the automatic resume of _llm_db_query and _summarize_docs
            entry = ChatEntry(
                message=message,
                timestamp=datetime.now(),
                use_db=use_db,
                search_depth=search_depth,
                show_chat_history=show_chat_history,
                is_reasoning_model=is_reasoning_model
            )
            self.history.append(entry)

        # Build final prompt with optional chat history
        history = None
        if show_chat_history:
            history = self._get_chat_history()

        prompt = message
        doc_summary_context = None
        doc_sources_map = None
        req_id = self._generation_req_id

        if req_id: # If a final generation task is still able to be resumed, we don't need to repeat its preprocessing
            generator = self.llm_service.resume(req_id)
        else:
            if use_db:
                db_query, db_response = await self._llm_db_query(message, is_reasoning_model)
                if db_response:
                    prompt += f"""This MongoDB query and response might be relevant to the previous message:\n
                                MongoDB query: {json.dumps(db_query, indent=2)}\n\n
                                MongoDB response: {json.dumps(db_response, indent=2)}"""
            elif search_depth:
                docs_data = self.rag_service.find_docs(message, search_depth)
                doc_summaries, _, doc_sources_map = self._summarize_docs(docs_data, message)
                doc_summary_context = "\n\n".join(doc_summaries)
                prompt += f'These texts might be relevant to the previous message:'

            req_id = f"{self.company_id}-{self.session_id}-final-answer"
            entry.answer_timestamp = datetime.now()
            generator = self.llm_service.query(
                prompt=prompt,
                context=doc_summary_context,
                history=history,
                req_id=req_id
            )
            self._generation_req_id = req_id

        try:
            async for chunk in generator:
                entry.answer += chunk
                yield chunk
        except asyncio.CancelledError:
            raise asyncio.CancelledError('Final generation request cancelled')

        if doc_sources_map:
            yield self._generate_source_references_str(doc_sources_map)

        entry.completed = True
        self.abort()


    async def resume(self):
        entry = self.history[-1]
        if not entry:
            return
        
        async for chunk in self.query(entry.message,
                                      entry.use_db,
                                      entry.search_depth,
                                      entry.show_chat_history,
                                      entry.is_reasoning_model,
                                      True):
            entry.answer += chunk
            yield chunk
        

    def pause(self):
        """Stop the current generation safely without affecting other chats."""
        if self._db_query_req_id:
            self.llm_service.pause(self._db_query_req_id)

        if self._summarize_doc_req_ids:
            for summarize_doc_req_id in self._summarize_doc_req_ids.values():
                self.llm_service.pause(summarize_doc_req_id)

        if self._generation_req_id:
            self.llm_service.pause(self._generation_req_id)


    def abort(self):
        if self._db_query_req_id:
            self.llm_service.abort(self._db_query_req_id)

        if self._summarize_doc_req_ids:
            for summarize_doc_req_id in self._summarize_doc_req_ids.values():
                self.llm_service.abort(summarize_doc_req_id)

        if self._generation_req_id:
            self.llm_service.abort(self._generation_req_id)

        self._db_query_req_id = None
        self._summarize_doc_req_ids = {}
        self._generation_req_id = None


    def _get_chat_history(self):
        history_parts = []
        for entry in self.history:
            history_parts.append(f"User: {entry.message}")
            if entry.answer:
                history_parts.append(f"Assistant: {entry.answer}")
    
        return "\n\n".join(history_parts)


    # -----------------------------
    # Summarize Docs
    # -----------------------------

    async def _summarize_docs(self, docs_data, message):
        """Process documents concurrently to generate summaries and collect metadata."""
        async def _resume_doc_summary(generator):
            summary = ''
            try:
                async for output in generator:
                    summary += output
            except asyncio.CancelledError:
                raise asyncio.CancelledError('Summarize request cancelled')

            return re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)

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
            
            req_id = self._summarize_doc_req_ids.get(doc_id)
            if req_id:
                generator = self.llm_service.resume(req_id)
                summarize_tasks.append(_resume_doc_summary(generator))
            else:
                async def _load_and_summarize_doc(doc_id, message):
                    # Loads and summarizes a single document.
                    txt_path = f"./companies/{self.company_id}/docs/{doc_id}.txt"
                    try:
                        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
                            doc_text = await f.read()
                    except FileNotFoundError:
                        return '' # TODO: Maybe throw exception

                    message = f'Summarize all the relevant information and facts needed to answer the following message from the text:\n\n{message}'
                    
                    req_id = f"{self.company_id}-{self.session_id}-doc-{doc_id}"
                    generator = Chat.llm_service.query(message, doc_text, req_id=req_id)
                    self._summarize_doc_req_ids[doc_id] = req_id

                    return await _resume_doc_summary(generator)
                summarize_tasks.append(_load_and_summarize_doc(doc_id, message))
    
        # Run all LLM summaries concurrently
        doc_summaries = await asyncio.gather(*summarize_tasks)
        self._summarize_doc_req_ids = {}
        return doc_summaries, doc_types, doc_sources_map
    
    
    # -----------------------------
    # LLM DB Query
    # -----------------------------
    
    async def _llm_db_query(self, message, is_reasoning_model=True):
        async def _resume_db_query(generator):
            answer_buffer = ''
            mongo_query = None
            try:
                async for chunk in generator:
                    answer_buffer += chunk

                    if '`' in chunk:
                        # Check for complete mongo_json block
                        mongo_match = re.search(r"mongo_json\s*(.*?)\s*", answer_buffer, re.DOTALL)
                        if mongo_match:
                            mongo_query = mongo_match.group(1)
                            break  # Stop the first generation
            except asyncio.CancelledError:
                raise asyncio.CancelledError('DB request cancelled')

            # If we found a mongo query, execute it and return it
            if mongo_query:
                # Init MongoDBConnector
                if not self.mongo_db_connector:
                    self.mongo_db_connector = MongoDBConnector(self.company_id)

                # Execute MongoDB query
                return self.mongo_db_connector.run(mongo_query, self.user_access_role)

            return None, None
        
        req_id = self._db_query_req_id
        if req_id:
            generator = self.llm_service.resume(req_id)
            return await _resume_db_query(generator)

        # Attach MongoDB schema information
        context = '\n\n\nYou have read-only access to the MongoDB `docs` collection. All the documents in it have a doc_type field. These are the JSON schemata for each doc_type:'
    
        for doc_type, doc_schema in self.rag_service.doc_schemata:
            context += f'\n\nDoc_type {doc_type}: {json.dumps(doc_schema)}'
        
        context += '\n\nIf you need to query the MongoDB, write a JSON query in tags like so: ```mongo_json YOUR_QUERY ```.'
        if is_reasoning_model:
            context += '\nWrite your final mongoDB json command with its tags inside your think tags. Like so: <think> YOUR THOUGHTS ```mongo_json YOUR_QUERY ``` </think>.'
        
        req_id = f"{self.company_id}-{self.session_id}-mongo"
        generator = self.llm_service.query(message, context, allow_chunking=False, req_id=req_id)
        self._db_query_req_id = req_id
        return await _resume_db_query(generator)
    
    
    # -----------------------------
    # Utils
    # -----------------------------

    def _generate_source_references_str(self, doc_sources_map):
        """Generate the source references string."""
        sources_info = 'Consult these documents for more detail:\n'
        for doc_id, pages in doc_sources_map.items():
            try:
                doc = self.rag_service.json_db.find_one({ '_id': doc_id })
                doc_pseudo_path = doc.get('path')
            except Exception:
                doc_pseudo_path = f"Document with ID {doc_id}"
                
            sources_info += doc_pseudo_path
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'
        return sources_info