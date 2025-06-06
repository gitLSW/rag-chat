import re
import json
import uuid
import asyncio
import aiofiles
from datetime import datetime
from collections import defaultdict
from typing import AsyncGenerator, Optional, Dict
from utils import get_env_var, get_company_path, safe_async_read
from pymongo import MongoClient
from vllm import SamplingParams
from services.vllm_service import LLMService
from services.rag_service import get_company_rag_service
from services.mongo_db_connector import MongoDBConnector

MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class ChatEntry:
    def __init__(self, message, use_db=None, rag_search_depth=None, show_chat_history=False):
        self.message = message
        self.time = datetime.now()
        self.answer: str = ""

        self.use_db = use_db
        self.rag_search_depth = rag_search_depth
        self.show_chat_history = show_chat_history


class LLMChat:
    def __init__(self, user_id, company_id, user_access_roles):
        self.user_id = user_id
        self.chat_id = str(uuid.uuid4())
        self.company_id = company_id
        self.user_access_roles = user_access_roles

        self.llm_service = LLMService()
        self.rag_service = get_company_rag_service(company_id)
        self.mongo_db_connector = None
        
        self._curr_chat_entry = None
        self._db_query_req_id: Optional[str] = None
        self._summarize_doc_req_ids: Optional[Dict[str, str]] = {} # [doc_id: doc_req_id]
        self._answer_req_id: Optional[str] = None

        # Create or connect to database
        client = MongoClient(MONGO_DB_URL)
        self.messages_db = client[company_id]['messages']


    async def query(self,
                    message,
                    use_db = False,
                    rag_search_depth = None,
                    show_chat_history = False) -> AsyncGenerator[str, None]:
        new_entry = ChatEntry(message, use_db, rag_search_depth, show_chat_history)

        self.abort()
        async for chunk in self._generate(new_entry):
            yield chunk


    async def resume(self):
        if not self._curr_chat_entry:
            raise ValueError("Nothing to continue")
        
        self.pause()
        async for chunk in self._generate(self._curr_chat_entry):
            yield chunk
        

    async def _generate(self, entry) -> AsyncGenerator[str, None]:
        if entry.rag_search_depth and entry.use_db:
            raise ValueError("Either rag_search_depth or use_db are allowed, not both.")
        
        self._curr_chat_entry = entry

        # Build final prompt with optional chat history
        history = self._get_chat_history() if entry.show_chat_history else None

        prompt = entry.message
        doc_sources_summary = None
        doc_sources_map = None
        resume_req_id = self._answer_req_id

        if resume_req_id: # If a final generation task is still able to be resumed, we don't need to repeat its preprocessing
            generator = self.llm_service.resume(resume_req_id)
        else:
            if entry.use_db:
                db_query, db_response = await self._llm_db_query(entry.message)
                if db_response:
                    prompt += f"""This MongoDB query and response might be relevant to the previous message:\n
                                MongoDB query: {json.dumps(db_query, indent=2)}\n\n
                                MongoDB response: {json.dumps(db_response, indent=2)}"""
            elif entry.rag_search_depth:
                docs_data = self.rag_service.find_docs(entry.message, entry.rag_search_depth)
                doc_summaries, doc_sources_map = self._summarize_docs(docs_data, entry.message)
                doc_sources_summary = '\n\n'.join(doc_summaries)
                prompt += "These texts might be relevant to the previous message:"

            resume_req_id = f"{self.user_id}-{self.chat_id}-final-answer"
            generator = self.llm_service.query(prompt=prompt,
                                               context=doc_sources_summary,
                                               history=history,
                                               req_id=resume_req_id)
            self._answer_req_id = resume_req_id

        # Now we can safely remove the preprocessing tasks, because they completed and we started and saved the final generation
        self._db_query_req_id = None
        self._summarize_doc_req_ids = {}

        try:
            async for chunk in generator:
                entry.answer += chunk
                yield chunk
        except asyncio.CancelledError:
            raise asyncio.CancelledError("Final generation request cancelled")

        if doc_sources_map:
            source_references_str = self._generate_source_references_str(doc_sources_map)
            entry.answer += source_references_str
            yield source_references_str

        self.messages_db.insert_one({
            'message': self.message,
            'time': self.time,
            'answer': self.answer,
            'userId': self.user_id,
            'chatId': self.chat_id
        })
        self.abort()


    async def pause(self):
        """Stop the current generation safely without affecting other chats."""
        if self._db_query_req_id:
            await self.llm_service.pause(self._db_query_req_id)

        if self._summarize_doc_req_ids:
            for summarize_doc_req_id in self._summarize_doc_req_ids.values():
                await self.llm_service.pause(summarize_doc_req_id)

        if self._answer_req_id:
            await self.llm_service.pause(self._answer_req_id)


    def abort(self):
        if self._db_query_req_id:
            asyncio.create_task(self.llm_service.abort(self._db_query_req_id))

        if self._summarize_doc_req_ids:
            for summarize_doc_req_id in self._summarize_doc_req_ids.values():
                asyncio.create_task(self.llm_service.abort(summarize_doc_req_id))

        if self._answer_req_id:
            asyncio.create_task(self.llm_service.abort(self._answer_req_id))

        self._db_query_req_id = None
        self._summarize_doc_req_ids = {}
        self._answer_req_id = None
        self._curr_chat_entry = None


    # -----------------------------
    # Summarize Docs
    # -----------------------------

    async def _summarize_docs(self, docs_data, message):
        """Process documents concurrently to generate summaries and collect metadata."""
        async def _load_and_summarize_doc(doc_id, message):
            # Loads and summarizes a single document.
            txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
            try:
                doc_text = await safe_async_read(txt_path)
            except Exception:
                return "" # TODO: Maybe throw exception

            message = f"Summarize all the relevant information and facts needed to answer the following message from the text:\n\n{message}"
                    
            req_id = f'{self.user_id}-{self.chat_id}-doc-{doc_id}'
            generator = LLMChat.llm_service.query(message, doc_text, req_id=req_id)
            self._summarize_doc_req_ids[doc_id] = req_id

            return await _resume_doc_summary(generator)
    
    
        async def _resume_doc_summary(generator):
            summary = ""
            try:
                async for output in generator:
                    summary += output
            except asyncio.CancelledError:
                raise asyncio.CancelledError("Summarize request cancelled")

            return re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL)


        doc_sources_map = defaultdict(set)
        summarize_tasks = []
        for doc_data in docs_data:
            doc_id = doc_data['docId']
            page_num = doc_data['pageNum']
    
            if page_num:
                doc_sources_map[doc_id].add(page_num)
            else:
                doc_sources_map[doc_id] = None
            
            req_id = self._summarize_doc_req_ids.get(doc_id)
            if req_id:
                generator = self.llm_service.resume(req_id)
                summarize_tasks.append(asyncio.create_task(_resume_doc_summary(generator)))
            else:
                summarize_tasks.append(asyncio.create_task(_load_and_summarize_doc(doc_id, message)))
    
        # Run all LLM summaries concurrently
        doc_summaries = await asyncio.gather(*summarize_tasks)
        return doc_summaries, doc_sources_map
    
    
    # -----------------------------
    # LLM DB Query
    # -----------------------------
    
    async def _llm_db_query(self, message):
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.4,
            max_tokens=2048
            # stop=["\n\n", "\n", "Q:", "###"]
        )
        async def _resume_db_query(generator):
            answer_buffer = ""
            mongo_query = None
            try:
                async for chunk in generator:
                    answer_buffer += chunk

                    if "`" in chunk:
                        # Check for complete mongo_json block
                        mongo_match = re.search(r"mongo_json\s*(.*?)\s*", answer_buffer, re.DOTALL)
                        if mongo_match:
                            mongo_query = mongo_match.group(1)
                            break  # Stop the first generation
            except asyncio.CancelledError:
                raise asyncio.CancelledError("DB request cancelled")

            # If we found a mongo query, execute it and return it
            if mongo_query:
                # Init MongoDBConnector
                if not self.mongo_db_connector:
                    self.mongo_db_connector = MongoDBConnector(self.company_id)

                # Execute MongoDB query
                return self.mongo_db_connector.run(mongo_query, self.user_access_roles)

            return None, None
        
        req_id = self._db_query_req_id
        if req_id:
            generator = self.llm_service.resume(req_id)
            return await _resume_db_query(generator)

        # Attach MongoDB schema information
        context = "\n\n\nYou have read-only access to the MongoDB `docs` collection. All the documents in it have a doc_type field. These are the JSON schemata for each doc_type:"
    
        for doc_type, doc_schema in self.rag_service.doc_schemata:
            context += f"\n\nDoc_type {doc_type}: {json.dumps(doc_schema)}"
        
        context += "\n\nTry to answer the following message. If you need to query the MongoDB, write a JSON query in tags like so: ```mongo_json YOUR_QUERY ```."
        
        req_id = f"{self.user_id}-{self.chat_id}-mongo"
        generator = self.llm_service.query(message, context, req_id=req_id, sampling_params=sampling_params, allow_chunking=False)
        self._db_query_req_id = req_id
        return await _resume_db_query(generator)
    
    
    # -----------------------------
    # Utils
    # -----------------------------

    def _get_chat_history(self):
        history = []
        for entry in self.get_chat_messages():
            message = entry.get('message')
            answer = entry.get('answer')
            if not message or not answer:
                continue
                
            history.append(f"User: {message}")
            history.append(f"You: {answer}")
        
        return '\n\n'.join(history)
    

    def get_chat_messages(self):
        return self.messages_db.find({
            'chatId': self.chat_id,
            'userId': self.user_id
        })


    def _generate_source_references_str(self, doc_sources_map):
        """Generate the source references string."""
        sources_info = "Consult these documents for more detail:\n"
        for doc_id, pages in doc_sources_map.items():
            doc_pseudo_path = None # prevents UnboundLocalError !
            try:
                doc = self.rag_service.docs_db.find_one({ '_id': doc_id })
                if doc:
                    doc_pseudo_path = doc.get('path')
            except Exception:
                pass

            if not doc_pseudo_path:
                doc_pseudo_path = f"Document with ID {doc_id}"
                
            sources_info += doc_pseudo_path
            if pages is None:
                sources_info += "\n"
            else:
                sources_info += f" on pages {', '.join(map(str, sorted(pages)))}\n"
        return sources_info