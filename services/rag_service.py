import re
import os
import json
import logging
from collections import defaultdict
from ..get_env_var import get_env_var

from filelock import FileLock
import asyncio
import aiofiles

import jsonschema
from fastapi import HTTPException
from fastapi.responses import JSONResponse

from pymongo import MongoClient
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from sentence_transformers import SentenceTransformer, util # Pretrained model to convert text into numerical vectors (embeddings)

from access_manager import get_access_manager, InsufficientAccessError, DocumentNotFoundError
from vllm import SamplingParams
from vllm_service import LLMService
from doc_extractor import DocExtractor
from doc_path_classifier import DocPathClassifier
from mongo_db_connector import MongoDBConnector

logger = logging.getLogger(__name__)

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph

BASE_DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "path": {"type": "string"},
        "docType": {"type": "string"},
        "accessGroups": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1
        }
    },
    "required": ["id", "path", "docType", "accessGroups"]
}


class OKResponse(JSONResponse):
    def __init__(self, detail='Success', data=None):
        self.data = data
        self.detail = detail
        super().__init__(status_code=200, content={
            "detail": detail,
            "data": data
        })


class RAGService:
    # Initialize persistent vector database (ChromaDB)
    vector_db = chromadb.PersistentClient(path="chroma_db")
    doc_extractor = DocExtractor()
    llm_service = LLMService()

    # Load sentence embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def __init__(self, company_id):
        """
        Initializes the RAGService instance.

        - Sets up a persistent vector database using ChromaDB to store text embeddings.
        - Loads a pre-trained OCR model with specific detection and recognition architectures.
        - Configures OCR to group detected text into lines and paragraphs.
        - Connects to a local LLM API (e.g., via Ollama).
        - Loads a sentence transformer for generating vector embeddings of text.
        """
        self.company_id = company_id
        self.access_manager = get_access_manager(company_id)
        self.vector_db = RAGService.vector_db.get_or_create_collection(name=company_id) # TODO: Check if this raises an exception, it should
        self.doc_path_classifier = DocPathClassifier(company_id)
        self.mongo_db_connector = MongoDBConnector(company_id)

        self.schemata_path = f'./companies/{company_id}/doc_schemata.json'
        with open(self.schemata_path, "r", encoding="utf-8", errors="ignore") as f:
            self.doc_schemata = json.loads(f.read())

        client = MongoClient(MONGO_DB_URL)

        # Create or connect to database
        self.json_db = client[company_id]['docs']


    def add_json_schema_type(self, doc_type, json_schema, user_access_role):
        if user_access_role != 'admin':
            raise InsufficientAccessError(user_access_role, 'Insufficient access rights, permission denied. Admin rights required')
        
        if doc_type in self.doc_schemata.keys():
            raise HTTPException(409, f'The document type {doc_type}, is already used by another JSON schema.')
        
        json_type = json_schema.get('type')
        if not json_type or json_type != 'object':
            raise HTTPException(400, 'The JSON schema must be an object type')

        jsonschema.Draft7Validator.check_schema(json_schema) # will raise jsonschema.exceptions.SchemaError if invalid
        
        lock = FileLock(self.schemata_path)
        with lock:
            self.doc_schemata[doc_type] = json_schema
            with open(self.schemata_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(json.dumps(self.doc_schemata))
        
        return OKResponse(f'Successfully added new JSON schema for {doc_type}', json_schema)


    def delete_json_schema_type(self, doc_type, user_access_role):
        if user_access_role != 'admin':
            raise InsufficientAccessError(user_access_role, 'Insufficient access rights, permission denied. Admin rights required')
        
        if self.json_db.find_one({ 'doc_type': doc_type }):
            raise HTTPException(409, f'Cannot delete schema "{doc_type}" because it was already used to extract a document.')
        
        deleted_schema = self.doc_schemata[doc_type]
        
        lock = FileLock(self.schemata_path)
        with lock:
            del self.doc_schemata[doc_type]
            with open(self.schemata_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(json.dumps(self.doc_schemata))
        
        return OKResponse(f'Successfully deleted JSON schema for "{doc_type}"')
    

    async def create_doc(self, source_path, doc_data, force_ocr, allow_override, user_access_role):
        doc_id = doc_data.get('id')
        if not doc_id:
            raise HTTPException(400, 'docData must contain an "id"')
        
        txt_path = f"./companies/{self.company_id}/docs/{doc_id}.txt"
        if not allow_override and os.path.exists(txt_path):
            raise HTTPException(409, f'Doc {doc_id} already exists and override was disallowed !')
        
        # Validate user access
        try:
            self.access_manager.has_doc_access(doc_id, user_access_role)
        except DocumentNotFoundError as e:
            pass # Expeceted behavior

        doc_data['accessGroups'] = self.access_manager.validate_new_access_groups(doc_data.get('accessGroups'))

        paragraphs = RAGService.doc_extractor.extract_paragraphs(source_path, force_ocr)
        doc_text = '\n\n'.join(paragraph for _, paragraph in paragraphs)
        
        # Classify the pseudo path (it is only used as a tool for users to organise themselves and has nothing to do with the file location)
        if not doc_data.get('path'):
            # Classify Document into a path if non existant
            file_name = source_path.split('/')[-1] # Last element
            doc_data['path'] = self.doc_path_classifier.classify_doc(doc_text) + file_name

        # Extract JSON
        doc_type = doc_data.get('docType')
        extracted_doc_data, doc_type, doc_schema, is_extract_valid = await self.extract_json(doc_text, doc_type)
        
        if is_extract_valid:
            # Overwrite extracted data with uploaded data
            doc_data = { **extracted_doc_data, **doc_data }
        
            # Build final schema
            doc_schema = self._merge_with_base_schema(doc_schema)
        else:
            doc_schema = BASE_DOC_SCHEMA
           
        print(doc_data)
        # Check if doc_data is valid and insert into DB
        jsonschema.validate(doc_data, doc_schema)
        self.json_db.replace_one({ '_id': doc_id }, doc_data, upsert=True) # Create doc or override existant doc

        # In case of override, remove all old embedding entries
        self.vector_db.delete(where={ 'doc_id': doc_id })
        for page_num, paragraph in paragraphs:
            # Convert paragraph into an embedding (vector representation)
            block_embedding = RAGService.embedding_model.encode(paragraph)

            # Store the paragraph's metadata in the vector DB using its embedding
            self.vector_db.upsert(
                embeddings=[block_embedding.tolist()],
                ids=[str(hash(str(block_embedding)))],
                metadatas=[{
                    'docId': doc_id,
                    'pageNum': page_num,
                    'docType': doc_type
                }]
            )

        # Save the extracted content in .txt file
        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_text)

        return OKResponse(f'Successfully processed Document', doc_data)


    def update_doc_data(self, doc_data, merge_existing, user_access_role):
        doc_id = doc_data.get('id')
        if not doc_id:
            raise HTTPException(400, 'docData must contain an "id"')
        
        old_doc = self.access_manager.has_doc_access(doc_id, user_access_role)
        
        old_doc_type = old_doc['docType']
        merged_doc = { **old_doc, **doc_data }
        
        doc_type = merged_doc.get('docType')
        if doc_type != old_doc_type:
            merge_existing = False
            
        doc_schema = self.doc_schemata.get(doc_type)
        if not doc_schema:
            raise HTTPException(422, 'Unknown doc_type. Register the JSON schema with POST /documentSchemata first')
        doc_schema = self._merge_with_base_schema(doc_schema)
        
        updated_doc = merged_doc if merge_existing else doc_data
        
        # Validate user access
        access_groups = updated_doc.get('accessGroups')
        if not access_groups:
            access_groups = old_doc['accessGroups']
        updated_doc['accessGroups'] = self.access_manager.validate_new_access_groups(access_groups)
        
        # If merge wasn't allowed and doc_data is missing a path, take the old one
        if not updated_doc.get('path'):
            updated_doc['path'] = old_doc['path']
        
        jsonschema.validate(updated_doc, doc_schema)
        self.json_db.replace_one({ '_id': doc_id }, updated_doc)
        
        return OKResponse(f'Successfully updated Document {doc_id}', updated_doc)


    def get_doc(self, doc_id, user_access_role):
        doc = self.access_manager.has_doc_access(doc_id, user_access_role)

        txt_path = f'./companies/{self.company_id}/docs/{doc_id}.txt'
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()
        
        doc['text'] = doc_text
        
        return OKResponse(f'Successfully retrieved Document {doc_id}', doc)
        
    
    def delete_doc(self, doc_id, user_access_role):
        # Validate user access
        self.access_manager.has_doc_access(doc_id, user_access_role)

        # Delete from vector DB
        self.vector_db.delete(where={ 'doc_id': doc_id })

        # Delete file
        txt_path = f"./companies/{self.company_id}/docs/{doc_id}.txt"
        os.remove(txt_path) # TODO: Check if this raises an exception, it should

        # Delete from json DB
        doc_data = self.json_db.delete_one({ '_id': doc_id })
        if not doc_data:
            raise HTTPException(404, f"Doc {doc_id} doesn't exist and thus couldn't be removed.")

        return OKResponse(f'Successfully deleted Document {doc_id}', doc_data)
    

    def find_docs(self, question, n_results, user_access_role):
        """
        Retrieves the most relevant document chunks for a given question using semantic search between question and known paragraphs

        Args:
            question (str): User's query or question.
            n_results (int): Number of top-matching results to return.

        Returns:
            list: Metadata entries (doc_id, page_num and doc_type) for the top matches.
        """
        # Convert user question into an embedding vector
        question_embedding = RAGService.embedding_model.encode(question).tolist()
        # Perform semantic search in ChromaDB (based on embedding similarity between question and the paragraphs of all document)
        results = self.vector_db.query(query_embeddings=[question_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        
        valid_docs_data = set()
        for doc_data in nearest_neighbors:
            try:
                doc_id = doc_data['id']
                self.access_manager.has_doc_access(doc_id, user_access_role)
            except DocumentNotFoundError:
                logger.warning(f'Corrupt data. VectorDB is referencing a missing doc with id {doc_id} for company {self.company_id} !')
                continue
            except InsufficientAccessError:
                continue
            
            if not any(lambda doc: doc == doc_data for doc in valid_docs_data):
                valid_docs_data.append(doc_data)

        return OKResponse(f'Found {len(valid_docs_data)}', valid_docs_data)


    async def extract_json(self, text, doc_type=None):
        """
        Extracts a filled JSON object from LLM output based on a schema and checks if all fields are filled.

        Args:
            text (str): Input text to extract data from.
            json_schema (dict): The target JSON schema structure.

        Returns:
            tuple:
                - dict: The parsed JSON object.
                - str: The document Type
                - dict: The JSON schema used to validate it
                - bool: Whether all schema fields were filled.
        """
        if doc_type:
            json_schema = self.doc_schemata.get(doc_type)
        else:
            json_schema, doc_type = self._identify_doc_type(text)

        if not json_schema:
            return None, doc_type, json_schema, False

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.4,
            max_tokens=4096
            # stop=["\n\n", "\n", "Q:", "###"]
        )

        prompt = f"""This is a JSON Schema which you need to fill:

            {json.dumps(json_schema)}

            ### TASK REQUIREMENT
            You are a json extractor. You are tasked with extracting the relevant information needed to fill the JSON schema from the text below.
            
            {text}

            ### STRICT RULES FOR GENERATING OUTPUT:
            **ALWAYS PROVIDE YOUR FINAL ANSWER**:
            - Always provide the filled JSON
            **JSON Schema Mapping:**:
            - Carefully check to find ALL relevant information in the text like ids, names, addresses, etc.
            - Sometimes ids my be called numbers
            - Strictly map the data you found to the given JSON Schema without modification or omissions.
            **Hierarchy Preservation:**:
            - Maintain proper parent-child relationships and follow the schema's hierarchical structure.
            **Correct Mapping of Attributes:**:
            - Map all the relevant information you find to its appropriate keys
            **JSON Format Compliance:**:
            - In your answer follow the JSON Format strictly !
            - If your answer doesn't conform to the JSON Format or is incompatible with the provided JSON schema, the output will be disgarded !
            
            Write your reasoning below here inside think tags and once you are done thinking, provide your answer in the described format !"""

        res = await RAGService.llm_service.query(prompt, sampling_params)
        
        parsed_json = None # prevents UnboundLocalError !
        try:
            answer_json = re.search(r"```json\s*(.*?)\s*```", res, re.DOTALL).group(1)
            parsed_json = json.loads(answer_json)
            jsonschema.validate(parsed_json, json_schema)
            return parsed_json, doc_type, json_schema, True
        except jsonschema.exceptions.ValidationError as e:
            return parsed_json, doc_type, json_schema, False
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Failed to extract or parse JSON from model response: {e}")
            return None, doc_type, json_schema, False
    

    def _identify_doc_type(self, paragraphs):
        from collections import defaultdict

        # Precompute schema embeddings
        schema_embeddings = {
            doc_type: RAGService.embedding_model.encode(json.dumps(schema))
            for doc_type, schema in self.doc_schemata.items()
        }

        vote_counts = defaultdict(int)
        valid_vote_count = 0

        for paragraph in paragraphs:
            text_embedding = RAGService.embedding_model.encode(paragraph)

            # Compare paragraph to all schema embeddings
            similarity_scores = {
                doc_type: util.pytorch_cos_sim(text_embedding, schema_embedding).item()
                for doc_type, schema_embedding in schema_embeddings.items()
            }

            # Find best match
            best_type, best_score = max(similarity_scores.items(), key=lambda x: x[1])

            print(f"Paragraph: {paragraph[:60]}... → Best Match: {best_type} ({best_score:.4f})")

            # Count vote only if above threshold
            if 0.5 <= best_score:
                vote_counts[best_type] += 1
                valid_vote_count += 1

        if valid_vote_count == 0:
            return None, None  # No reliable match

        # Normalize vote counts by valid votes
        normalized_scores = {
            doc_type: count / valid_vote_count
            for doc_type, count in vote_counts.items()
        }

        # Select document type with highest normalized score
        final_type, max_normalized_score = max(normalized_scores.items(), key=lambda x: x[1])
        
        # If a document's best potential type didn't score above 30% of all the document's paragraphs, it is too ambigous to categorize.
        if max_normalized_score < 0.3:
            return None, None

        return self.doc_schemata[final_type], final_type
    
    
    async def rag_query(self, question, n_results, user_access_role):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.
        
        Args:
            question (str): The user's natural language question.
            n_results (int): Number of documents to retrieve
            user_access_role (str): User's access role for filtering documents
        
        Yields:
            str: Streamed LLM response chunks
        """
        # Retrieve relevant documents
        docs_data = self.find_docs(question, n_results, user_access_role).data
        
        # Process documents and prepare summaries
        doc_summaries, doc_types, doc_sources_map = await self._summarize_docs(docs_data, question)
        
        async for chunk in self._query_llm(question, doc_summaries, doc_types, doc_sources_map, user_access_role):
            yield chunk
        
    
    async def _query_llm(self, question, doc_summaries, doc_types, doc_sources_map, user_access_role, allow_mongo_db_query=True):
        """ Generates the prompt and queries the LLM """
        # Build the initial prompt
        prompt = self._build_prompt(question, doc_summaries, doc_types, allow_mongo_db_query)
        
        # First LLM query - watch for mongo_json commands
        answer_buffer = ""
        mongo_query = None
        async for chunk in RAGService.llm_service.query(prompt, stream=True):
            answer_buffer += chunk
            yield chunk
            
            if allow_mongo_db_query and chunk.contains('`'):
                # Check for complete mongo_json block
                mongo_match = re.search(r"```mongo_json\s*(.*?)\s*```", answer_buffer, re.DOTALL)
                if mongo_match:
                    mongo_query = mongo_match.group(1)
                    break  # Stop the first generation
        
        # If we found a mongo query, execute it and do second LLM call
        if mongo_query and allow_mongo_db_query:
            # Execute MongoDB query
            mongo_result = self.mongo_db_connector.run(mongo_query, user_access_role)
            if mongo_result:
                # Build follow-up prompt with MongoDB results
                follow_up_prompt = (
                    f'{question}\n\nUse the following texts to briefly and precisely answer the previous question in a concise manner:\n\n'
                    f'Document summaries:\n\n{"\n\n\n".join(doc_summaries)}\n\n\n'
                    f'Here is a helpful database query and respone: \n\n{json.dumps(mongo_result, indent=2)}\n\n\n'
                    f'Please answer the question "{question}" briefly and precisely using all the available information.'
                )
                
                # Stream the second LLM response
                async for chunk in RAGService.llm_service.query(follow_up_prompt, stream=True):
                    yield chunk
            else:
                # The LLM provided a invalid database query
                async for chunk in self._query_llm(question, doc_summaries, doc_types, doc_sources_map, user_access_role, allow_mongo_db_query=False):
                    yield chunk
        
        # Stream the document sources string at the very end
        yield self._generate_source_references_str(doc_sources_map)
    
    
    async def _summarize_docs(self, docs_data, question):
        """Process documents concurrently to generate summaries and collect metadata."""
        doc_types = set()
        doc_sources_map = defaultdict(set)
        summarize_tasks = []
        
        for doc_data in docs_data:
            doc_id = doc_data['docId']
            page_num = doc_data['pageNum']
            doc_types.add(doc_data['docType'])
    
            if page_num is None:
                doc_sources_map[doc_id] = None
            else:
                doc_sources_map[doc_id].add(page_num)
            
            summarize_tasks.append(self._load_and_summarize_doc(doc_id, question))
    
        # Run all LLM summaries concurrently
        doc_summaries = await asyncio.gather(*summarize_tasks)
        return doc_summaries, doc_types, doc_sources_map
    
    
    async def _load_and_summarize_doc(self, doc_id, question):
        # Loads and summarizes a single document.
        txt_path = f"./companies/{self.company_id}/docs/{doc_id}.txt"
        async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
            doc_text = await f.read()
        
        summarize_prompt = f'{doc_text}\n\nSummarize all the relevant information and facts needed to answer the following question from the previous text:\n\n{question}'
        summary = await RAGService.llm_service.query(summarize_prompt)
        return re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    
    
    def _build_prompt(self, question, doc_summaries, doc_types, thinking_model=True, allow_mongo_db_query=True):
        """Construct the final prompt for the LLM."""
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the previous question in a concise manner:\n"
        for doc_summary in doc_summaries:
            prompt += '\n\n' + doc_summary
    
        # Attach MongoDB schema information
        if allow_mongo_db_query:
            prompt += '\n\n\nYou have read-only access to the MongoDB `docs` collection. All the documents in it have a doc_type field. These are the JSON schemata for each doc_type:'
        
            for doc_type in doc_types:
                prompt += f'\n\Doc_type {doc_type}: {json.dumps(self.doc_schemata[doc_type])}'
            
            prompt += '\n\nIf you need to query the MongoDB, write a JSON query in tags like so: ```mongo_json YOUR_QUERY ```.'
            if thinking_model:
                prompt += '\nWrite your final mongoDB json command with its tags inside your think tags. Like so: <think> YOUR THOUGHTS ```mongo_json YOUR_QUERY ``` </think>.'
            
        return prompt
    
    
    def _generate_source_references_str(self, doc_sources_map):
        """Generate the source references string."""
        sources_info = 'Consult these documents for more detail:\n'
        for doc_id, pages in doc_sources_map.items():
            try:
                doc_pseudo_path = self.json_db.find_one({ '_id': doc_id }).get('path')
            finally:
                doc_pseudo_path = f"Document with ID {doc_id}"
                
            sources_info += doc_pseudo_path
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'
        return sources_info
        

    def _merge_with_base_schema(self, json_schema):
        return {
            "type": "object",
            "properties": {**json_schema.get("properties", {}), **BASE_DOC_SCHEMA["properties"]},
            "required": list(set(BASE_DOC_SCHEMA["required"] + json_schema.get("required", [])))
        }


# company = 'MyCompany'
# rag = RAGService(company)
# print(rag.add_doc(f'./companies/{company}/uploads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf', 'CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf').detail)
# print(rag.add_doc(f'./companies/{company}/uploads/AlexanderMeetingDiogines.pdf', 'AlexanderMeetingDiogines.pdf').detail)
# rag.add_doc('_On the usefulness of context data.pdf')

# async def main1():
#     question = 'Did Charlemagne ever meet Alexander the Great ?'
#     relevant_docs_data = rag.find_docs(question, 5)
#     answer = await rag.query_llm(question=question, docs_data=relevant_docs_data)
#     print("\nAnswer:\n", answer.data['llm_response'])


# def main2():
#     question = 'What context data was used ?'
#     relevant_docs_data = rag.find_docs(question, 5)
#     answer = RAGService.query_llm(question=question, docs_data=relevant_docs_data)
#     print("\nAnswer:\n", answer)

# main1()

# import asyncio
# # Run the async function
# asyncio.run(main1())
# asyncio.run(main2())