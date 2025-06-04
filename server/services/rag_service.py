import os
import re
import json
import uuid
import logging
from utils import get_env_var, get_company_path, data_path, safe_async_read, safe_async_write

import jsonschema
from fastapi import HTTPException
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

from pymongo import MongoClient
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from sentence_transformers import SentenceTransformer # Pretrained model to convert text into numerical vectors (embeddings)

from services.access_manager import get_access_manager
from vllm import SamplingParams
from services.vllm_service import LLMService, tokenizer, LLM_MAX_TEXT_LEN
from services.doc_extractor import DocExtractor
from services.doc_path_classifier import DocPathClassifier

logger = logging.getLogger(__name__)

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph

BASE_DOC_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'docType': {'type': ['string', 'null']},
        'accessGroups': {
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'minItems': 1
        }
    },
    'required': ['id', 'docType', 'accessGroups']
}


class RAGService:
    # Initialize persistent vector database (ChromaDB)
    vector_db = chromadb.PersistentClient(path=os.path.join(data_path, 'databases', 'chroma_db'))
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
        
        # TODO: Add a classifer trained on the docTypes
        try:
            self.doc_path_classifier = DocPathClassifier(company_id)
        except FileNotFoundError:
            self.doc_path_classifier = None

        self.schemata_path = get_company_path(company_id, 'doc_schemata.json')
        if os.path.exists(self.schemata_path):
            schemata_path = self.schemata_path
        else:
            schemata_path = os.path.join(data_path, 'default_doc_schemata.json')

        # No file lock needed, only one RAGService exists per schemata file
        with open(schemata_path, 'r', errors='ignore') as f:
            self.doc_schemata = json.loads(f.read())

        # Create or connect to database
        client = MongoClient(MONGO_DB_URL)
        self.docs_db = client[company_id]['docs']


    async def add_doc_schema(self, doc_type, json_schema, user):
        user.assert_admin()
        
        if doc_type in self.doc_schemata.keys():
            raise HTTPException(409, f"The document type {doc_type}, is already used by another JSON schema.")
        
        json_type = json_schema.get('type')
        if not json_type or json_type != 'object':
            raise HTTPException(400, "The JSON schema must be an object type")

        jsonschema.Draft7Validator.check_schema(json_schema) # will raise jsonschema.exceptions.SchemaError if invalid
        
        self.doc_schemata[doc_type] = json_schema
        doc_schemata_str = json.dumps(self.doc_schemata)
        doc_schemata_tokens = tokenizer.encode(doc_schemata_str, add_special_tokens=False)
        if int(LLM_MAX_TEXT_LEN * 0.5) < len(doc_schemata_tokens):
            raise HTTPException(409, "Too many json schemata are registered at this company. The LLM will not be able to properly extract data for so many classes.")
        
        await safe_async_write(self.schemata_path, doc_schemata_str)
        
        return OKResponse(f"Successfully added new JSON schema for {doc_type}", json_schema)


    def get_doc_schemata(self):
        return OKResponse(f"Successfully retrieved JSON schemas", self.doc_schemata)


    async def delete_doc_schema(self, doc_type, user):
        user.assert_admin()
        
        if self.docs_db.find_one({ 'doc_type': doc_type }):
            raise HTTPException(409, f"Cannot delete schema '{doc_type}' because it was already used to extract a document.")
        
        doc_schema = self.doc_schemata.get(doc_type)
        if not doc_schema:
            raise HTTPException(404, f"No schema for document type {doc_type} found.")
        
        del self.doc_schemata[doc_type]
        await safe_async_write(self.schemata_path, json.dumps(self.doc_schemata))
        
        logger.info(f"User '{user.id}' at '{user.company_id}' deleted doc schema '{doc_type}'")
        
        return OKResponse(f"Successfully deleted JSON schema for '{doc_type}'", doc_schema)
    

    async def create_doc(self, source_path, doc_data, force_ocr, allow_override, user):
        doc_id = doc_data['id']
        
        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        if not allow_override and os.path.exists(txt_path):
            raise HTTPException(409, f"Doc {doc_id} already exists and override was disallowed !")
        
        chosen_doc_type = doc_data.get('docType')
        if chosen_doc_type:
            chosen_doc_schema = self.doc_schemata.get(chosen_doc_type)
            if not chosen_doc_schema:
                raise HTTPException(409, f"No doc schema found for doc type '{chosen_doc_type}'. Add the schema first with POST /documentSchemata")
        
        # Validate user access
        try:
            user.has_doc_access(doc_id)
        except DocumentNotFoundError as e:
            pass # Expeceted behavior

        doc_data['accessGroups'] = self.access_manager.validate_new_access_groups(doc_data.get('accessGroups'))

        paragraphs = RAGService.doc_extractor.extract_paragraphs(source_path, force_ocr)
        doc_text = '\n\n'.join(paragraph for _, paragraph in paragraphs)
    
        # Classify the pseudo path (it is only used as a tool for users to organise themselves and has nothing to do with the file location)
        if self.doc_path_classifier and not doc_data.get('path'):
            # Classify Document into a path if non existant
            file_name = os.path.basename(source_path)
            doc_data['path'] = self.doc_path_classifier.classify_doc(doc_text) + '/' + file_name
            
        # Retrain the classifier based on the exsting data, every time a human selects a doc type manually
        # if chosen_doc_type and 500 < self.docs_db.countDocuments({}):
        #     txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        #     text = await safe_async_read(txt_path)
        #     self.doc_type_classifier.train(texts, doc_types) # TODO: Add train function

        # Extract JSON
        extracted_doc_data, doc_type, doc_schema, is_extract_valid = await self.extract_json(doc_text, chosen_doc_type)

        if is_extract_valid:
            # Build final schema
            doc_schema = self._merge_with_base_schema(doc_schema)
            # Overwrite extracted data with uploaded data
            doc_data = { **extracted_doc_data, **doc_data }
            doc_data['docType'] = doc_type
        else:
            doc_data['docType'] = chosen_doc_type
            doc_schema = self._merge_with_base_schema(chosen_doc_schema)

        # Check if doc_data is valid and insert into DB
        jsonschema.validate(doc_data, doc_schema)
        self.docs_db.replace_one({ '_id': doc_id }, doc_data, upsert=True) # Create doc or override existant doc

        # In case of override, remove all old embedding entries
        self.vector_db.delete(where={ 'docId': doc_id })
        for page_num, paragraph in paragraphs:
            # Convert paragraph into an embedding (vector representation)
            block_embedding = RAGService.embedding_model.encode(paragraph)

            # Store the paragraph's metadata in the vector DB using its embedding
            paragraph_data = {
                'docId': doc_id,
                'pageNum': page_num
            }
            self.vector_db.upsert(
                embeddings=[block_embedding.tolist()],
                ids=[str(hash(str(block_embedding)))],
                metadatas=[{k: v for k, v in paragraph_data.items() if v is not None}] # Filter None types, chromaDB forbidds them
            )

        # Save the extracted content in .txt file
        await safe_async_write(txt_path, doc_text)
        
        doc_data['text'] = doc_text # Add doc_text to response
        
        return OKResponse(f"Successfully processed Document {doc_id}", doc_data)


    async def update_doc_data(self, doc_data, merge_existing, extract_json, user):
        doc_id = doc_data['id']
        old_doc = user.has_doc_access(doc_id)

        if merge_existing and extract_json:
            raise HTTPException(400, "Merging requires unchanged docTypes and extracting JSON requires changing docTypes. Only one can be done at once.")

        doc_type = doc_data.get('docType')
        old_doc_type = old_doc.get('docType')
        
        if merge_existing:
            if doc_type != old_doc_type:
                raise HTTPException(400, "Merging is only supoported when the docType remains unchanged.")

            merged_doc = { **old_doc, **doc_data }
        elif extract_json:
            if not doc_type:
                raise HTTPException(400, "JSON extraction requires a non-null docType.")
            elif doc_type == old_doc_type:
                raise HTTPException(400, "JSON extraction requires a changing docType.")
        
        updated_doc = merged_doc if merge_existing else doc_data
        
        # Validate user access
        access_groups = doc_data.get('accessGroups')
        if access_groups:
            # It exists in doc_data, so validate whether it's None or a value
            updated_doc['accessGroups'] = self.access_manager.validate_new_access_groups(access_groups)
        else:
            updated_doc['accessGroups'] = old_doc.get('accessGroups') if merge_existing else None
        
        if doc_type:
            doc_schema = self.doc_schemata.get(doc_type)
            if not doc_schema:
                raise HTTPException(422, "Unknown doc_type. Register the JSON schema with POST /documentSchemata first.")
            doc_schema = self._merge_with_base_schema(doc_schema)
        else:
            doc_schema = BASE_DOC_SCHEMA

        # If merge wasn't allowed and doc_data is missing a path, take the old one
        if merge_existing and not updated_doc.get('path'):
            updated_doc['path'] = old_doc.get('path')
        elif extract_json:
            # Extract JSON
            txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
            doc_text = await safe_async_read(txt_path)
            extracted_doc_data, _, _, is_extract_valid = await self.extract_json(doc_text, doc_type) # docType can never be None for extractJSON, thus the extracted schema and type will not change, thus can be ignored
            if is_extract_valid:
                updated_doc = { **extracted_doc_data, **updated_doc }

        jsonschema.validate(updated_doc, doc_schema)
        self.docs_db.replace_one({ '_id': doc_id }, updated_doc)

        logger.info(f"User '{user.id}' at '{user.company_id}' updated doc '{doc_id}'")
        
        return OKResponse(f"Successfully updated Document {doc_id}", updated_doc)


    async def get_doc(self, doc_id, user):
        doc = user.has_doc_access(doc_id)
        del doc['_id'] # Remove the mongoDB id from the response

        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        doc['text'] = await safe_async_read(txt_path)
        
        return OKResponse(f"Successfully retrieved Document {doc_id}", doc)
        
    
    def delete_doc(self, doc_id, user):
        # Validate user access
        user.has_doc_access(doc_id)

        # Delete from vector DB
        self.vector_db.delete(where={ 'docId': doc_id })

        # Delete file
        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        os.remove(txt_path) # TODO: Check if this raises an exception, it should

        # Delete from json DB
        res = self.docs_db.delete_one({ '_id': doc_id })
        if res.deleted_count == 0:
            raise HTTPException(404, f"Doc {doc_id} doesn't exist.")

        logger.info(f"User '{user.id}' at '{user.company_id}' deleted doc '{doc_id}'")

        return OKResponse(f"Successfully deleted Document {doc_id}")
    

    def find_docs(self, message, n_results, user):
        """
        Retrieves the most relevant document chunks for a given message using semantic search between message and known paragraphs

        Args:
            message (str): User's query or message.
            n_results (int): Number of top-matching results to return.

        Returns:
            list: Metadata entries (doc_id, page_num and doc_type) for the top paragraph matches.
        """
        # Convert user message into an embedding vector
        message_embedding = RAGService.embedding_model.encode(message).tolist()
        # Perform semantic search in ChromaDB (based on embedding similarity between message and the paragraphs of all document)
        results = self.vector_db.query(query_embeddings=[message_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        
        valid_docs_data = []
        for paragraph_data in nearest_neighbors:
            try:
                doc_id = paragraph_data['docId']
                user.has_doc_access(doc_id)
            except DocumentNotFoundError:
                logger.warning(f"Corrupt data. VectorDB is referencing a missing doc with id {doc_id} for company {self.company_id} !")
                continue
            except InsufficientAccessError:
                continue
            
            # Only add unique document references
            if not any(lambda doc: doc == paragraph_data for doc in valid_docs_data):
                valid_docs_data.append(paragraph_data)

        return OKResponse(f"Found {len(valid_docs_data)}", valid_docs_data)


    async def extract_json(self, doc_text, doc_type=None, sampling_params=None):
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
        if not sampling_params:
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.4,
                max_tokens=2048
                # stop=["\n\n", "\n", "Q:", "###"]
            )

        if not doc_type:
            # if self.doc_type_classifier:
            #     doc_type = self.doc_type_classifier.classify_doc(doc_text)
            # else:
            return await self.identify_and_extract_json(doc_text, sampling_params)

        json_schema = self.doc_schemata.get(doc_type)
        if not json_schema:
            return None, doc_type, None, False

        prompt = f"""This is a JSON Schema which you need to fill:

            {json.dumps(json_schema)}

            ### TASK REQUIREMENT
            You are a json extractor. You are tasked with extracting the relevant information needed to fill the JSON schema from the text below.
            
            {doc_text}

            ### STRICT RULES FOR GENERATING OUTPUT:
            **ALWAYS PROVIDE YOUR FINAL ANSWER**:
            - Always provide the filled JSON for the provided Schema
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
            
            Provide your final answer like this: ```json FILLED_JSON```
            If you want to make notes, do so without the markdown tags. Provide only your final answer with the markdown tags !!
            Start your answer or thought process here:"""
        
        answer = ""
        answer_json = None
        req_id = str(uuid.uuid4())
        async for chunk in RAGService.llm_service.query(prompt, req_id=req_id, sampling_params=sampling_params, allow_chunking=False):
            answer += chunk     
            if '`' in chunk:
                answer_json = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
                if answer_json:
                    answer_json = answer_json.group(1)
                    if await RAGService.llm_service.abort(req_id):
                        break

        parsed_json = None # prevents UnboundLocalError !
        try:
            if not answer_json:
                raise ValueError("No JSON found in LLM response")
            
            parsed_json = json.loads(answer_json)
            jsonschema.validate(parsed_json, json_schema)
            return parsed_json, doc_type, json_schema, True
        except jsonschema.exceptions.ValidationError as e:
            print('JSON Schema Validation error:', e)
            return parsed_json, doc_type, json_schema, False
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            # print(f"Failed to extract or parse JSON from model response: {e}")
            return None, doc_type, json_schema, False
    

    async def identify_and_extract_json(self, doc_text, sampling_params):
        prompt = f"""These are the available JSON Schemata:

            {json.dumps(self.doc_schemata)}

            ### TASK REQUIREMENT
            You are a json extractor. You are tasked with identifying the single most appropriate JSON for the text below and extracting the relevant information needed to fill your chosen JSON schema.
            
            {doc_text}

            ### STRICT RULES FOR GENERATING OUTPUT:
            **ALWAYS PROVIDE YOUR FINAL ANSWER**:
            - Always provide the filled JSON for your chosen Schema
            **JSON Schema Mapping:**:
            - Carefully check to find ALL relevant information in the text like ids, names, addresses, etc.
            - Sometimes ids my be called numbers
            - Strictly map the data you found to your chosen JSON Schema without modification or omissions.
            **Hierarchy Preservation:**:
            - Maintain proper parent-child relationships and follow the schema's hierarchical structure.
            **Correct Mapping of Attributes:**:
            - Map all the relevant information you find to its appropriate keys
            **JSON Format Compliance:**:
            - In your answer follow the JSON Format strictly !
            - If your answer doesn't conform to the JSON Format or is incompatible with the provided JSON schema, the output will be disgarded !
            
            Only chose one schema.
            Keep in mind you are working for {self.company_id} when selecting incoming and outgoing documents.
            Provide your FINAL answer like this: ```json {json.dumps({
                "schema name": 'YOUR CHOSEN SCHEMA NAME',
                "filled json": 'YOUR CHOSEN FILLED JSON',
            })}```
            If you want to make notes, do so without the markdown tags. Provide only your final answer with the markdown tags !!
            Start your answer or thought process here:"""
        
        answer = ""
        answer_json = None
        req_id = str(uuid.uuid4())
        async for chunk in RAGService.llm_service.query(prompt, req_id=req_id, sampling_params=sampling_params, allow_chunking=False):
            answer += chunk     
            if '`' in chunk:
                answer_json = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
                if answer_json:
                    answer_json = answer_json.group(1)
                    if await RAGService.llm_service.abort(req_id):
                        break

        doc_type = None # prevents UnboundLocalError !
        json_schema = None # prevents UnboundLocalError !
        parsed_json = None # prevents UnboundLocalError !
        try:
            if answer_json:
                answer_json = json.loads(answer_json)
                doc_type = answer_json.get('schema name')
                parsed_json = answer_json.get('filled json')
            else:
                raise ValueError("No JSON found in LLM response")
            
            if not doc_type:
                raise ValueError("No doc_type found in LLM response")

            json_schema = self.doc_schemata.get(doc_type)
            if not json_schema:
                raise ValueError("Unknown doc_type found in LLM response")
            
            jsonschema.validate(parsed_json, json_schema)
            return parsed_json, doc_type, json_schema, True
        except jsonschema.exceptions.ValidationError as e:
            print('JSON Schema Validation error:', e)
            return parsed_json, doc_type, json_schema, False
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            # print(f"Failed to extract or parse JSON from model response: {e}")
            return None, doc_type, json_schema, False

    
    def _merge_with_base_schema(self, json_schema):
        return {
            'type': 'object',
            'properties': {**json_schema.get('properties', {}), **BASE_DOC_SCHEMA['properties']},
            'required': list(set(BASE_DOC_SCHEMA['required'] + json_schema.get('required', [])))
        }


rag_service_cache = {}
def get_company_rag_service(company_id):
    rag_service = rag_service_cache.get(company_id)
    if rag_service:
        return rag_service
    
    rag_service = RAGService(company_id)
    rag_service_cache[company_id] = rag_service
    return rag_service

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