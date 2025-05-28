import os
import re
import json
import torch
import logging
from utils import get_env_var, get_company_path, data_path

from filelock import FileLock

import jsonschema
from fastapi import HTTPException
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

from pymongo import MongoClient
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from sentence_transformers import SentenceTransformer, util # Pretrained model to convert text into numerical vectors (embeddings)

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
        'path': {'type': 'string'},
        'docType': {'type': ['string', 'null']},
        'accessGroups': {
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'minItems': 1
        }
    },
    'required': ['id', 'path', 'docType', 'accessGroups']
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
        self.doc_path_classifier = DocPathClassifier(company_id)

        self.schemata_embeddings = None
        self.schemata_path = get_company_path(company_id, 'doc_schemata.json')
        if os.path.exists(self.schemata_path):
            schemata_path = self.schemata_path
        else:
            schemata_path = os.path.join(data_path, 'default_doc_schemata.json')

        with open(schemata_path, 'r', errors='ignore') as f:
            self.doc_schemata = json.loads(f.read())

        # Create or connect to database
        client = MongoClient(MONGO_DB_URL)
        self.docs_db = client[company_id]['docs']


    def add_json_schema_type(self, doc_type, json_schema, user):
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
        if int(LLM_MAX_TEXT_LEN * 0.75) < len(doc_schemata_tokens):
            raise HTTPException(409, "Too many json schemata are registered at this company. The LLM will not be able to properly extract data for so many classes.")
        
        lock = FileLock(self.schemata_path)
        with lock:
            with open(self.schemata_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_schemata_str)
        
        if self.schemata_embeddings:
            self._update_schemata_embeddings()
        
        return OKResponse(f"Successfully added new JSON schema for {doc_type}", json_schema)


    def delete_json_schema_type(self, doc_type, user):
        user.assert_admin()
        
        if self.docs_db.find_one({ 'doc_type': doc_type }):
            raise HTTPException(409, f"Cannot delete schema '{doc_type}' because it was already used to extract a document.")
        
        lock = FileLock(self.schemata_path)
        with lock:
            del self.doc_schemata[doc_type]
            with open(self.schemata_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(json.dumps(self.doc_schemata))
        
        if self.schemata_embeddings:
            self._update_schemata_embeddings()
        
        return OKResponse(f"Successfully deleted JSON schema for '{doc_type}'")
    

    async def create_doc(self, source_path, doc_data, force_ocr, allow_override, user):
        doc_id = doc_data.get('id')
        if not doc_id:
            raise HTTPException(400, "docData must contain an 'id'")
        
        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        if not allow_override and os.path.exists(txt_path):
            raise HTTPException(409, f"Doc {doc_id} already exists and override was disallowed !")
        
        doc_type = doc_data.get('docType')
        if doc_type and not doc_type in self.doc_schemata.keys():
            raise HTTPException(409, f"No doc schema found for doc type '{doc_type}'. Add the schema first with POST /documentSchemata")
        
        # Validate user access
        try:
            user.has_doc_access(doc_id)
        except DocumentNotFoundError as e:
            pass # Expeceted behavior

        doc_data['accessGroups'] = self.access_manager.validate_new_access_groups(doc_data.get('accessGroups'))

        paragraphs = RAGService.doc_extractor.extract_paragraphs(source_path, force_ocr)
        doc_text = '\n\n'.join(paragraph for _, paragraph in paragraphs)
        
        # Classify the pseudo path (it is only used as a tool for users to organise themselves and has nothing to do with the file location)
        if not doc_data.get('path'):
            # Classify Document into a path if non existant
            file_name = os.path.basename(source_path)
            doc_data['path'] = self.doc_path_classifier.classify_doc(doc_text) + '/' + file_name

        # Extract JSON
        extracted_doc_data, doc_type, doc_schema, is_extract_valid = await self.extract_json(paragraphs, doc_type)

        if is_extract_valid:
            # Build final schema
            doc_schema = self._merge_with_base_schema(doc_schema)
            # Overwrite extracted data with uploaded data
            doc_data = { **extracted_doc_data, **doc_data }
            doc_data['docType'] = doc_type
        else:
            doc_schema = BASE_DOC_SCHEMA
            doc_data['docType'] = None
        
        print('FINAL DOC DATA:', doc_data)
        print('FINAL DOC SCHEMA:', doc_schema)

        # Check if doc_data is valid and insert into DB
        jsonschema.validate(doc_data, doc_schema)
        self.docs_db.replace_one({ '_id': doc_id }, doc_data, upsert=True) # Create doc or override existant doc

        # In case of override, remove all old embedding entries
        self.vector_db.delete(where={ 'doc_id': doc_id })
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
        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_text)

        return OKResponse(f"Successfully processed Document {doc_id}", doc_data)


    def update_doc_data(self, doc_data, merge_existing, user):
        doc_id = doc_data.get('id')
        if not doc_id:
            raise HTTPException(400, "docData must contain an 'id'")
        
        old_doc = user.has_doc_access(doc_id)
        
        old_doc_type = old_doc.get('docType')
        merged_doc = { **old_doc, **doc_data }
        
        doc_type = merged_doc.get('docType')
        if not doc_type and not old_doc_type:
            raise HTTPException(400, "No docType defined.")

        if doc_type != old_doc_type:
            merge_existing = False
        
        doc_schema = self.doc_schemata.get(doc_type)
        if not doc_schema:
            raise HTTPException(422, "Unknown doc_type. Register the JSON schema with POST /documentSchemata first.")
        doc_schema = self._merge_with_base_schema(doc_schema)
        
        updated_doc = merged_doc if merge_existing else doc_data
        
        # Validate user access
        if 'accessGroups' not in doc_data:
            if not merge_existing:
                raise HTTPException(400, "The new document must contain an accessGroups field")
            updated_doc['accessGroups'] = old_doc.get('accessGroups')
        else:
            # It exists in doc_data, so validate whether it's None or a value
            updated_doc['accessGroups'] = self.access_manager.validate_new_access_groups(doc_data.get('accessGroups'))
        
        # If merge wasn't allowed and doc_data is missing a path, take the old one
        if not updated_doc.get('path'):
            updated_doc['path'] = old_doc['path']
        
        jsonschema.validate(updated_doc, doc_schema)
        self.docs_db.replace_one({ '_id': doc_id }, updated_doc)
        
        return OKResponse(f"Successfully updated Document {doc_id}", updated_doc)


    def get_doc(self, doc_id, user):
        doc = user.has_doc_access(doc_id)

        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        with open(txt_path, 'r', errors='ignore') as f:
            doc_text = f.read()
        
        doc['text'] = doc_text
        
        return OKResponse(f"Successfully retrieved Document {doc_id}", doc)
        
    
    def delete_doc(self, doc_id, user):
        # Validate user access
        user.has_doc_access(doc_id)

        # Delete from vector DB
        self.vector_db.delete(where={ 'doc_id': doc_id })

        # Delete file
        txt_path = get_company_path(self.company_id, f'docs/{doc_id}.txt')
        os.remove(txt_path) # TODO: Check if this raises an exception, it should

        # Delete from json DB
        doc_data = self.docs_db.delete_one({ '_id': doc_id })
        if not doc_data:
            raise HTTPException(404, f"Doc {doc_id} doesn't exist and thus couldn't be removed.")

        return OKResponse(f"Successfully deleted Document {doc_id}", doc_data)
    

    def find_docs(self, message, n_results, user):
        """
        Retrieves the most relevant document chunks for a given message using semantic search between message and known paragraphs

        Args:
            message (str): User's query or message.
            n_results (int): Number of top-matching results to return.

        Returns:
            list: Metadata entries (doc_id, page_num and doc_type) for the top matches.
        """
        # Convert user message into an embedding vector
        message_embedding = RAGService.embedding_model.encode(message).tolist()
        # Perform semantic search in ChromaDB (based on embedding similarity between message and the paragraphs of all document)
        results = self.vector_db.query(query_embeddings=[message_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        
        valid_docs_data = set()
        for doc_data in nearest_neighbors:
            try:
                doc_id = doc_data['id']
                user.has_doc_access(doc_id)
            except DocumentNotFoundError:
                logger.warning(f"Corrupt data. VectorDB is referencing a missing doc with id {doc_id} for company {self.company_id} !")
                continue
            except InsufficientAccessError:
                continue
            
            if not any(lambda doc: doc == doc_data for doc in valid_docs_data):
                valid_docs_data.append(doc_data)

        return OKResponse(f"Found {len(valid_docs_data)}", valid_docs_data)


    async def extract_json(self, paragraphs, doc_type=None):
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
        paragraph_texts = map(lambda paragraph: paragraph[1], paragraphs)

        if doc_type:
            json_schema = self.doc_schemata.get(doc_type)
        else:
            json_schema, doc_type = self._identify_doc_type(paragraph_texts)

        if not json_schema:
            return None, doc_type, None, False

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.4,
            max_tokens=2048
            # stop=["\n\n", "\n", "Q:", "###"]
        )

        prompt = f"""This is a JSON Schema which you need to fill:

            {json.dumps(json_schema)}

            ### TASK REQUIREMENT
            You are a json extractor. You are tasked with extracting the relevant information needed to fill the JSON schema from the text below.
            
            {'/n/n'.join(paragraph_texts)}

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
            
            Provide your answer in the described format !!!"""
        
        print('JSON EXTRACT PROMPT:', prompt)
        
        answer = ""
        async for chunk in RAGService.llm_service.query(prompt, sampling_params=sampling_params, allow_chunking=False):
            answer += chunk

        print('JSON EXTRACT ANSWER:', answer)

        parsed_json = None # prevents UnboundLocalError !
        try:
            answer_json = re.search(r"```json\s*(.*?)\s*```", answer, re.DOTALL)
            if answer_json:
                answer_json = answer_json.group(1)
            else:
                raise ValueError("No JSON found in LLM response")
            
            print('EXTRACTED JSON:', answer_json)
            
            parsed_json = json.loads(answer_json)
            jsonschema.validate(parsed_json, json_schema)
            return parsed_json, doc_type, json_schema, True
        except jsonschema.exceptions.ValidationError as e:
            print('JSON Validation error:', e)
            return parsed_json, doc_type, json_schema, False
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            # print(f"Failed to extract or parse JSON from model response: {e}")
            return None, doc_type, json_schema, False
    

    def _identify_doc_type(self, paragraphs):
        from collections import defaultdict

        # Precompute schema embeddings
        if not self.schemata_embeddings:
            self._update_schemata_embeddings()

        vote_counts = defaultdict(int)
        valid_vote_count = 0

        for paragraph in paragraphs:
            paragraph_embedding = RAGService.embedding_model.encode(paragraph)

            # Compare paragraph to all schema embeddings
            similarity_scores = {
                doc_type: util.pytorch_cos_sim(
                    torch.tensor(paragraph_embedding).unsqueeze(0),
                    torch.tensor(schema_embedding).unsqueeze(0)
                ).item()
                for doc_type, schema_embedding in self.schemata_embeddings.items()
            }

            # Find best match
            best_type, best_score = max(similarity_scores.items(), key=lambda x: x[1])

            print(f"Paragraph: {paragraph[:60]}... → Best Match: {best_type} ({best_score:.4f})")

            # Count vote only if above threshold
            if 0.2 <= best_score:
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
        
        # If a document's best potential type didn't score above 40% of all the document's paragraphs, it is too ambigous to categorize.
        if max_normalized_score < 0.4:
            return None, None

        return self.doc_schemata[final_type], final_type

    
    def _merge_with_base_schema(self, json_schema):
        return {
            'type': 'object',
            'properties': {**json_schema.get('properties', {}), **BASE_DOC_SCHEMA['properties']},
            'required': list(set(BASE_DOC_SCHEMA['required'] + json_schema.get('required', [])))
        }
    
    
    def _update_schemata_embeddings(self):
        self.schemata_embeddings = {
            doc_type: RAGService.embedding_model.encode(json.dumps(schema))
            for doc_type, schema in self.doc_schemata.items()
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