from collections import defaultdict
from sentence_transformers import SentenceTransformer, util # Pretrained model to convert text into numerical vectors (embeddings)
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
import re
import os
import json
from pymongo import MongoClient
from filelock import FileLock
import asyncio
import aiofiles
from api_responses import OKResponse
from vllm_service import LLMService
from doc_extractor import DocExtractor
from doc_path_classifier import DocPathClassifier
from path_normalizer import PathNormalizer
from vllm import SamplingParams
import jsonschema

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph

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
        self.company = company_id
        self.vector_db = RAGService.vector_db.get_or_create_collection(name=company_id) # TODO: Check if this raises an exception, it should
        # self.doc_path_classifier = DocPathClassifier(company_id)
        self.path_normalizer = PathNormalizer(company_id)

        self.schemata_path = f'./{company_id}/doc_schemata.json'
        with open(self.schemata_path, "r", encoding="utf-8", errors="ignore") as f:
            self.doc_schemata = json.loads(f.read())

        client = MongoClient("mongodb://localhost:27017/")

        # Create or connect to database
        self.json_db = client[company_id]


    def add_json_schema_type(self, json_schema, new_doc_type):
        if new_doc_type in self.doc_schemata.keys():
            raise ValueError(f'The document type {new_doc_type}, is already used by another JSON schema')

        jsonschema.Draft7Validator.check_schema(json_schema) # will raise jsonschema.exceptions.SchemaError if invalid
        
        lock = FileLock(self.schemata_path)
        with lock:
            self.doc_schemata[new_doc_type] = json_schema
            with open(self.schemata_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(json.dumps(self.doc_schemata))
        
        return OKResponse(f'Successfully added new JSON schema for {new_doc_type}')


    async def add_doc(self, source_path, access_groups, dest_path=None, doc_type=None):
        """
        Reads a PDF file from the given path and runs OCR to extract structured content.
        Saves the conent as a .txt file in the same location
        Then it adds its paragraphs and their embeddings to the vector database.

        Args:
            path (str): Path to the document file (PDF).

        Returns:
            None
        """
        paragraphs = RAGService.doc_extractor.extract_paragraphs(source_path)

        doc_text = '\n\n'.join(paragraph for _, paragraph in paragraphs)
        
        # Sort Document into path
        if dest_path is None:
            file_name = source_path.split('/')[-1] # Last element
            dest_path = self.doc_path_classifier.classify_doc(doc_text) + file_name

        # Extract JSON and create or overwrite in DB
        filled_json, doc_type, is_json_complete = await self.extract_json(doc_text, doc_type)
        doc_json_collection = self.json_db[doc_type]       
        if doc_type and is_json_complete:
            doc_json_collection.replace_one({ '_id': dest_path }, {
                    'doc_path': dest_path,
                    'access_groups': access_groups,
                    **filled_json
                }, upsert=True)
        else: # or delete a doc if it was overwritten with a new doc from which no data was extracted
            doc_json_collection.delete_one({ '_id': dest_path })

        # Load and process PDF file using OCR
        for page_num, paragraph in paragraphs:
            # Convert paragraph into an embedding (vector representation)
            block_embedding = RAGService.embedding_model.encode(paragraph)

            # Store the text and its embedding in the vector DB (ChromaDB)
            self.vector_db.upsert(
                embeddings=[block_embedding.tolist()],
                ids=[str(hash(str(block_embedding)))],
                metadatas=[{
                    'doc_path': dest_path,
                    'page_num': page_num,
                    'doc_type': doc_type
                }]
            )

        # Save the extracted content in .txt file
        txt_path = self.path_normalizer.get_full_company_path(dest_path) + '.txt'
        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_text)

        return OKResponse(detail=f'Successfully processed Document', data={
            'doc_path': dest_path,
            'extracted_json': filled_json,
            'is_json_extract_complete': is_json_complete
        })


    # TEST THIS FUNCTION
    def update_doc(self, old_path, new_path, new_json, new_access_groups):
        # Convert PDF paths to TXT paths
        old_txt_path = self.path_normalizer.get_full_company_path(old_path) + '.txt'
        new_txt_path = self.path_normalizer.get_full_company_path(new_path) + '.txt'
        
        if os.path.exists(new_txt_path):
            raise FileExistsError()

        # Perform the file move/rename
        os.rename(old_txt_path, new_txt_path) # TODO: Check if this raises an exception, it should
        
        # Update ChromaDB metadata
        doc_entries = self.vector_db.get(where={'doc_path': old_path})
        
        for doc_id in doc_entries["ids"]:
            # Preserve all existing metadata, only update the path
            current_metadata = self.vector_db.get(ids=[doc_id])["metadatas"][0]
            updated_metadata = {
                **current_metadata,  # Keep all existing metadata
                'doc_path': new_path  # Only update the path
            }
            self.vector_db.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )

        for collection_name in self.json_db.list_collection_names():
            collection = self.json_db[collection_name]
            old_doc = collection.find_one({ '_id': old_path })
            if old_doc is None:
                continue
            updated_doc = {
                    'doc_path': new_path,
                    'access_groups': new_access_groups,
                    **old_doc,
                    **new_json
                }
            jsonschema.validate(updated_doc, self.doc_schemata[collection])
            collection.insert_one({ '_id': new_path }, updated_doc) # Will raise a DuplicateKey error if already existant
            collection.delete_one({ '_id': old_path })

        return OKResponse(detail=f'Successfully updated Document {new_path}', data=new_path)


    # TEST THIS FUNCTION
    def delete_doc(self, path):
        txt_path = self.path_normalizer.get_full_company_path(path) + '.txt'
        
        # Delete file
        os.remove(txt_path) # TODO: Check if this raises an exception, it should
        
        # Delete from json DB
        for collection_name in self.json_db.list_collection_names():
            self.json_db[collection_name].delete_one({ '_id': path })

        # Delete from ector DB
        self.vector_db.delete(where={'doc_path': path})

        return OKResponse(detail=f'Successfully deleted Document {path}', data=path)
    

    def find_docs(self, question, n_results):
        """
        Retrieves the most relevant document chunks for a given question using semantic search between question and known paragraphs

        Args:
            question (str): User's query or question.
            n_results (int): Number of top-matching results to return.

        Returns:
            list: Metadata entries (paths and pages) for the top matches.
        """
        # Convert user question into an embedding vector
        question_embedding = RAGService.embedding_model.encode(question).tolist()
        # Perform semantic search in ChromaDB (based on embedding similarity between question and the paragraphs of all document)
        results = self.vector_db.query(query_embeddings=[question_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        return nearest_neighbors

    
    async def query_llm(self, question, docs_data, stream=False):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.

        Args:
            question (str): The user's natural language question.

        Returns:
            str: Final LLM-generated answer with source references.
        """
        async def _load_and_summarize_doc(rel_doc_path):
            txt_path = self.path_normalizer.get_full_company_path(rel_doc_path) + '.txt'
            async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
                doc_text = await f.read()

            summarize_prompt = f'{doc_text}\n\nSummarize all the relevant information and facts needed to answer the following question from the previous text:\n\n{question}'
            return await RAGService.llm_service.query(summarize_prompt)
        
        # Read and summarize all docs concurrently
        summarize_tasks = []
        doc_types = set()
        doc_sources_map = defaultdict(set)
        for doc_data in docs_data:
            rel_doc_path = doc_data['doc_path']
            page_num = doc_data['page_num']
            doc_types.add(doc_data['doc_type'])

            if page_num is None:
                doc_sources_map[rel_doc_path] = None
            else:
                doc_sources_map[rel_doc_path].add(page_num)
            
            summarize_tasks.append(_load_and_summarize_doc(rel_doc_path))

        # Compose source references
        sources_info = 'Consult these documents for more detail:\n'
        for doc_path, pages in doc_sources_map.items():
            sources_info += doc_path
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'
                
        # Run all LLM summaries concurrently via vLLM server
        doc_summaries = await asyncio.gather(*summarize_tasks)

        # Create prompt with summaries
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the question in a concise manner:\n"
        for doc_summary in doc_summaries:
            clean_summary = re.sub(r'<think>.*?</think>', '', doc_summary, flags=re.DOTALL)
            prompt += '\n\n' + clean_summary

        if not stream:
            # Attach MongoDB to prompt
            prompt += '\n\n\nYou have read-only access to the Mongo database. These are the JSON schemata for each MongoDB collection:'

            for doc_type in doc_types:
                prompt += f'\n\nCollection {doc_type}: {json.dumps(self.doc_schemata[doc_type])}'
            
            prompt += '\n\nIf you want to query the MongoDB, write a mongosh query in tags like so: ```mongosh YOUR COMMAND ```'

            # Final prompt to answer the question (still single prompt)
            answer = await RAGService.llm_service.query(prompt, stream=False)

            # Extract mongosh commands if existant
            mongosh_commands = re.search(r"```mongosh\s*(.*?)\s*```", answer, re.DOTALL).group(1)
            if mongosh_commands:
                self.mongosh_service.run(mongosh_commands)

            # Clean out any lingering <think> tags
            answer_text = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
            
            return OKResponse(data={ 'llm_response': sources_info + answer_text })
        
        # TODO: HOW TO ADD MONGO DB TO THIS ?!
        # Stream LLM response tokens over the WebSocket
        async for chunk in RAGService.llm_service.query(prompt, stream=True):
            yield chunk
    

    async def extract_json(self, text, doc_type=None):
        """
        Extracts a filled JSON object from LLM output based on a schema and checks if all fields are filled.

        Args:
            text (str): Input text to extract data from.
            json_schema (dict): The target JSON schema structure.

        Returns:
            tuple:
                - dict: The parsed JSON object.
                - bool: Whether all schema fields were filled.
        """
        if doc_type is None:
            json_schema, doc_type = RAGService.identify_doc_json(text)
        else:
            json_schema = self.doc_schemata[doc_type] # raises Error if doc_type is invalid

        if doc_type is None:
            return None, None, False

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

        try:
            answer_json = re.search(r"```json\s*(.*?)\s*```", res, re.DOTALL).group(1)
            parsed_json = json.loads(answer_json)
            jsonschema.validate(parsed_json, json_schema)
            return parsed_json, doc_type, True
        except jsonschema.exceptions.ValidationError as e:
            return parsed_json, doc_type, False
        except (IndexError, json.JSONDecodeError) as e:
            print(f"Failed to extract or parse JSON from model response: {e}")
            return None, doc_type, False
    

    @staticmethod
    def identify_doc_json(self, paragraphs):
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
        final_type = max(normalized_scores.items(), key=lambda x: x[1])[0]

        return self.doc_schemata[final_type], final_type


# company = 'MyCompany'
# rag = RAGService(company)
# print(rag.add_doc(f'./{company}/uploads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf', 'CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf').detail)
# print(rag.add_doc(f'./{company}/uploads/AlexanderMeetingDiogines.pdf', 'AlexanderMeetingDiogines.pdf').detail)
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