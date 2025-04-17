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

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph

# Start LLMBatchProcessor
# llm_service = LLMService()

doc_extractor = DocExtractor()

class RAGService:
    # Initialize persistent vector database (ChromaDB)
    vector_db = chromadb.PersistentClient(path="chroma_db")

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
        self.collection = RAGService.vector_db.get_or_create_collection(name=company_id) # TODO: Check if this raises an exception, it should
        # self.doc_path_classifier = DocPathClassifier(company_id)
        self.path_normalizer = PathNormalizer(company_id)

        client = MongoClient("mongodb://localhost:27017/")

        # 📁 Create or connect to database
        self.json_db = client[company_id]

        
    
    # def add_file(self, source_path, dest_path):
    #     paragraphs = DocExtractor().extract_paragraphs(source_path)
    #     return self.add_doc(paragraphs, dest_path)


    def add_doc(self, source_path, dest_path=None, json_schema=None, doc_type_colection=None):
        """
        Reads a PDF file from the given path and runs OCR to extract structured content.
        Saves the conent as a .txt file in the same location
        Then it adds its paragraphs and their embeddings to the vector database.

        Args:
            path (str): Path to the document file (PDF).

        Returns:
            None
        """
        paragraphs = doc_extractor.extract_paragraphs(source_path)

        doc_text = '\n\n'.join(paragraph for _, paragraph in paragraphs)
        
        # Sort Document into path
        if dest_path is None:
            file_name = source_path.split('/')[-1] # Last element
            dest_path = self.doc_path_classifier.classify_doc(doc_text) + file_name

        # Load and process PDF file using OCR
        for page_num, paragraph in paragraphs:
            # Convert paragraph into an embedding (vector representation)
            block_embedding = RAGService.embedding_model.encode(paragraph)

            # Store the text and its embedding in the vector DB (ChromaDB)
            self.collection.upsert(
                embeddings=[block_embedding.tolist()],
                ids=[str(hash(str(block_embedding)))],
                metadatas=[{
                    'doc_path': dest_path,
                    'page_num': page_num
                }]
            )

        # Save the extracted content in .txt file
        txt_path = self.path_normalizer.get_full_company_path(dest_path) + '.txt'
        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_text)

        filled_json, doc_type_colection, is_json_complete = self.extract_json(doc_text, json_schema)
        self.json_db[doc_type_colection].insert_one(filled_json)

        return OKResponse(detail=f'Successfully processed Document', data={
            'doc_path': dest_path,
            'extracted_json': filled_json,
            'is_json_extract_complete': is_json_complete
        })


    # TEST THIS FUNCTION
    def update_doc(self, old_path, new_path):
        # Convert PDF paths to TXT paths
        old_txt_path = self.path_normalizer.get_full_company_path(old_path) + '.txt'
        new_txt_path = self.path_normalizer.get_full_company_path(new_path) + '.txt'
        
        if os.path.exists(new_txt_path):
            raise FileExistsError()

        # Perform the file move/rename
        os.rename(old_txt_path, new_txt_path) # TODO: Check if this raises an exception, it should
        
        # Update ChromaDB metadata
        doc_entries = self.collection.get(where={'doc_path': old_path})
        
        for doc_id in doc_entries["ids"]:
            # Preserve all existing metadata, only update the path
            current_metadata = self.collection.get(ids=[doc_id])["metadatas"][0]
            updated_metadata = {
                **current_metadata,  # Keep all existing metadata
                'doc_path': new_path  # Only update the path
            }
            self.collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )

        return OKResponse(detail=f'Successfully updated Document {new_path}', data=new_path)


    # TEST THIS FUNCTION
    def delete_doc(self, path):
        txt_path = self.path_normalizer.get_full_company_path(path) + '.txt'
        
        # Delete file
        os.remove(txt_path) # TODO: Check if this raises an exception, it should
        
        # Delete from DB
        self.collection.delete(where={'doc_path': path})

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
        results = self.collection.query(query_embeddings=[question_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        return nearest_neighbors
    
    
    async def query_llm(self, question, docs_data):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.

        Args:
            question (str): The user's natural language question.

        Returns:
            str: Final LLM-generated answer with source references.
        """
        
        async def load_and_summarize_doc(rel_doc_path):
            txt_path = self.path_normalizer.get_full_company_path(rel_doc_path) + '.txt'
            async with aiofiles.open(txt_path, mode='r', encoding='utf-8') as f:
                doc_text = await f.read()

            summarize_prompt = f'{doc_text}\n\nSummarize all the relevant information and facts needed to answer the following question from the previous text:\n\n{question}'
            return await RAGService.llm_service.query(summarize_prompt)

        # Read and summarize all docs concurrently
        summarize_tasks = []
        doc_sources_map = defaultdict(set)
        for doc_data in docs_data:
            rel_doc_path = doc_data['doc_path']
            page_num = doc_data['page_num']

            if page_num is None:
                doc_sources_map[rel_doc_path] = None
            else:
                doc_sources_map[rel_doc_path].add(page_num)
            
            summarize_tasks.append(load_and_summarize_doc(rel_doc_path))

        # Run all LLM summaries concurrently via vLLM server
        doc_summaries = await asyncio.gather(*summarize_tasks)

        # Step 3: Create final prompt
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the question in a concise manner:\n\n\n"
        for doc_summary in doc_summaries:
            clean_summary = re.sub(r'<think>.*?</think>', '', doc_summary, flags=re.DOTALL)
            prompt += clean_summary + '\n\n'

        # Step 4: Final prompt to answer the question (still single prompt)
        final_answer = await RAGService.llm_service.query(prompt)

        # Step 5: Compose source references
        sources_info = 'Consult these documents for more detail:\n'
        for doc_path, pages in doc_sources_map.items():
            sources_info += doc_path
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'

        # Clean out any lingering <think> tags
        answer_text = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL)

        return OKResponse(data={ 'llm_response': sources_info + answer_text })
    

    async def extract_json(self, text, json_schema=None):
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
        
        if json_schema is None:
            json_schema, doc_type_colection = RAGService.identify_doc_json(text)
        else:
            _, doc_type_colection = RAGService.identify_doc_json(text)

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
        except (IndexError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to extract or parse JSON from model response: {e}")

        # Helper function to check if all schema keys are present and filled
        def all_fields_filled(schema_part, json_part):
            if isinstance(schema_part, dict):
                for key, value in schema_part.items():
                    if key not in json_part or json_part[key] in [None, "", []]:
                        return False
                    if isinstance(value, dict) and isinstance(json_part[key], dict):
                        if not all_fields_filled(value, json_part[key]):
                            return False
            return True

        is_complete = all_fields_filled(json_schema, parsed_json)

        return parsed_json, doc_type_colection, is_complete
    

    @staticmethod
    def identify_doc_json(content):
        text_embedding = RAGService.embedding_model.encode(content)

        invoice_schema = {
            "invoiceNumber": "string",
            "invoiceDate": "string (ISO 8601 date)",
            "dueDate": "string (ISO 8601 date)",
            "supplierName": "string",
            "supplierAddress": "string",
            "supplierTaxID": "string",
            "customerName": "string",
            "customerAddress": "string",
            "customerTaxID": "string",
            "lineItems": [
                {
                "description": "string",
                "quantity": "number",
                "unitPrice": "number",
                "totalPrice": "number"
                }
            ],
            "subtotal": "number",
            "tax": "number",
            "totalAmount": "number",
            "currency": "string"
        }
        invoice_json_embedding = RAGService.embedding_model.encode(json.dumps(invoice_schema))

        delivery_note_schema = {
            "deliveryNoteNumber": "string",
            "deliveryDate": "string (ISO 8601 date)",
            "supplierName": "string",
            "supplierAddress": "string",
            "customerName": "string",
            "customerAddress": "string",
            "deliveryAddress": "string",
            "itemsDelivered": [
                {
                "description": "string",
                "quantity": "number",
                "unitOfMeasure": "string",
                "batchNumber": "string"
                }
            ],
            "deliveryMethod": "string",
            "vehicleNumber": "string",
            "receiverName": "string",
            "receiverSignature": "string (optional, can be a path to an image or note about presence)"
        }
        delivery_note_json_embedding = RAGService.embedding_model.encode(json.dumps(delivery_note_schema))

        invoice_similarity_score = util.pytorch_cos_sim(text_embedding, invoice_json_embedding).item()
        delivery_note_similarity_score = util.pytorch_cos_sim(text_embedding, delivery_note_json_embedding).item()

        print('FIND A THRESHHOLD OF RETURN NONE AND RETURN NONE')
        print('invoice_similarity_score:', invoice_similarity_score)
        print('delivery_note_similarity_score:', delivery_note_similarity_score)

        if delivery_note_similarity_score < invoice_similarity_score:
            return invoice_schema, 'invoice'
        else:
            return delivery_note_schema, 'delivery_note'


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