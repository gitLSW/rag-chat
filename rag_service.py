from collections import defaultdict
from sentence_transformers import SentenceTransformer  # Pretrained model to convert text into numerical vectors (embeddings)
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from chromadb.utils import embedding_functions
from openai import OpenAI as LLMApi  # Used to call local or remote large language models (LLMs)
import re
import os
from filelock import FileLock
from api_responses import OKResponse
from vllm_service import LLMService
from doc_extractor import DocExtractor

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph

# Start LLMBatchProcessor
# llm_service = LLMService()

class RAGService:
    # Initialize the Doc Extractor
    doc_extractor = DocExtractor()

    # Initialize persistent vector database (ChromaDB)
    db_client = chromadb.PersistentClient(path="chroma_db")

    llm_service = LLMService()

    # Load sentence embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def __init__(self, company):
        """
        Initializes the RAGService instance.

        - Sets up a persistent vector database using ChromaDB to store text embeddings.
        - Loads a pre-trained OCR model with specific detection and recognition architectures.
        - Configures OCR to group detected text into lines and paragraphs.
        - Connects to a local LLM API (e.g., via Ollama).
        - Loads a sentence transformer for generating vector embeddings of text.
        """
        self.company = company
        self.collection = RAGService.db_client.get_or_create_collection(name=company) # TODO: Check if this raises an exception, it should
        

    def add_doc(self, file_name):
        """
        Reads a PDF file from the given path and runs OCR to extract structured content.
        Saves the conent as a .txt file in the same location
        Then it adds its paragraphs and their embeddings to the vector database.

        Args:
            path (str): Path to the document file (PDF).

        Returns:
            None
        """

        # Load and process PDF file using OCR
        page_paragraphs = RAGService.doc_extractor.extract_paragraphs(f'./{self.company}/uploads/{file_name}')
        doc_text = ''
        for page_num, paragraph in page_paragraphs:
            doc_text += paragraph + '\n\n'

            # Convert paragraph into an embedding (vector representation)
            block_embedding = RAGService.embedding_model.encode(paragraph)

            # Store the text and its embedding in the vector DB (ChromaDB)
            self.collection.upsert(
                embeddings=[block_embedding.tolist()],
                ids=[str(hash(str(block_embedding)))],
                metadatas=[{
                    'doc_name': file_name,
                    'page_num': page_num
                }]
            )

        # Save the OCR extracted content in .txt file
        txt_path = f'./{self.company}/{file_name}.txt'
        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(doc_text)

        return OKResponse(detail=f'Successfully stored Document {txt_path}', data=txt_path)


    # TEST THIS FUNCTION
    def update_doc(self, old_file_name, new_file_name):
        # Convert PDF paths to TXT paths
        old_txt_path = f'./{self.company}/{old_file_name}.txt'
        new_txt_path = f'./{self.company}/{new_file_name}.txt'
        
        if os.path.exists(new_txt_path):
            raise FileExistsError()

        # Perform the file move/rename
        os.rename(old_txt_path, new_txt_path) # TODO: Check if this raises an exception, it should
        
        # Update ChromaDB metadata
        doc_entries = self.collection.get(where={'doc_name': old_file_name})
        
        for doc_id in doc_entries["ids"]:
            # Preserve all existing metadata, only update the path
            current_metadata = self.collection.get(ids=[doc_id])["metadatas"][0]
            updated_metadata = {
                **current_metadata,  # Keep all existing metadata
                'doc_name': new_file_name  # Only update the path
            }
            self.collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )

        return OKResponse(detail=f'Successfully updated Document {new_txt_path}', data=new_txt_path)


    # TEST THIS FUNCTION
    def delete_doc(self, file_name):
        txt_path = f'./{self.company}/{file_name}.txt'
        
        # Delete file
        os.remove(txt_path) # TODO: Check if this raises an exception, it should
        
        # Delete from DB
        self.collection.delete(where={'doc_name': file_name})

        return OKResponse(detail=f'Successfully deleted Document {txt_path}', data=txt_path)
    

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


    def query_llm(self, question, docs_data):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.

        Args:
            question (str): The user's natural language question.
            n_results (int): Number of document chunks to retrieve for context (default is 5).

        Returns:
            str: Final LLM-generated answer with source references.
        """
        # Group relevant documents and collect prompts to summarize them
        summarize_prompts = []
        doc_sources_map = defaultdict(set)
        for doc_data in docs_data:
            doc_name = doc_data['doc_name']
            page_num = doc_data['page_num']
            if page_num is None:
                doc_sources_map[doc_name] = None # Some docs have no pages
            else:   
                doc_sources_map[doc_name].add(page_num)

            # Extract and aggregate text from relevant documents
            doc_path = f'{self.company}/{doc_name}.txt'
            doc_text = RAGService._gather_docs_knowledge([doc_path])
            summarize_prompts.append(f'{doc_text}\n\nSummarize all the relevant information and facts needed to answer the following question from the previous text:\n\n{question}')
        
        # Have the LLM summarize the docs concurrently
        doc_summaries = RAGService.llm_service.query(summarize_prompts)

        # Build the final prompt using the summaries and query the LLM
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the question in a concise manner:\n\n\n"
        for doc_summary in doc_summaries:
            print(doc_summary)
            # Remove the <think> clause from each doc_summary and append into one big prompt
            prompt += re.sub(r'<think>.*?</think>', '', doc_summary, flags=re.DOTALL) + '\n\n'

        # Send prompt to local ollama LLM
        final_answer = RAGService.llm_service.query([prompt])[0]

        # Prepare final answer with source info
        sources_info = 'Consult these documents for more detail:\n'
        for doc_name, pages in doc_sources_map.items():
            sources_info += doc_name
            if pages is None:
                sources_info += '\n'
            else:
                sources_info += f' on pages {", ".join(map(str, sorted(pages)))}\n'

        # Strip out internal model markers like <think> tags
        answer_text = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL)

        return OKResponse(data={ 'llm_response': sources_info + answer_text })


    @staticmethod
    def _gather_docs_knowledge(txt_paths):
        """
        Extracts and combines text from the specified documents.

        Args:
            doc_paths (iterable): Set of file paths to load and extract text from.

        Returns:
            str: Combined textual content from all provided documents, separated clearly.
        """
        # Load and extract clean text from multiple document paths
        unique_paths = set(txt_paths)
        documents_text = []
        
        for path in unique_paths:
            # Read plain text content from converted .txt document
            with open(path + '.txt', 'r') as f:
                documents_text.append(f.read())
    
        # Separate different documents with more spacing
        return "\n\n\n\n\n\n\n\nNext Relevant Text:\n".join(documents_text)


rag = RAGService('MyCompany')
# print(rag.add_doc('CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf').detail)
# print(rag.add_doc('AlexanderMeetingDiogines.pdf').detail)
# rag.add_doc('_On the usefulness of context data.pdf')

def main1():
    question = 'Did Charlemagne ever meet Alexander the Great ?'
    relevant_docs_data = rag.find_docs(question, 5)
    answer = rag.query_llm(question=question, docs_data=relevant_docs_data)
    print("\nAnswer:\n", answer.data['llm_response'])


# def main2():
#     question = 'What context data was used ?'
#     relevant_docs_data = rag.find_docs(question, 5)
#     answer = RAGService.query_llm(question=question, docs_data=relevant_docs_data)
#     print("\nAnswer:\n", answer)

main1()

# import asyncio
# # Run the async function
# asyncio.run(main1())
# asyncio.run(main2())