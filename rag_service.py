import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor  # OCR = Optical Character Recognition, extracts text from images or PDFs
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

# Configuration
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph
TEXT_DETECTION_MODEL = 'fast_base'  # Model for detecting where text is on the page
TEXT_RECOGNITION_MODEL = 'crnn_vgg16_bn'  # Model for recognizing characters within the detected text regions

DEVICE = torch.device("cuda:0") # The OCR must run on a gpu or it will seg fault.

class OCR:
    def __init__(self):
        # Load OCR model with text detection and recognition components
        self.ocr_model = ocr_predictor(det_arch=TEXT_DETECTION_MODEL,
                                       reco_arch=TEXT_RECOGNITION_MODEL,
                                       pretrained=True,
                                       assume_straight_pages=True,
                                       preserve_aspect_ratio=True).to(DEVICE)
        self.ocr_model.doc_builder.resolve_lines = True  # Group words into lines
        self.ocr_model.doc_builder.resolve_blocks = True  # Group lines into blocks (paragraphs)

    def read_pdf(self, pages):
        return self.ocr_model(pages)

# Start LLMBatchProcessor
llm_service = LLMService()

class RAGService:
    # Initialize OCR
    ocr_model = OCR()

    # Initialize persistent vector database (ChromaDB)
    db_client = chromadb.PersistentClient(path="chroma_db")

    # Load sentence embedding model
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def __init__(self, collection):
        """
        Initializes the RAGService instance.

        - Sets up a persistent vector database using ChromaDB to store text embeddings.
        - Loads a pre-trained OCR model with specific detection and recognition architectures.
        - Configures OCR to group detected text into lines and paragraphs.
        - Connects to a local LLM API (e.g., via Ollama).
        - Loads a sentence transformer for generating vector embeddings of text.
        """
        self.collection = RAGService.db_client.get_or_create_collection(name=collection) # TODO: Check if this raises an exception, it should
        

    def add_doc(self, path):
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
        pages = DocumentFile.from_pdf(path + '.pdf')
        doc = RAGService.ocr_model.read_pdf(pages)

        # Save the OCR extracted content in .txt file
        txt_path = path + '.txt'

        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f: # TODO: Check if this raises an exception, it should
                f.write(RAGService._convert_doc_data_to_text(doc))

        for page in doc.pages:
            for block in page.blocks:
                block_text = RAGService._convert_block_data_to_text(block)
                # Convert paragraph into an embedding (vector representation)
                block_embedding = RAGService.embedding_model.encode(block_text).tolist()

                # Store the text and its embedding in the vector DB (ChromaDB)
                self.collection.upsert(
                    embeddings=[block_embedding],
                    ids=[str(hash(str(block_embedding)))],
                    metadatas=[{
                        'path': path,
                        'page_num': page.page_idx + 1
                    }]
                )

        return OKResponse(detail=f'Successfully stored Document {path}', data=txt_path)


    # TEST THIS FUNCTION
    def update_doc(self, old_path, new_path):
        # Convert PDF paths to TXT paths
        old_txt_path = old_path + '.txt'
        new_txt_path = new_path + '.txt'
        
        # Perform the file move/rename
        os.rename(old_txt_path, new_txt_path) # TODO: Check if this raises an exception, it should
        
        # Update ChromaDB metadata
        doc_entries = self.collection.get(where={"path": old_path})
        
        for doc_id in doc_entries["ids"]:
            # Preserve all existing metadata, only update the path
            current_metadata = self.collection.get(ids=[doc_id])["metadatas"][0]
            updated_metadata = {
                **current_metadata,  # Keep all existing metadata
                "path": new_path  # Only update the path
            }
            self.collection.update(
                ids=[doc_id],
                metadatas=[updated_metadata]
            )

        return OKResponse(detail=f'Successfully updated Document {new_path}', data=new_txt_path)


    # TEST THIS FUNCTION
    def delete_doc(self, path):
        txt_path = path + '.txt'
        
        # Delete file
        os.remove(txt_path) # TODO: Check if this raises an exception, it should
        
        # Delete from DB
        self.collection.delete(where={'path': path})

        return OKResponse(detail=f'Successfully deleted Document {path}', data=txt_path)
    

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


    def query_llm(question, docs_data):
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
            doc_path = doc_data['path']
            doc_sources_map[doc_path].add(doc_data['page_num'])

            # Extract and aggregate text from relevant documents
            doc_text = RAGService._gather_docs_knowledge([doc_path])
            summarize_prompts.append(f'{doc_text}\n\nSummarize all the relevant information and facts needed to answer the following question from the previous text:\n\n{question}')
        
        # Have the LLM summarize the docs concurrently
        doc_summaries = llm_service.query(summarize_prompts)

        # Build the final prompt using the summaries and query the LLM
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the question in a concise manner:\n\n\n"
        for doc_summary in doc_summaries:
            print(doc_summary)
            # Remove the <think> clause from each doc_summary and append into one big prompt
            prompt += re.sub(r'<think>.*?</think>', '', doc_summary, flags=re.DOTALL) + '\n\n'

        # Send prompt to local ollama LLM
        final_answer = llm_service.query([prompt])[0]

        # Prepare final answer with source info
        sources_info = 'Consult these documents for more detail:\n'
        for path, pages in doc_sources_map.items():
            sources_info += f'{path}.pdf on pages {", ".join(map(str, sorted(pages)))}\n'

        # Strip out internal model markers like <think> tags
        answer_text = re.sub(r'<think>.*?</think>', '', final_answer, flags=re.DOTALL)

        return OKResponse(data={ 'llm_response': sources_info + answer_text })
    

    def _convert_block_data_to_text(block_data):
        lines = []
        for line in block_data.lines:
            # Join words with spaces and strip trailing whitespace
            line_text = " ".join(word.value for word in line.words).rstrip()
                        
            # Remove trailing hyphen if present
            if line_text.endswith('-'):
                line_text = line_text[:-1].rstrip()  # Remove hyphen and any remaining whitespace
            
            lines.append(line_text)
            
        # Join processed lines with single space
        return " ".join(lines)


    def _convert_doc_data_to_text(doc_data):
        """
        Converts an OCR document into plain text.

        Args:
            block_data: A doc object from docTR OCR output.

        Returns:
            str: Text content from the document.
        """
        blocks = []
        for page in doc_data.pages:
            for block in page.blocks:
                block_text = RAGService._convert_block_data_to_text(block)
                blocks.append(block_text)
    
        # Separate text blocks within a document
        return "\n\n".join(blocks)


    def _gather_docs_knowledge(doc_paths):
        """
        Extracts and combines text from the specified documents.

        Args:
            doc_paths (iterable): Set of file paths to load and extract text from.

        Returns:
            str: Combined textual content from all provided documents, separated clearly.
        """
        # Load and extract clean text from multiple document paths
        unique_paths = set(doc_paths)
        documents_text = []
        
        for path in unique_paths:
            # Read plain text content from converted .txt document
            with open(path + '.txt', 'r') as f:
                documents_text.append(f.read())
    
        # Separate different documents with more spacing
        return "\n\n\n\n\n\n\n\nNext Relevant Text:\n".join(documents_text)


rag = RAGService('MyCompany')
# rag.add_doc('/home/lsw/Downloads/CHARLEMAGNE_AND_HARUN_AL-RASHID')
# rag.add_doc('/home/lsw/Downloads/AlexanderMeetingDiogines')
# rag.add_doc('/home/lsw/Downloads/_ALL THESIS/_On the usefulness of context data.pdf')

def main1():
    question = 'Did Charlemagne ever meet Alexander the Great ?'
    relevant_docs_data = rag.find_docs(question, 5)
    answer = RAGService.query_llm(question=question, docs_data=relevant_docs_data)
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