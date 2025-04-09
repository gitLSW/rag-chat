import torch
from doctr.io import DocumentFile
from doctr.models import ocr_predictor  # OCR = Optical Character Recognition, extracts text from images or PDFs
from collections import defaultdict
from sentence_transformers import SentenceTransformer  # Pretrained model to convert text into numerical vectors (embeddings)
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from chromadb.utils import embedding_functions
from openai import OpenAI as LLMApi  # Used to call local or remote large language models (LLMs)
from filelock import FileLock
import re
import os
import json

# Configuration
LLM_MODEL = 'deepseek-r1:32b'  # The specific LLM model used to answer questions
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph
TEXT_DETECTION_MODEL = 'fast_base'  # Model for detecting where text is on the page
TEXT_RECOGNITION_MODEL = 'crnn_vgg16_bn'  # Model for recognizing characters within the detected text regions

DEVICE = torch.device("cuda:0") # The OCR must run on a gpu or it will seg fault.

ACCESS_TABLE_PATH = './access_rights.json'

class RAGService:
    def __init__(self):
        """
        Initializes the RAGService instance.

        - Sets up a persistent vector database using ChromaDB to store text embeddings.
        - Loads a pre-trained OCR model with specific detection and recognition architectures.
        - Configures OCR to group detected text into lines and paragraphs.
        - Connects to a local LLM API (e.g., via Ollama).
        - Loads a sentence transformer for generating vector embeddings of text.
        """
        # Initialize persistent vector database (ChromaDB)
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name="documents")

        # Load OCR model with text detection and recognition components
        self.ocr_model = ocr_predictor(det_arch=TEXT_DETECTION_MODEL,
                                       reco_arch=TEXT_RECOGNITION_MODEL,
                                       pretrained=True,
                                       assume_straight_pages=True,
                                       preserve_aspect_ratio=True).to(DEVICE)
        self.ocr_model.doc_builder.resolve_lines = True  # Group words into lines
        self.ocr_model.doc_builder.resolve_blocks = True  # Group lines into blocks (paragraphs)

        # Connect to a local LLM (e.g., via Ollama)
        self.llm_client = LLMApi(
            base_url="http://localhost:11434/v1",  # Ollama's default port
            api_key="ollama"
        )

        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        if not os.path.exists(ACCESS_TABLE_PATH):
            self.access_rights = {}
            return

        lock = FileLock(ACCESS_TABLE_PATH)
        with lock:
            with open(ACCESS_TABLE_PATH, "r") as f:
                self.access_rights = json.load(f)
        

    def add_doc(self, path, access_groups, access_role):
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
        pages = DocumentFile.from_pdf(path)
        doc = self.ocr_model(pages)

        # Save the OCR extracted content in .txt file
        txt_path = path.replace('.pdf', '.txt')

        access_rights = self.access_rights.get(txt_path)
        if access_rights != None and not access_role in access_rights:
            return # INSUFFICIENT RIGHTS

        lock = FileLock(txt_path)
        with lock:
            with open(txt_path, 'w') as f:
                f.write(self.convert_doc_data_to_text(doc))
        
        lock = FileLock(ACCESS_TABLE_PATH)
        with lock:
            with open(ACCESS_TABLE_PATH, 'w') as f:
                self.access_rights[txt_path] = access_groups
                json.dump(self.access_rights, f)

        for page in doc.pages:
            for block in page.blocks:
                block_text = self.convert_block_data_to_text(block)
                # Convert paragraph into an embedding (vector representation)
                block_embedding = self.embedding_model.encode(block_text).tolist()

                # Store the text and its embedding in the vector DB (ChromaDB)
                self.collection.upsert(
                    embeddings=[block_embedding],
                    ids=[str(hash(str(block_embedding)))],
                    metadatas=[{
                        'txt_path': txt_path,
                        'page_num': page.page_idx + 1
                    }]
                )

        print(f'Successfully stored Document {path}')
        return txt_path
    

    # TEST THIS FUNCTION
    def delete_doc(self, path, access_role):
        txt_path = path.replace('.pdf', '.txt')
        access_rights = self.access_rights.get(txt_path)
        if access_rights != None and not access_role in access_rights:
            return # INSUFFICIENT RIGHTS
        
        lock = FileLock(ACCESS_TABLE_PATH)
        with lock:
            with open(ACCESS_TABLE_PATH, 'w') as f:
                del self.access_rights[txt_path]
                json.dump(self.access_rights, f)
        
        try:
            # Delete file
            os.remove(txt_path)
        except FileNotFoundError:
            print(f'No such file at {txt_path} found !')
        except Exception as e:
            print(f'Error {e}')
            return
        
        # Delete from DB
        self.collection.delete(where={'txt_path': txt_path})


    # TEST THIS FUNCTION
    def update_doc(self, access_role, path, new_path, new_access_groups):
        # Convert PDF paths to TXT paths
        old_txt_path = path.replace('.pdf', '.txt')
        new_txt_path = new_path.replace('.pdf', '.txt')
        
        # Check access rights
        access_rights = self.access_rights.get(old_txt_path)
        if access_rights is not None and access_role not in access_rights:
            return False  # INSUFFICIENT RIGHTS
        
        # Check if source TXT file exists
        if not os.path.exists(old_txt_path):
            print(f"Error: Source text file '{old_txt_path}' not found.")
            return False
        
        # Check if destination TXT file already exists
        if os.path.exists(new_txt_path):
            print(f"Error: Destination text file '{new_txt_path}' already exists.")
            return False
        
        try:
            # Perform the file move/rename
            os.rename(old_txt_path, new_txt_path)
            
            # Update access rights
            lock = FileLock(ACCESS_TABLE_PATH)
            with lock:
                # Remove old entry and add new one
                if old_txt_path in self.access_rights:
                    del self.access_rights[old_txt_path]
                self.access_rights[new_txt_path] = new_access_groups
                
                with open(ACCESS_TABLE_PATH, 'w') as f:
                    json.dump(self.access_rights, f)
            
            # Update ChromaDB metadata
            doc_entries = self.collection.get(where={"txt_path": old_txt_path})
            
            for doc_id in doc_entries["ids"]:
                # Preserve all existing metadata, only update the path
                current_metadata = self.collection.get(ids=[doc_id])["metadatas"][0]
                updated_metadata = {
                    **current_metadata,  # Keep all existing metadata
                    "txt_path": new_txt_path  # Only update the path
                }
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[updated_metadata]
                )
            
            return True
        
        except Exception as e:
            print(f"Error during file operation: {str(e)}")
            return False
    

    def convert_block_data_to_text(self, block_data):
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


    def convert_doc_data_to_text(self, doc_data):
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
                block_text = self.convert_block_data_to_text(block)
                blocks.append(block_text)
    
        # Separate text blocks within a document
        return "\n\n".join(blocks)


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
        question_embedding = self.embedding_model.encode(question).tolist()
        # Perform semantic search in ChromaDB (based on embedding similarity between question and the paragraphs of all document)
        results = self.collection.query(query_embeddings=[question_embedding], n_results=n_results)
        
        # Return metadata of top matching results (e.g., file path and page number)
        nearest_neighbors = results['metadatas'][0] if results['metadatas'] else []
        return nearest_neighbors


    def gather_docs_knowledge(self, doc_paths):
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
            with open(path, 'r') as f:
                documents_text.append(f.read())
    
        # Separate different documents with more spacing
        return "\n\n\n\n\n\n\n\nNext Relevant Text:\n".join(documents_text)
        

    def query_llm(self, question, access_role, n_results=5):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.

        Args:
            question (str): The user's natural language question.
            n_results (int): Number of document chunks to retrieve for context (default is 5).

        Returns:
            str: Final LLM-generated answer with source references.
        """
        # Step 1: Perform semantic search for relevant document sections
        found_docs_data = self.find_docs(question, n_results)

        # Step 2: Group results by document and page
        summarized_docs = ''
        doc_sources_map = defaultdict(set)
        for doc_data in found_docs_data:
            txt_path = doc_data['txt_path']

            # Check if user has access to the document
            access_groups = self.access_rights[txt_path]
            if not access_role in access_groups:
                continue
            
            doc_sources_map[txt_path].add(doc_data['page_num'])

            # Step 3: Extract and aggregate text from relevant documents
            doc_text = self.gather_docs_knowledge([txt_path])
            summarize_prompt = f'{question}\n\nSummarize all the relevant information and facts needed to answer the question in a manner from the following text:\n\n{doc_text}'
            doc_summary = self.llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": summarize_prompt}],
                temperature=0.3
            )
            doc_summary = re.sub(r'<think>.*?</think>', '', doc_summary.choices[0].message.content, flags=re.DOTALL)
            summarized_docs += f'{doc_summary}\n\n'

        # Step 4: Build prompt and query the LLM
        prompt = f"{question}\n\nUse the following texts to briefly and precisely answer the question in a concise manner:\n\n\n{summarized_docs}"

        # Send prompt to local LLM (e.g., Ollama)
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Step 5: Prepare final answer with source info
        sources_info = 'Consult these documents for more detail:\n'
        for path, pages in doc_sources_map.items():
            sources_info += f'{path.replace('.txt', '.pdf')} on pages {", ".join(map(str, sorted(pages)))}\n'

        # Strip out internal model markers like <think> tags
        answer_text = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)

        return sources_info + answer_text
    
rag = RAGService()
rag.add_doc('/home/lsw/Downloads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf', ['accounting', 'management'], 'test')
# rag.add_doc('/home/lsw/Downloads/_ALL THESIS/_On the usefulness of context data.pdf', ['accounting', 'management'])
rag.add_doc('/home/lsw/Downloads/AlexanderMeetingDiogines.pdf', ['accounting', 'management'], 'management')

response = rag.query_llm('Did Charlemagne ever meet Alexander the Great ?', 'management')
print(response)