from doctr.io import DocumentFile
from doctr.models import ocr_predictor  # OCR = Optical Character Recognition, extracts text from images or PDFs
from collections import defaultdict
from sentence_transformers import SentenceTransformer  # Pretrained model to convert text into numerical vectors (embeddings)
import chromadb  # A vector database for storing and retrieving paragraph embeddings efficiently
from chromadb.utils import embedding_functions
from openai import OpenAI as LLMApi  # Used to call local or remote large language models (LLMs)
import re

# Configuration
LLM_MODEL = 'deepseek-r1:32b'  # The specific LLM model used to answer questions
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model used to generate the embedding vector representation of a paragraph
TEXT_DETECTION_MODEL = 'fast_base'  # Model for detecting where text is on the page
TEXT_RECOGNITION_MODEL = 'crnn_vgg16_bn'  # Model for recognizing characters within the detected text regions

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
        self.ocr_model = ocr_predictor(det_arch=TEXT_DETECTION_MODEL, reco_arch=TEXT_RECOGNITION_MODEL, pretrained=True, assume_straight_pages=True, preserve_aspect_ratio=True)
        self.ocr_model.doc_builder.resolve_lines = True  # Group words into lines
        self.ocr_model.doc_builder.resolve_blocks = True  # Group lines into blocks (paragraphs)

        # Connect to a local LLM (e.g., via Ollama)
        self.llm_client = LLMApi(
            base_url="http://localhost:11434/v1",  # Ollama's default port
            api_key="ollama"
        )

        # Load sentence embedding model
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)


    def read_doc(self, path):
        """
        Reads a PDF file from the given path and runs OCR to extract structured content.

        Args:
            path (str): Path to the PDF file.

        Returns:
            Document: A Doctr-processed document object containing pages, blocks, lines, and words.
        """
        # Load and process PDF file using OCR
        pages = DocumentFile.from_pdf(path)
        return self.ocr_model(pages)


    def add_doc(self, path):
        """
        Processes a document and adds its paragraphs and their embeddings to the vector database.

        Args:
            path (str): Path to the document file (PDF).

        Returns:
            None
        """
        doc = self.read_doc(path)
        for page in doc.pages:
            for block in page.blocks:
                block_text = self.convert_block_to_text(block)
                # Convert block text into an embedding (vector representation)
                block_embedding = self.embedding_model.encode(block_text).tolist()

                # Store the text and its embedding in the vector DB (ChromaDB)
                self.collection.upsert(
                    embeddings=[block_embedding],
                    ids=[f"{path}-{page.page_idx}-{hash(block_text)}"],
                    metadatas=[{
                        "pdf_path": path,
                        "page_num": page.page_idx + 1
                    }]
                )
        print(f'Successfully stored Document {path}')


    def convert_block_to_text(self, block_data):
        """
        Converts an OCR block (paragraph) into plain text.

        Args:
            block_data: A block object from Doctr OCR output.

        Returns:
            str: Text content from the block, joined line by line.
        """
        processed_lines = []
        for line in block_data.lines:
            # Combine individual words in the line
            line_text = " ".join(word.value for word in line.words).rstrip()
            
            # Handle hyphenated line breaks
            if line_text.endswith('-'):
                line_text = line_text[:-1].rstrip()
            
            processed_lines.append(line_text)
            
            # Only returning the first line here due to early return
            return " ".join(processed_lines)


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
        nearest_neighbors = results["metadatas"][0] if results["metadatas"] else []
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
            doc = self.read_doc(path)

            all_blocks = []
            for page in doc.pages:
                for block in page.blocks:
                    block_text = self.convert_block_to_text(block)
                    all_blocks.append(block_text)
        
            # Separate text blocks within a document
            document_text = "\n\n".join(all_blocks)
            documents_text.append(document_text)
    
        # Separate different documents with more spacing
        return "\n\n\n\n\n\n\n\n".join(documents_text)
        

    def query_llm(self, question, n_results=5):
        """
        Answers a user question by retrieving relevant document content and querying a local LLM.

        Args:
            question (str): The user’s natural language question.
            n_results (int): Number of document chunks to retrieve for context (default is 5).

        Returns:
            str: Final LLM-generated answer with source references.
        """
        # Step 1: Perform semantic search for relevant document sections
        found_docs_data = rag_service.find_docs(question, n_results)

        # Step 2: Group results by document and page
        doc_sources_map = defaultdict(set)
        for doc_data in found_docs_data:
            doc_sources_map[doc_data['pdf_path']].add(doc_data['page_num'])
        doc_sources_map = dict(doc_sources_map)

        # Step 3: Extract and aggregate text from relevant documents
        docs_text = rag_service.gather_docs_knowledge(doc_sources_map.keys())
        
        # Step 4: Build prompt and query the LLM
        prompt = f"{question}\n\n\n\n\nUse the following information to answer:\n\n\n{docs_text}"

        # Send prompt to local LLM (e.g., Ollama)
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        # Step 5: Prepare final answer with source info
        sources_info = 'Consult these documents for more detail:\n'
        for path, pages in doc_sources_map.items():
            sources_info += f'{path} on pages {", ".join(map(str, pages))}\n'

        # Strip out internal model markers like <think> tags
        answer_text = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)

        return f'{sources_info}\n{answer_text}'