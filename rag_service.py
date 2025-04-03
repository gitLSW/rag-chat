from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI as LLMApi

# Configuration
LLM_MODEL = 'deepseek-r1:32b'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TEXT_DETECTION_MODEL = 'fast_base'
TEXT_RECOGNITION_MODEL = 'crnn_vgg16_bn'

class RAGService:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name="documents")

        self.ocr_model = ocr_predictor(det_arch=TEXT_DETECTION_MODEL, reco_arch=TEXT_RECOGNITION_MODEL, pretrained=True, assume_straight_pages=True, preserve_aspect_ratio=True)
        self.ocr_model.doc_builder.resolve_lines = True
        self.ocr_model.doc_builder.resolve_blocks = True

        self.llm_client = LLMApi(
            base_url="http://localhost:11434/v1",  # Ollama's default port
            api_key="ollama"
        )

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)


    def add_doc(self, path):
        for i in range(3, 10): # REMOVE THIS, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!
            pages = DocumentFile.from_pdf(path)
            out = self.ocr_model(pages[i:i+1]) # REMOVE THIS, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!
            # print_plain_text(out)
            for page in out.pages:
                for block in page.blocks:
                    block_text = " ".join(
                        " ".join(word.value for word in line.words) for line in block.lines
                    )
                    block_embedding = self.embedding_model.encode(block_text).tolist()

                    # Store in ChromaDB
                    self.collection.upsert(
                        embeddings=[block_embedding],
                        ids=[f"{path}-{page.page_idx}-{hash(block_text)}"],
                        metadatas=[{
                            "pdf_path": path,
                            "page_num": page.page_idx + 1 + i, # REMOVE i, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!
                        }]
                    )
            print(f'Successfully stored page {i} Document {path}') # REMOVE i, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!


    def find_docs(self, query, n=5):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        # Extract metadata from results
        nearest_neighbors = results["metadatas"][0] if results["metadatas"] else []
        return nearest_neighbors


    def gather_docs_knowledge(self, doc_paths):
        # Get unique document paths
        unique_paths = set(doc_paths)
        documents_text = []
        
        for path in unique_paths:
            pages = DocumentFile.from_pdf(path)
            # Process all pages with OCR
            for i in range(4, 7): # REMOVE THIS LINE, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!
                out = self.ocr_model(pages[i:i+1]) # REMOVE THIS, ITS SO MY MISINSTALLED UBUNTU DOESN'T ALWAYS SEGFAULT. THE OCR CAN HANDLE WHOLE DOCS !!!!

                all_blocks = []
                for page in out.pages:
                    for block in page.blocks:
                        processed_lines = []
                        for line in block.lines:
                            # Join words with spaces and strip trailing whitespace
                            line_text = " ".join(word.value for word in line.words).rstrip()
                        
                            # Remove trailing hyphen if present
                            if line_text.endswith('-'):
                                line_text = line_text[:-1].rstrip()  # Remove hyphen and any remaining whitespace
                        
                            processed_lines.append(line_text)
                    
                        # Join processed lines with single space
                        block_text = " ".join(processed_lines)
                        all_blocks.append(block_text)
            
                # Join blocks with 2 newlines within document
                document_text = "\n\n".join(all_blocks)
                documents_text.append(document_text)
    
        # Join documents with 8 newlines between them
        return "\n\n\n\n\n\n\n\n".join(documents_text)
        

    def query_llm(self, question, context_sources_num):
        found_docs_data = rag_service.find_docs(question, 5)
        # print(found_docs_data)

        # map search results to dict
        doc_sources_map = defaultdict(set)
        for doc_data in found_docs_data:
            doc_sources_map[doc_data['pdf_path']].add(doc_data['page_num'])
        doc_sources_map = dict(doc_sources_map)

        docs_text = rag_service.gather_docs_knowledge(doc_sources_map.keys())
        
        prompt = f"{question}\n\n\n\n\nUse the following information to answer:\n\n\n{docs_text}"

        # Query LLM with specified model
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        sources_info = 'Matches were found in:\n'
        for path, pages in doc_sources_map.items():
            sources_info += f'{path} on pages {", ".join(map(str, pages))}\n'

        return f'{sources_info}\n\n\n{response.choices[0].message.content}'
    
        
rag_service = RAGService()

# Store document embeddings
# doc_path = '/home/lsw/Downloads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf'
# doc_path = '/home/lsw/Downloads/_ALL THESIS/_On the usefulness of context data.pdf'
# rag_service.add_doc(doc_path)

# Example usage:
answer = rag_service.query_llm("Who is charlemagne?", 5)
print(answer)