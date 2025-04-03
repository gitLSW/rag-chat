import os
import fitz  # PyMuPDF
import pytesseract
import uuid
from PIL import Image
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from openai import OpenAI

# Configuration
LLM_MODEL = "deepseek-r1:32b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_SPLITTER_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "length_function": len
}

class PDFProcessor:
    def __init__(self, enable_ocr=True):
        self.enable_ocr = enable_ocr
        
    def extract_pages(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page numbers and metadata"""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            metadata = {
                "source": pdf_path,
                "page": page_num + 1,
                "doc_id": str(uuid.uuid4())
            }
            
            if not page_text.strip() and self.enable_ocr:
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                
            pages.append({
                "text": page_text,
                "metadata": metadata
            })
            
        return pages

class RAGSystem:
    def __init__(self, persist_dir="chroma_db"):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(**TEXT_SPLITTER_CONFIG)
        self.vector_store = Chroma(
            collection_name="rag_collection",
            embedding_function=self.embedding_model.encode,
            persist_directory=persist_dir
        )
        self.llm_client = OpenAI(
            base_url="http://localhost:11434/v1",  # Ollama's default port
            api_key="ollama"
        )
        
    def add_documents(self, pdf_paths: List[str]):
        """Process and add PDF documents with metadata"""
        processor = PDFProcessor()
        
        for path in pdf_paths:
            pages = processor.extract_pages(path)
            for page in pages:
                # Create LangChain documents with metadata
                docs = [Document(
                    page_content=page["text"],
                    metadata=page["metadata"]
                )]
                # Split and add to vector store
                split_docs = self.text_splitter.split_documents(docs)
                if split_docs:
                    texts = [doc.page_content for doc in split_docs]
                    metadatas = [doc.metadata for doc in split_docs]
                    self.vector_store.add_texts(
                        texts=texts,
                        metadatas=metadatas,
                        embeddings=self.embedding_model.encode(texts)
                    )
        
        self.vector_store.persist()
            
    def query(self, question: str, top_k: int = 3) -> str:
        """Execute RAG query with source tracing"""
        # Retrieve relevant context with metadata
        query_embedding = self.embedding_model.encode(question)
        docs = self.vector_store.similarity_search_by_vector(
            query_embedding,
            k=top_k
        )
        
        # Build context with sources
        context = ""
        for doc in docs:
            context += f"From {doc.metadata['source']} (page {doc.metadata['page']}):\n"
            context += doc.page_content + "\n\n"
        
        # Create prompt with source references
        prompt_template = PromptTemplate.from_template(
            """Answer the question using only this context. Cite sources using [source] notation:
            
            {context}
            
            Question: {question}
            Answer:"""
        )
        prompt = prompt_template.format(context=context, question=question)
        
        # Query LLM with specified model
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content

# Example Usage
if __name__ == "__main__":
    # Initialize RAG system with persistent storage
    rag = RAGSystem(persist_dir="./chroma_db")
    
    # Add documents (processed with metadata)
    rag.add_documents(["doc1.pdf", "doc2.pdf"])
    
    # Query with source tracing
    question = "What safety protocols are mentioned in the documents?"
    answer = rag.query(question)
    print(f"Question: {question}")
    print(f"Answer:\n{answer}")