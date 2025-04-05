# RAG Service

RAG (Retrieval Augmented Generation) is an approach that combines a large language model (LLM) with integrated semantic search. It enables the system to find and utilize relevant resources from a corpus of documents to answer user queries more accurately, especially when the information isn’t directly available in the model's training data. In this implementation, the RAG Service extracts text from PDFs using OCR, transforms the text into embeddings with a sentence transformer, stores them in a vector database (ChromaDB), and uses a local LLM to generate answers based on the retrieved context.

## Installation

Install the required packages by running:

```bash
pip install -r requirements.txt

## Usage

### Initialize
rag_service = RAGService()

### Add Documents to VectorDB for semantic search
rag_service.add_doc('path/to/your/document1')
rag_service.add_doc('path/to/your/document2')

### Example usage (queries the LLM with retrieved doc content)
answer = rag_service.query_llm("What is the meaning of life ?", n_results=5)
print(answer)