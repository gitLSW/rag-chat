# RAG Service

RAG (Retrieval Augmented Generation) is an approach that combines a large language model (LLM) with integrated semantic search. It enables the system to find and utilize relevant resources from a corpus of documents to answer user queries more accurately, especially when the information isn’t directly available in the model's training data. In this implementation, the RAG Service extracts text from PDFs using OCR (Optical Character Recognition), transforms the text into embeddings with a sentence transformer, stores them in a vector database (ChromaDB), and uses a local ollama LLM to generate answers based on the retrieved context.

## Installation

Use python 3.11.0 and CUDA 12.6 (I used NVIDIA GPU Driver Version: 560.35.03)
Install the required packages by running:

```bash
pip install -r requirements.txt
```

## Usage

Initialize
```python
rag_service = RAGService()
```

Add Documents to VectorDB for semantic search
```python
rag_service.add_doc('path/to/your/document1', access_groups=['group1, group2])
rag_service.add_doc('path/to/your/document2', access_groups=['group2])
```

Example LLM Query usage (performs a semantic document search and provides the LLM with the relevant documents)
```python
answer = rag_service.query_llm("What is the meaning of life ?", access_role='group2', n_results=5)
print(answer)
```

Example of semantic document search finding the 5 most similar paragraphs
```python
docs = rag_service.find_docs("What is the meaning of life ?", access_role='group1', n_results=9)
```