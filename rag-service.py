from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

class RAGService:
    def __init__(self):
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path="chroma_db")
        self.collection = self.client.get_or_create_collection(name="documents")

        self.ocr_model = ocr_predictor(det_arch='fast_base', reco_arch='crnn_vgg16_bn', pretrained=True, assume_straight_pages=True, preserve_aspect_ratio=True)
        self.ocr_model.doc_builder.resolve_lines = True
        self.ocr_model.doc_builder.resolve_blocks = True

        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


    def add_doc(self, path):
        start_value = 9
        pages = DocumentFile.from_pdf(path)
        out = self.ocr_model(pages[start_value:start_value+3]) # Seg fault if too large
        # print_plain_text(out)
        for page in out.pages:
            for block in page.blocks:
                block_text = " ".join(
                    " ".join(word.value for word in line.words) for line in block.lines
                )
                block_embedding = self.embedding_model.encode(block_text).tolist()

                # Store in ChromaDB
                self.collection.add(
                    embeddings=[block_embedding],
                    metadatas=[{
                        "pdf_path": path,
                        "page_num": page.page_idx + 1 + start_value, # Remove start_value
                    }],
                    ids=[f"{path}-{page.page_idx}-{block_embedding}"]
                )
        print(f'Successfully stored Document {path}')


    def find_docs(self, query, n=5):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        # Extract metadata from results
        nearest_neighbors = results["metadatas"][0] if results["metadatas"] else []
        return nearest_neighbors


    def gather_docs(self, doc_paths):
        # Get unique document paths
        unique_paths = list(set(doc_paths))
        documents_text = []
        
        for path in unique_paths:
            pages = DocumentFile.from_pdf(path)
            # Process all pages with OCR
            out = self.ocr_model(pages)

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
        
        
        
        


rag_service = RAGService()

# Store document embeddings
# doc_path = '/home/lsw/Downloads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf'
# rag_service.add_doc(doc_path)

# Example usage:
query = "Who is charlemagne?"
results = rag_service.find_docs(query, 5)
for res in results:
    print(res)
results_texts = rag_service.gather_docs(results)
print(results_texts)


# def print_plain_text(out):
#     for page in out.pages:
#         print(f"Page {page.page_idx + 1}:")
#         for block in page.blocks:
#             block_text = []
#             for line in block.lines:
#                 line_text = " ".join(word.value for word in line.words)
#                 block_text.append(line_text)
#             print("\n".join(block_text))
#             print("\n" + "-"*40 + "\n")  # Separator between blocks

