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
                        "page_number": page.page_idx + 1 + start_value, # RENAME TO page_num
                        "text": block_text # REMOVE THIS ISNT NEEDED
                    }],
                    ids=[f"{path}-{page.page_idx}-{hash(block_text)}"]
                )

        print(f'Successfully stored Document {path}')

    doc_path = '/home/lsw/Downloads/CHARLEMAGNE_AND_HARUN_AL-RASHID.pdf'



    def find_docs(self, query, n=5):
        query_embedding = self.embedding_model.encode(query).tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=n)
        
        # Extract metadata from results
        nearest_neighbors = results["metadatas"][0] if results["metadatas"] else []
        return nearest_neighbors



    # REWORK
    def gather_docs(self, results):
        results_texts = ''
        for res in results: # MAKE RESULT DOCUMENT PATHS UNIQUE
            path = res['pdf_path']
            page = res['page_number']
            pages = DocumentFile.from_pdf(path)
            out = self.ocr_model([pages[page]]) # Seg fault if too large, REMOVE page selection

            results_texts = ''
            for page in out.pages:
                for block in page.blocks:
                    block_text = []
                    for line in block.lines:
                        line_text = " ".join(word.value for word in line.words)
                        block_text.append(line_text)
                    results_texts += "\n".join(block_text) + "\n" # Separator between blocks
            results_texts += "\n\n\n\n"
        
        return results_texts



rag_service = RAGService()

# Store document embeddings
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

