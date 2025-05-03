import os
import uuid
from dotenv import load_dotenv
# from typing import List, Optional

from mimetypes import guess_type

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, WebSocket
from pydantic import BaseModel

from services.rag_service import RAGService
from services.doc_extractor import DocExtractor
from auth.token_middleware import TokenMiddleware
from auth.api_access_middlware import APIAccessMiddleware

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_ALLOWED_IPs = os.getenv('API_ALLOWED_IPs')
PUBLIC_KEY_URL = os.getenv('PUBLIC_KEY_SOURCE_URL')

# -----------------------------
# Request Models
# -----------------------------

# Every req header must contain a Bearer token in which the Authorization server encoded the user's company_id and access role
# and every endpoint for the CPU server (= all endpoints, except /chat) must additonally contain a x-api-key key.

class CreateAccessGroupReq(BaseModel):
    access_group: str

class AddDocSchemaReq(BaseModel):
    docType: str
    docSchema: dict # JSON Schema

class UpdateDocReq(BaseModel):
    mergeExisting: bool = False
    allowOverride: bool = True
    docData: dict # = {
    #     id: str
    #     accessGroups: List[str]
    #     path: Optional[str] = None
    #     docType: Optional[str] = None
    #     # more fields for the doc_data, which are doc_type's JSON Schema
    # }

class CreateDocReq(UpdateDocReq):
    file: UploadFile = File(...)

class DocReq(BaseModel):
    id: str

class SemanticSearchReq(BaseModel):
    question: str
    searchDepth: int = 10


# -----------------------------
# Helper to Init Company Server
# -----------------------------

rag_service_cache = {}
def get_company_rag_service(company_id):
    rag_service = rag_service_cache.get(company_id)
    if rag_service:
        return rag_service
    
    rag_service = RAGService(company_id)
    rag_service_cache[company_id] = rag_service
    return rag_service
    


# -----------------------------
# API Routes
# -----------------------------

app = FastAPI()
app.add_middleware(TokenMiddleware, public_key_url=PUBLIC_KEY_URL)
app.add_middleware(
    APIAccessMiddleware,
    api_key=API_KEY,
    allowed_ips=API_ALLOWED_IPs,
    exempt_paths={'/chat'}
)


@app.post("/createAccessGroup")
async def create_access_group(req: CreateAccessGroupReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.access_manager.create_access_group(req.access_group, req.state.user_access_role)


@app.post("/addDocumentSchema")
async def add_doc_schema(req: AddDocSchemaReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.add_json_schema_type(req.docType, req.docSchema, req.state.user_access_role)


@app.post("/deleteDocumentSchema")
async def delete_doc_schema(req: Request):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.delete_json_schema_type(req.docType, req.state.user_access_role)


@app.post("/createDocument")
async def create_doc(req: CreateDocReq):
    mime_type, _ = guess_type(req.file.filename)

    # Check if MIME type is supported by DocExtractor
    supported_mime_types = DocExtractor._get_handlers().keys()
    if mime_type not in supported_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {mime_type}. Supported types: {', '.join(supported_mime_types)}"
        )
    
    # Ensure the directory exists
    upload_dir = f'{req.state.company_id}/uploads/{uuid.uuid4()}'
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    source_path = f'{upload_dir}/{req.file.filename}'

    # Save the file
    with open(source_path, "wb") as buffer:
        buffer.write(await req.file.read())

    # Add and process doc
    rag_service = get_company_rag_service(req.state.company_id)
    res = await rag_service.create_doc(source_path, req.docData, req.allowOverride, req.mergeExisting, req.state.user_role)

    if res.status_code == 200:
        os.remove(source_path) # The original file is no longer needed

    return res


@app.post("/updateDocument")
async def update_doc(req: UpdateDocReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.update_doc_data(req.docData, req.allowOverride, req.mergeExisting, req.state.user_role)


@app.post("/getDocument")
async def get_doc(req: DocReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.get_doc(req.id, req.state.user_access_role)


@app.post("/deleteDocument")
async def delete_doc(req: DocReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.delete_doc(req.id, req.state.user_role)


# TODO: Gather docs for download (if not downlaoding from honesty system)
@app.post("/search")
async def search_docs(req: SemanticSearchReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.find_docs(req.question, req.search_depth, req.state.user_role)


@app.websocket("/chat")
async def websocket_query(websocket: WebSocket):
    company_id = websocket.scope["state"].company_id
    user_role = websocket.scope["state"].user_role
    rag_service = get_company_rag_service(company_id)
    
    while True:
        # Accept the WebSocket connection
        await websocket.accept()

        # Receive the chat message payload (should contain question and optional search_depth)
        data = await websocket.receive_json()
        question = data.get("question")
        search_depth = data.get("searchDepth", 10)

        # Stream tokens from the RAG pipeline
        async for token in rag_service.rag_query(question, search_depth, user_role):
            await websocket.send_text(token)
    


# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=7500, reload=True)