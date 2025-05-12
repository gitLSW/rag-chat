import os
import uuid
import logging
from get_env_var import get_env_var
# from typing import List, Optional

from mimetypes import guess_type

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, WebSocket
from pydantic import BaseModel

from services.rag_service import get_company_rag_service
from services.doc_extractor import DocExtractor
from services.chat_websocket import router as chat_ws_router
from middleware.auth.token_middleware import TokenMiddleware
from middleware.auth.api_access_middlware import APIAccessMiddleware
from middleware.error_handler_middleware import ErrorHandlerMiddleware, ErrorHandlingMiddleware

# Load environment variables
API_KEY = get_env_var('API_KEY')
API_ALLOWED_IPs = get_env_var('API_ALLOWED_IPs')
PUBLIC_KEY_URL = get_env_var('PUBLIC_KEY_SOURCE_URL')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error_handler.log'),
        logging.StreamHandler()
    ]
)


# -----------------------------
# Request Models
# -----------------------------

# Every req header must contain a Bearer token in which the Authorization server encoded the user's company_id and access role
# and every endpoint for the CPU server (= all endpoints, except /chat) must additonally contain a x-api-key key.

class SemanticSearchReq(BaseModel):
    question: str
    searchDepth: int = 10

class CreateAccessGroupReq(BaseModel):
    accessGroup: str

class AddDocSchemaReq(BaseModel):
    docType: str
    docSchema: dict # JSON Schema

class CreateDocReq(BaseModel):
    file: UploadFile = File(...)
    forceOcr: bool = False
    allowOverride: bool = True
    docData: dict # = {
    #     id: str
    #     accessGroups: List[str]
    #     path: Optional[str] = ML classified path
    #     docType: Optional[str] = ML identified docType
    #     # more fields according to the doc_type's JSON Schema
    # }
    
class UpdateDocReq(BaseModel):
    mergeExisting: bool = False
    docData: dict # = {
    #     id: str
    #     accessGroups: Optional[List[str]] = oldDocData.accessGroups
    #     path: Optional[str] = oldDocData.path
    #     docType: Optional[str] = oldDocData.docType
    #     # more fields according to the doc_type's JSON Schema
    # }


# -----------------------------
# API Routes
# -----------------------------

app = FastAPI()


# Add the HTTP middleware
# error_handler = ErrorHandlerMiddleware(app)
# app.add_middleware(ErrorHandlingMiddleware, error_handler=error_handler)
# app.add_middleware(TokenMiddleware, public_key_url=PUBLIC_KEY_URL)
# app.add_middleware(
#     APIAccessMiddleware,
#     api_key=API_KEY,
#     allowed_ips=API_ALLOWED_IPs,
#     exempt_paths={'/chat'}
# )

app.include_router(chat_ws_router) # Add /chat endpoint


@app.post("/accessGroups")
async def create_access_group(req: CreateAccessGroupReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.access_manager.create_access_group(req.accessGroup, req.state.user_access_role)


@app.post("/documentSchemata")
async def add_doc_schema(req: AddDocSchemaReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.add_json_schema_type(req.docType, req.docSchema, req.state.user_access_role)


@app.delete("/documentSchemata/{doc_type}")
async def delete_doc_schema(doc_type, req: Request):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.delete_json_schema_type(doc_type, req.state.user_access_role)


@app.post("/documents")
async def create_doc(req: CreateDocReq):
    mime_type, _ = guess_type(req.file.filename)

    # Check if MIME type is supported by DocExtractor
    supported_mime_types = DocExtractor._get_handlers().keys()
    if mime_type not in supported_mime_types:
        raise HTTPException(400, f"Unsupported file type: {mime_type}. Supported types: {', '.join(supported_mime_types)}")

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
    res = await rag_service.create_doc(source_path, req.docData, req.forceOcr, req.allowOverride, req.state.user_role)

    if res.status_code == 200:
        os.remove(source_path) # The original file is no longer needed

    return res


@app.put("/documents/{doc_id}")
async def update_doc(doc_id, req: UpdateDocReq):
    body_doc_id = req.docData.get('id')
    if not body_doc_id:
        req.docData['id'] = doc_id
    elif body_doc_id != doc_id:
        raise HTTPException(400, "URL document id doesn't match request body's document id!")
    
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.update_doc_data(req.docData, req.mergeExisting, req.state.user_role)


@app.get("/documents/{doc_id}")
async def get_doc(doc_id, req):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.get_doc(doc_id, req.state.user_access_role)


@app.delete("/documents/{doc_id}")
async def delete_doc(doc_id, req):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.delete_doc(doc_id, req.state.user_role)


# TODO: Gather docs for download (if not downlaoding from honesty system)
@app.post("/search")
async def search_docs(req: SemanticSearchReq):
    rag_service = get_company_rag_service(req.state.company_id)
    return rag_service.find_docs(req.question, req.search_depth, req.state.user_role)


# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7500)
    print('Started Server...')