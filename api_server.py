import os
import uuid
from path_normalizer import merge_path
from mimetypes import guess_type
from doc_extractor import DocExtractor
from api_responses import *
from fastapi import FastAPI, HTTPException, File, UploadFile, WebSocket
from pydantic import BaseModel
from typing import List, Optional
from company_middleware import CompanyMiddleware
from auth_middleware import AuthMiddleware
from websocket_service import WebsocketService

# -----------------------------
# Request Models
# -----------------------------

class AddDocRequest(BaseModel):
    path: Optional[str]
    new_doc_access_groups: List[str]
    file: UploadFile = File(...)


class UpdateDocRequest(BaseModel):
    old_path: str
    new_path: str
    new_doc_json: dict
    new_access_groups: List[str]


class DeleteDocRequest(BaseModel):
    path: str


class RAGRequest(BaseModel):
    question: str
    search_depth: int = 10



# -----------------------------
# Helper to Init Company Server
# -----------------------------

company_mw_cache = {}
def get_company_middleware(company_id):
    company_mw = company_mw_cache.get(company_id)
    if company_mw is not None:
        return company_mw
    
    company_mw = CompanyMiddleware(company_id)
    company_mw_cache[company_id] = company_mw
    return company_mw
    


# -----------------------------
# API Routes
# -----------------------------

app = FastAPI()
app.add_middleware(AuthMiddleware)

@app.post("/create")
async def add_doc(req: AddDocRequest):
    mime_type, _ = guess_type(req.file.filename)

    # Check if MIME type is supported by DocExtractor
    supported_mime_types = DocExtractor._get_handlers().keys()
    if mime_type not in supported_mime_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {mime_type}. Supported types: {', '.join(supported_mime_types)}"
        )
    
    # Initialize the access middleware
    company_mw = get_company_middleware(req.state.company_id)
    
    # Ensure the directory exists
    upload_dir = f'{req.state.company_id}/uploads/{uuid.uuid4()}'
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    source_path = merge_path(upload_dir, req.file.filename)

    # Save the file
    with open(source_path, "wb") as buffer:
        buffer.write(await req.file.read())

    # Add document logic
    res = await company_mw.add_doc(source_path, req.path, req.new_doc_access_groups, req.state.user_role)

    if res.status_code == 200:
        os.remove(source_path) # The original file is no longer needed

    return res


@app.post("/update")
def update_doc(req: UpdateDocRequest):
    company_mw = get_company_middleware(req.state.company_id)
    return company_mw.update_doc(req.old_path, req.new_path, req.new_doc_json, req.new_access_groups, req.state.user_role)


@app.post("/delete")
def delete_doc(req: DeleteDocRequest):
    company_mw = get_company_middleware(req.state.company_id)
    return company_mw.delete_doc(req.path, req.state.user_role)


# TODO: Gather docs for download (if not downlaoding from honesty system)
@app.post("/search")
def search_docs(req: RAGRequest):
    company_mw = get_company_middleware(req.state.company_id)
    return company_mw.search_docs(req.question, req.state.user_role, req.search_depth)


@app.websocket("/chat")
async def websocket_query(websocket: WebSocket):
    company_id = websocket.scope["state"].company_id
    user_role = websocket.scope["state"].user_role
    company_mw = get_company_middleware(company_id)
    
    while True:
        # Accept the WebSocket connection
        await websocket.accept()

        # Receive the chat message payload (should contain question and optional search_depth)
        data = await websocket.receive_json()
        question = data.get("question")
        search_depth = data.get("search_depth", 10)

        # Stream tokens from the RAG pipeline
        async for token in company_mw.query_llm(question, user_role, search_depth, stream=True):
            await websocket.send_text(token)
    


# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=7500, reload=True)