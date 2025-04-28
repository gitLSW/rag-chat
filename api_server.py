import os
import uuid
from path_normalizer import merge_path
from mimetypes import guess_type
from doc_extractor import DocExtractor
from api_responses import *
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from company_middleware import CompanyMiddleware
from auth_middleware import AuthMiddleware  # test postman


# -----------------------------
# Request Models
# -----------------------------

class BaseCompanyRequest(BaseModel):
    company_id: str
    user_role: str


class AddDocRequest(BaseCompanyRequest):
    path: Optional[str]
    new_doc_access_groups: List[str]
    file: UploadFile = File(...)


class UpdateDocRequest(BaseCompanyRequest):
    old_path: str
    new_path: str
    new_doc_json: dict
    new_access_groups: List[str]


class DeleteDocRequest(BaseCompanyRequest):
    path: str


class RAGRequest(BaseCompanyRequest):
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
app.add_middleware(AuthMiddleware)  # test postman


@app.post("/add_doc")
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
    company_mw = get_company_middleware(req.company_id)
    
    # Ensure the directory exists
    upload_dir = f'{req.company_id}/uploads/{uuid.uuid4()}'
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    source_path = merge_path(upload_dir, req.file.filename)

    # Save the file
    with open(source_path, "wb") as buffer:
        buffer.write(await req.file.read())

    # Add document logic
    res = await company_mw.add_doc(source_path, req.path, req.new_doc_access_groups, req.user_role)

    if res.status_code == 200:
        os.remove(source_path) # The original file is no longer needed

    return res


@app.post("/update_doc")
def update_doc(req: UpdateDocRequest):
    company_mw = get_company_middleware(req.company_id)
    return company_mw.update_doc(req.old_path, req.new_path, req.new_doc_json, req.new_access_groups, req.user_role)


@app.post("/delete_doc")
def delete_doc(req: DeleteDocRequest):
    company_mw = get_company_middleware(req.company_id)
    return company_mw.delete_doc(req.path, req.user_role)


# TODO: Gather docs for download (if not downlaoding from honesty system)
@app.post("/search_docs")
def search_docs(req: RAGRequest):
    company_mw = get_company_middleware(req.company_id)
    return company_mw.search_docs(req.question, req.user_role, req.search_depth)


@app.post("/query_llm")
async def query_llm(req: RAGRequest):
    company_mw = get_company_middleware(req.company_id)
    return await company_mw.query_llm(req.question, req.user_role, req.search_depth)
    
# TODO: Handle non ApiErrors !!! 

# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=7500, reload=True)


# test postman 

from fastapi import WebSocket, Depends
import jwt
import os
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret")

@app.post("/login")
def login():
    # return token for user_id = 123
    payload = {
        "sub": "123",
        "company_id": "mycompany",
        "role": "admin",
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
    return {"token": token}

@app.websocket("/connectChat")
async def connect_chat(websocket: WebSocket):
    from rag_service import RAGService
    await websocket.accept()

    try:
        token = websocket.headers.get("Authorization")
        if not token:
            await websocket.close(code=1008)
            return

        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        company_id = payload.get("company_id")

        rag = RAGService(company_id)
        async for message in websocket.iter_text():
            async for chunk in rag.query(message):
                await websocket.send_text(chunk)
    except Exception as e:
        await websocket.close(code=1011)
        print("Error:", e)
