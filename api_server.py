import os
import uuid
from dotenv import load_dotenv
# from typing import List, Optional

from mimetypes import guess_type

from fastapi import FastAPI, Request, HTTPException, File, UploadFile, WebSocket
import jsonschema

from services.rag_service import RAGService
from services.doc_extractor import DocExtractor
from auth.auth_middleware import AuthMiddleware
from auth.verify_source_middlware import VerifySourceMiddleware

# Load environment variables
load_dotenv()
API_KEY = os.getenv('API_KEY')
API_ALLOWED_IPs = os.getenv('API_ALLOWED_IPs')
PUBLIC_KEY_URL = os.getenv('PUBLIC_KEY_SOURCE_URL')



# -----------------------------
# API Request Schemata
# -----------------------------

# Base Document Schema (common to all documents)
BASE_DOC_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "path": {"type": ["string", "null"]},
        "docType": {"type": ["string", "null"]},
        "accessGroups": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["id", "accessGroups"],
    "additionalProperties": True  # Allows custom properties from docType schemas
}


# Endpoint-specific Schemas
ADD_DOC_SCHEMA_REQ = {
    "type": "object",
    "properties": {
        "docType": {"type": "string"},
        "docSchema": {"type": "object"}
    },
    "required": ["docType", "docSchema"]
}


DEL_DOC_REQ = {
    "type": "object",
    "properties": {
        "id": {"type": "string"}
    },
    "required": ["id"]
}


SEMANTIC_SEARCH_REQ = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "searchDepth": {"type": "integer", "default": 10}
    },
    "required": ["question"]
}



# -----------------------------
# Doc Data Validation
# -----------------------------

def validate_doc_data(doc_data, rag_service):
    """Validate document data against base + docType schema"""
    doc_type = doc_data.get("docType")
    if not doc_type:
        jsonschema.validate(doc_data, BASE_DOC_SCHEMA)
        return
    
    if doc_type_schema := rag_service.get_doc_type_schema(doc_type):
        # Merge schemas (base schema properties take precedence)
        full_schema = {
            "type": "object",
            "properties": {**doc_type_schema["properties"], **BASE_DOC_SCHEMA["properties"]},
            "required": list(set(BASE_DOC_SCHEMA["required"] + doc_type_schema.get("required", []))),
            "additionalProperties": True
        }
        jsonschema.validate(doc_data, full_schema)
    else:
        jsonschema.validate(doc_data, BASE_DOC_SCHEMA)



# -----------------------------
# API Routes with Direct Validation
# -----------------------------

@app.post("/addDocumentSchema")
async def add_doc_schema(req: Request):
    try:
        data = await req.json()
        jsonschema.validate(data, ADD_DOC_SCHEMA_REQ)
        
        rag_service = get_company_rag_service(req.state.company_id)
        return rag_service.add_json_schema_type(data["docType"], data["docSchema"], req.state.user_access_role)
    except jsonschema.ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/addDocument")
async def add_doc(req: Request, file: UploadFile = File(...)):
    try:
        # Validate file type
        mime_type, _ = guess_type(file.filename)
        if mime_type not in DocExtractor._get_handlers().keys():
            raise HTTPException(400, detail=f"Unsupported file type: {mime_type}")
        
        # Validate and parse JSON data
        form_data = await request.form()
        doc_data = json.loads(form_data["doc_data"])
        
        # Validate document data
        rag_service = get_company_rag_service(req.state.company_id)
        validate_doc_data(doc_data, rag_service)
        
        # Process file upload
        upload_dir = f'{req.state.company_id}/uploads/{uuid.uuid4()}'
        os.makedirs(upload_dir, exist_ok=True)
        source_path = f'{upload_dir}/{file.filename}'
        
        with open(source_path, "wb") as buffer:
            buffer.write(await file.read())
        
        # Add document
        res = await rag_service.add_doc(source_path, doc_data, req.state.user_role)
        if res.status_code == 200:
            os.remove(source_path)
        return res
    except json.JSONDecodeError:
        raise HTTPException(400, detail="Invalid JSON in doc_data")
    except jsonschema.ValidationError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        raise HTTPException(400, detail=str(e))


@app.post("/updateDocument")
async def update_doc(req: Request):
    try:
        doc_data = await req.json()
        rag_service = get_company_rag_service(req.state.company_id)
        validate_doc_data(doc_data, rag_service)
        return rag_service.update_doc(doc_data, req.state.user_role)
    except jsonschema.ValidationError as e:
        raise HTTPException(422, detail=str(e))


@app.post("/deleteDocument")
async def delete_doc(req: Request):
    try:
        data = await req.json()
        validate(data, DEL_DOC_REQ)
        
        rag_service = get_company_rag_service(req.state.company_id)
        return rag_service.delete_doc(data["id"], req.state.user_role)
    except jsonschema.ValidationError as e:
        raise HTTPException(422, detail=str(e))


@app.post("/search")
async def search_docs(req: Request):
    try:
        data = await req.json()
        validate(data, SEMANTIC_SEARCH_REQ)
        
        rag_service = get_company_rag_service(req.state.company_id)
        return rag_service.find_docs(
            data["question"],
            data.get("searchDepth", 10),
            req.state.user_role
        )
    except jsonschema.ValidationError as e:
        raise HTTPException(422, detail=str(e))


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
        async for token in rag_service.query_llm(question, search_depth, user_role, stream=True):
            await websocket.send_text(token)
    


# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=7500, reload=True)