import os
import shutil
import uuid
import json
import logging
from utils import logs_path, get_env_var, get_company_path
from jsonschema_fill_default import fill_default

from mimetypes import guess_type

from fastapi import FastAPI, Query, Request, Form, HTTPException, File, UploadFile
from jsonschema import validate

from services.rag_service import get_company_rag_service
from services.doc_extractor import DocExtractor
from services.chat_websocket import router as chat_ws_router
from middleware.auth.token_middleware import TokenMiddleware
from middleware.auth.api_access_middleware import APIAccessMiddleware
from middleware.error_handler_middleware import register_exception_handlers

# Load environment variables
PORT = get_env_var('PORT')
API_KEY = get_env_var('API_KEY')
API_ALLOWED_IPs = get_env_var('API_ALLOWED_IPs')
PUBLIC_KEY_URL = get_env_var('PUBLIC_KEY_SOURCE_URL')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(logs_path, 'server.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# -----------------------------
# Request Model Schemata
# -----------------------------

# Every req header must contain a Bearer token in which the Authorization server encoded the user's company_id and access role
# and every endpoint for the CPU server (= all endpoints, except /chat) must additonally contain a x-api-key key.
ADD_DOC_SCHEMA_SCHEMA = {
    'type': 'object',
    'properties': {
        'docType': {'type': 'string'},
        'docSchema': {'type': 'object'} # The schema which shall be defined
    },
    'required': ['docType', 'docSchema']
}

DOC_DATA_SCHEMA = {
    'type': 'object',
    'properties': {
        'id': {'type': 'string'},
        'accessGroups': {
            'type': ['array', 'null'],
            'items': {'type': 'string'},
            'minItems': 1
        },
        # 'path': {'type': ['string', 'null']},
        'docType': {'type': ['string', 'null']}
        # Add more fields according to the doc_type's JSON Schema
    },
    'required': ['id'],
    'additionalProperties': True
}

UPDATE_DOC_SCHEMA = {
    'type': 'object',
    'properties': {
        'mergeExisting': {'type': 'boolean', 'default': False},
        'docData': DOC_DATA_SCHEMA,
    },
    'required': ['docData']
}


# -----------------------------
# API Routes
# -----------------------------

app = FastAPI()


# Add the HTTP middleware
register_exception_handlers(app)
app.add_middleware(TokenMiddleware, public_key_url=PUBLIC_KEY_URL)
app.add_middleware(
    APIAccessMiddleware,
    api_key=API_KEY,
    allowed_ips=API_ALLOWED_IPs,
    exempt_paths={'/documentSchemata', '/search', '/chat'}
)

app.include_router(chat_ws_router) # Add /chat endpoint



@app.post('/users')
async def create_user(req: Request):
    user_data = await req.json()
    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.access_manager.create_overwrite_user(user_data, req.state.user)


@app.delete('/users/{user_id}')
async def delete_user(user_id, req: Request):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.access_manager.delete_user(user_id, req.state.user)



@app.post('/documentSchemata')
async def add_doc_schema(req: Request):
    body = await req.json()
    validate(body, ADD_DOC_SCHEMA_SCHEMA)

    rag_service = get_company_rag_service(req.state.user.company_id)
    return await rag_service.add_doc_schema(body['docType'], body['docSchema'], req.state.user)


@app.get('/documentSchemata') 
async def get_doc_schemata(req: Request):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.get_doc_schemata()


@app.delete('/documentSchemata/{doc_type}')
async def delete_doc_schema(doc_type, req: Request):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return await rag_service.delete_doc_schema(doc_type, req.state.user)



@app.post('/documents')
async def create_doc(req: Request,
                     file: UploadFile = File(...),
                     forceOcr: bool = Form(False),
                     allowOverride: bool = Form(True),
                     docData: str = Form(...)):
    mime_type, _ = guess_type(file.filename)

    # Check if MIME type is supported by DocExtractor
    supported_mime_types = DocExtractor._get_handlers().keys()
    if mime_type not in supported_mime_types:
        raise HTTPException(400, f"Unsupported file type: {mime_type}. Supported types: {', '.join(supported_mime_types)}")

    docData = json.loads(docData)
    validate(docData, DOC_DATA_SCHEMA)

    # Ensure the directory exists
    upload_dir = get_company_path(req.state.user.company_id, f'uploads/{uuid.uuid4()}')
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    source_path = os.path.join(upload_dir, file.filename)

    # Save the file
    with open(source_path, 'wb') as buffer:
        buffer.write(await file.read())
    
    error = None
    try:
        # Add and process doc
        rag_service = get_company_rag_service(req.state.user.company_id)
        res = await rag_service.create_doc(source_path, docData, forceOcr, allowOverride, req.state.user)
    except (HTTPException, Exception) as e:
        error = e
    finally:
        shutil.rmtree(upload_dir, ignore_errors=True) # The original file is no longer needed
    
    if error:
        raise error

    return res


@app.put('/documents/{doc_id}')
async def update_doc(doc_id: str, req: Request):
    body = await req.json()

    doc_data = body.get('docData')
    if isinstance(doc_data, dict):
        body_doc_id = doc_data.get('id')
        if not body_doc_id:
            body['docData']['id'] = doc_id
        elif body_doc_id != doc_id:
            raise HTTPException(400, "URL document id doesn't match request body's document id!")
    
    fill_default(body, UPDATE_DOC_SCHEMA)
    validate(body, UPDATE_DOC_SCHEMA)

    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.update_doc_data(body['docData'], body['mergeExisting'], req.state.user)


@app.get('/documents/{doc_id}')
async def get_doc(doc_id, req: Request):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return await rag_service.get_doc(doc_id, req.state.user)


@app.delete('/documents/{doc_id}')
async def delete_doc(doc_id, req: Request):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.delete_doc(doc_id, req.state.user)



# TODO: Gather docs for download (if not downlaoding from honesty system
@app.get('/search')
async def search_docs(req: Request,
                      question: str = Query(..., description="The search question"),
                      searchDepth: int = Query(10, ge=1, description="Search depth, default 10")):
    rag_service = get_company_rag_service(req.state.user.company_id)
    return rag_service.find_docs(question, searchDepth, req.state.user)



# main.py
import uvicorn

if __name__ == '__main__':
    logger.info("Started Server...")
    uvicorn.run(app, host='0.0.0.0', port=PORT)