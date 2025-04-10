import os
import json
from api_responses import ApiResponse, OKResponse
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from company_middleware import CompanyMiddleware

# -----------------------------
# Request Models
# -----------------------------

class BaseCompanyRequest(BaseModel):
    company_id: str
    user_role: str


class AddDocRequest(BaseCompanyRequest):
    file_name: str
    new_doc_access_groups: List[str]
    file: UploadFile = File(...)


class UpdateDocRequest(BaseCompanyRequest):
    old_file_name: str
    new_file_name: str
    new_access_groups: List[str]


class DeleteDocRequest(BaseCompanyRequest):
    file_name: str


class FindDocsRequest(BaseCompanyRequest):
    question: str
    n_results: int = 5


class QueryLLMRequest(FindDocsRequest):
    pass



# -----------------------------
# Helper to Init Company Server
# -----------------------------

company_mw_cache = {}
def get_access_middleware(company_id):
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

@app.post("/add_doc")
async def add_doc(req: AddDocRequest):
    # Ensure the file is a PDF
    if not req.file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    # Initialize the access middleware
    company_mw = get_access_middleware(req.company_id)
    
    # Ensure the directory exists
    upload_dir = f'{req.company_id}/uploads'
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    file_path = os.path.join(upload_dir, req.file_name)

    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(await req.file.read())

    # Add document logic
    res = company_mw.add_doc(req.file_name, req.new_doc_access_groups, req.user_role)

    if res.status_code == 200:
        os.remove(file_path) # The pdf is no longer needed


@app.post("/update_doc")
def update_doc(req: UpdateDocRequest):
    company_mw = get_access_middleware(req.company_id)
    return company_mw.update_doc(req.old_file_name, req.new_file_name, req.new_access_groups, req.user_role)


@app.post("/delete_doc")
def delete_doc(req: DeleteDocRequest):
    company_mw = get_access_middleware(req.company_id)
    return company_mw.delete_doc(req.file_name, req.user_role)


# TODO: Gather docs for download (if not downlaoding from honesty system)
# @app.post("/search_docs")
# def search_docs(req: FindDocsRequest):
#     company_mw = get_access_middleware(req.company_id)
#     docs = company_mw.search_docs(req.question, req.user_role, req.n_results)
#     return {"docs": docs}


@app.post("/query_llm")
def query_llm(req: QueryLLMRequest):
    company_mw = get_access_middleware(req.company_id)
    return company_mw.query_llm(req.question, req.user_role, req.n_results)
    
# TODO: Handle non ApiErrors !!! 

# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=8000, reload=True)
