import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional
from access_middleware import AccessMiddleware

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

access_mw_cache = {}
def get_access_middleware(company_id):
    try:
        access_mw = access_mw_cache.get(company_id)
        if access_mw is not None:
            return access_mw
        
        access_mw = AccessMiddleware(company_id)
        access_mw_cache[company_id] = access_mw
        return access_mw
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not initialize company {company_id}: {e}")
    


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
    access_mw = get_access_middleware(req.company_id)
    
    # Ensure the directory exists
    upload_dir = f'{req.company_id}/uploads'
    os.makedirs(upload_dir, exist_ok=True)

    # Create the file path to save the uploaded PDF
    file_path = os.path.join(upload_dir, req.file_name)

    # Save the file
    with open(file_path, "wb") as buffer:
        buffer.write(await req.file.read())

    # Add document logic
    access_mw.add_doc(req.file_name, req.new_doc_access_groups, req.user_role)

    return {"message": "Document added successfully."}


@app.post("/update_doc")
def update_doc(req: UpdateDocRequest):
    access_mw = get_access_middleware(req.company_id)
    access_mw.update_doc(req.old_file_name, req.new_file_name, req.new_access_groups, req.user_role)
    return {"message": "Document updated successfully."}


@app.post("/delete_doc")
def delete_doc(req: DeleteDocRequest):
    access_mw = get_access_middleware(req.company_id)
    access_mw.delete_doc(req.file_name, req.user_role)
    return {"message": "Document deleted successfully."}


@app.post("/find_docs")
def find_docs(req: FindDocsRequest):
    access_mw = get_access_middleware(req.company_id)
    docs = access_mw.find_docs(req.question, req.user_role, req.n_results)
    return {"docs": docs}


@app.post("/query_llm")
def query_llm(req: QueryLLMRequest):
    access_mw = get_access_middleware(req.company_id)
    answer = access_mw.query_llm(req.question, req.user_role, req.n_results)
    return {"answer": answer}
    
# TODO: Handle non ApiErrors !!! 

# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="172.0.0.1", port=8000, reload=True)
