from rag_service import RAGService
from access_manager import AccessManager
from api_responses import ApiError, AccessNotFoundError, InsufficientAccessError, ApiResponse, OKResponse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from company_api_server import CompanyAPIServer

app = FastAPI()

# -----------------------------
# Request Models
# -----------------------------

class BaseCompanyRequest(BaseModel):
    company_id: str
    user_role: str


class AddDocRequest(BaseCompanyRequest):
    file_name: str
    new_doc_access_groups: List[str]


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


class CompanyAPIServer:
    def __init__(self, company):
        self.rag_service = RAGService(company)
        self.access_manger = AccessManager(company)


    def add_doc(self, file_name, new_doc_access_groups, user_access_role):
        path = self.access_manger.create_file_access(file_name, new_doc_access_groups, user_access_role).data
        self.rag_service.add_doc(path)

    def delete_doc(self, file_name, user_access_role):
        path = self.access_manger.delete_file_access(file_name, user_access_role).data
        self.rag_service.delete_doc(path)

        
    def update_doc(self, old_file_name, new_file_name, new_access_groups, user_access_role):
        new_path = self.access_manger.update_file_access(old_file_name, new_file_name, new_access_groups, user_access_role)
        
        old_standard_name = old_file_name.split('.')[0]
        old_path = f'./{self.company}/{old_standard_name}'
        
        self.rag_service.update_doc(old_path, new_path)
        

    def find_docs(self, question, user_access_role, n_results):
        # Perform semantic search for relevant document sections
        found_docs_data = self.rag_service.find_docs(question, n_results)

        # Check if user has permission for these docs
        valid_docs_data = []
        for doc_data in found_docs_data:
            try:
                self.access_manger.has_file_access(doc_data['path'], user_access_role)
            except InsufficientAccessError as e:
                continue
            except AccessNotFoundError as e:
                print("ACCESS WASN'T FOUND FOR A FILE THAT SHOULD EXIST", e)
                raise e # If the access wasn't found the data is corrupt, as doc_data comes from the DB
            except e:
                print("SHOULD NEVER OCCUR", e)
                raise e

            valid_docs_data.append(doc_data)

        return valid_docs_data


    def query_llm(self, question, user_access_role, n_results):
        valid_docs_data = self.find_docs(question, user_access_role, n_results)
        
        # Ask LLM to answer based on the valid sources
        return self.rag_service.query_llm(question, valid_docs_data)
    
    


# -----------------------------
# Helper to Init Company Server
# -----------------------------

def get_company_api_server(company_id: str) -> CompanyAPIServer:
    try:
        return CompanyAPIServer(company_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not initialize company: {e}")


# -----------------------------
# API Routes
# -----------------------------

@app.post("/add_doc")
def add_doc(req: AddDocRequest):
    server = get_company_api_server(req.company_id)
    try:
        server.add_doc(req.file_name, req.new_doc_access_groups, req.user_role)
        return {"message": "Document added successfully."}
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/update_doc")
def update_doc(req: UpdateDocRequest):
    server = get_company_api_server(req.company_id)
    try:
        server.update_doc(req.old_file_name, req.new_file_name, req.new_access_groups, req.user_role)
        return {"message": "Document updated successfully."}
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/delete_doc")
def delete_doc(req: DeleteDocRequest):
    server = get_company_api_server(req.company_id)
    try:
        server.delete_doc(req.file_name, req.user_role)
        return {"message": "Document deleted successfully."}
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/find_docs")
def find_docs(req: FindDocsRequest):
    server = get_company_api_server(req.company_id)
    try:
        docs = server.find_docs(req.question, req.user_role, req.n_results)
        return {"docs": docs}
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/query_llm")
def query_llm(req: QueryLLMRequest):
    server = get_company_api_server(req.company_id)
    try:
        answer = server.query_llm(req.question, req.user_role, req.n_results)
        return {"answer": answer}
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
# TODO: Handle non ApiErrors !!! 

# main.py
import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
