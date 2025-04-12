from rag_service import RAGService
from access_manager import AccessManager
from api_responses import *


class CompanyMiddleware:
    def __init__(self, company):
        self.rag_service = RAGService(company)
        self.access_manger = AccessManager(company)


    def add_doc(self, path, access_groups, user_access_role):
        file_name = self.access_manger.create_file_access(path, access_groups, user_access_role).data
        return self.rag_service.add_doc(file_name) # TODO: Check if this raises an exception, it should

        
    def update_doc(self, old_path, new_path, new_access_groups, user_access_role):
        old_file_name = AccessManager._normalize_filename(old_path)
        new_file_name = self.access_manger.update_file_access(old_path, new_path, new_access_groups, user_access_role).data
        return self.rag_service.update_doc(old_file_name, new_file_name) # TODO: Check if this raises an exception, it should
        
    
    def delete_doc(self, path, user_access_role):
        file_name = self.access_manger.delete_file_access(path, user_access_role).data
        return self.rag_service.delete_doc(file_name) # TODO: Check if this raises an exception, it should


    def search_docs(self, question, user_access_role, n_results):
        valid_docs_data = self._find_docs(question, user_access_role, n_results)
        return OKResponse(data=valid_docs_data)
        

    def query_llm(self, question, user_access_role, n_results):
        valid_docs_data = self._find_docs(question, user_access_role, n_results)
        
        # Ask LLM to answer based on the valid sources
        return self.rag_service.query_llm(question, valid_docs_data)
    
    
    def _find_docs(self, question, user_access_role, n_results):
        # Perform semantic search for relevant document sections
        found_docs_data = self.rag_service.find_docs(question, n_results)

        # Check if user has permission for these docs
        valid_docs_data = []
        for doc_data in found_docs_data:
            try:
                self.access_manger.has_file_access(doc_data['doc_id'], user_access_role)
            except InsufficientAccessError as e:
                continue

            valid_docs_data.append(doc_data)

        return valid_docs_data
