from rag_service import RAGService
from access_manager import AccessManager
from api_responses import *
from path_normalizer import PathNormalizer

class CompanyMiddleware:

    def __init__(self, company_id):
        self.rag_service = RAGService(company_id)
        self.access_manger = AccessManager(company_id)
        self.path_normalizer = PathNormalizer(company_id)


    async def add_doc(self, source_path, dest_path, access_groups, user_access_role):
        path = self.access_manger.create_file_access(dest_path, access_groups, user_access_role).data
        return await self.rag_service.add_doc(source_path, access_groups, path) # TODO: Check if this raises an exception, it should

        
    def update_doc(self, old_path, new_path, new_json, new_access_groups, user_access_role):
        old_path = self.path_normalizer.get_relative_comany_path(old_path)
        new_path = self.access_manger.update_file_access(old_path, new_path, new_access_groups, user_access_role).data
        return self.rag_service.update_doc(old_path, new_path, new_json, new_access_groups) # TODO: Check if this raises an exception, it should
        
    
    def delete_doc(self, path, user_access_role):
        path = self.access_manger.delete_file_access(path, user_access_role).data
        return self.rag_service.delete_doc(path) # TODO: Check if this raises an exception, it should


    def search_docs(self, question, user_access_role, n_results):
        valid_docs_data = self._find_docs(question, user_access_role, n_results)
        return OKResponse(data=valid_docs_data)
        

    async def query_llm(self, question, user_access_role, n_results):
        valid_docs_data = self._find_docs(question, user_access_role, n_results)
        
        # Ask LLM to answer based on the valid sources
        return await self.rag_service.query_llm(question, valid_docs_data)
    
    
    def _find_docs(self, question, user_access_role, n_results):
        # Perform semantic search for relevant document sections
        found_docs_data = self.rag_service.find_docs(question, n_results)

        # Check if user has permission for these docs
        valid_docs_data = []
        for doc_data in found_docs_data:
            try:
                self.access_manger.has_file_access(doc_data['doc_path'], user_access_role)
            except InsufficientAccessError as e:
                continue

            valid_docs_data.append(doc_data)

        return valid_docs_data
