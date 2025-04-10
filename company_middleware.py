from rag_service import RAGService
from access_manager import AccessManager
from api_responses import AccessNotFoundError, InsufficientAccessError


class CompanyMiddleware:
    def __init__(self, company):
        self.rag_service = RAGService(company)
        self.access_manger = AccessManager(company)


    def add_doc(self, file_name, new_doc_access_groups, user_access_role):
        path = self.access_manger.create_file_access(file_name, new_doc_access_groups, user_access_role).data
        return self.rag_service.add_doc(path) # TODO: Check if this raises an exception, it should

        
    def update_doc(self, old_file_name, new_file_name, new_access_groups, user_access_role):
        new_path = self.access_manger.update_file_access(old_file_name, new_file_name, new_access_groups, user_access_role)
        
        old_standard_name = old_file_name.split('.')[0]
        old_path = f'./{self.company}/{old_standard_name}'
        
        return self.rag_service.update_doc(old_path, new_path) # TODO: Check if this raises an exception, it should
        
    
    def delete_doc(self, file_name, user_access_role):
        path = self.access_manger.delete_file_access(file_name, user_access_role).data
        return self.rag_service.delete_doc(path) # TODO: Check if this raises an exception, it should


    # def search_docs(self, question, user_access_role, n_results):
    #     valid_docs_data = self._find_docs(question, user_access_role, n_results)
    # TODO: Gather docs for download (if not downlaoding from honesty system)
        

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
