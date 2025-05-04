import os
import json
from ..get_env_var import get_env_var
from fastapi import HTTPException
from filelock import FileLock
from pymongo import MongoClient
from rag_service import OKResponse


MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class AccessNotFoundError(HTTPException):
    def __init__(self, doc_id, detail='File not found'):
        super().__init__(status_code=404, detail=detail)
        self.doc_id = doc_id
        

class InsufficientAccessError(HTTPException):
    def __init__(self, user_access_role, doc_id, detail='Insufficient access rights, permission denied'):
        super().__init__(status_code=403, detail=detail)
        self.user_access_role = user_access_role
        self.doc_id = doc_id


class AccessManager:
    def __init__(self, company_id):
        self.company_id = company_id
        self.access_data_path = f'./{company_id}/access_data.json'
        
        # Connect to admin database (requires admin privileges)
        self.db_client = MongoClient(MONGO_DB_URL)

        if not os.path.exists(self.access_data_path):
            self.docs_access = {}
            self.valid_access_groups = {}
            return

        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, "r") as f:
                doc_data = json.load(f)
                self.docs_access = doc_data['docs_access']
                self.valid_access_groups = doc_data['access_groups']


    def create_access_group(self, access_group, user_access_role):
        """
        Creates an LLM user with restricted access to a company-specific view.
        
        Args:
            access_role: The access role level (determines which view they can access)
        """
        if user_access_role != 'admin':
            raise InsufficientAccessError(user_access_role, 'Insufficient access rights, permission denied. Admin rights required')
        
        # Create view that filters documents where the user's role is in access_groups
        view_name = f'access_view_{access_group}'
        company_db = self.db_client[self.company_id]
        if view_name not in company_db.list_collection_names():
                company_db.command({
                    'create': view_name,
                    'viewOn': 'docs',
                    'pipeline': [{
                        '$match': {
                            'access_groups': { '$in': [access_group] }
                        }
                    }]
                })
        
        # Configuration
        username = f'llm_user_{self.company_id}_{access_group}'
        password = os.getenv(f'LLM_USER_{self.company_id}_PW')
        
        self.db_client['admin'].command({
            'createUser': username,
            'pwd': password,
            'roles': [{
                'role': 'read',
                'db': self.company_id,
                'collection': view_name
            }]
        })
        
        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, 'w') as f:
                self.valid_access_groups.add(access_group)
                json.dump({
                    'docs_access': self.docs_access,
                    'access_groups': self.valid_access_groups
                }, f)

        return OKResponse(f'Successfully added {access_group}')


    def has_doc_access(self, doc_id, user_access_role):
        access_groups = self.docs_access.get(doc_id)
        if not access_groups:
            raise AccessNotFoundError(doc_id)

        if not user_access_role in access_groups:
            raise InsufficientAccessError(user_access_role, doc_id)
        return access_groups
    

    # Creates or overrides file
    def create_doc_access(self, doc_id, new_access_groups, user_access_role):
        try:
            # If the file already exists and the user has permission to override, it will be overwritten
            access_groups = self.has_doc_access(doc_id, user_access_role)
            # Overrides are allowed in the access_data
        except AccessNotFoundError as e:
            pass # Expected and correct behavior

        return self._set_doc_access(doc_id, access_groups, new_access_groups)


    # Updates access
    def update_doc_access(self, doc_id, new_access_groups, user_access_role):
        # If the access is missing a error will be raised. The file should first be created.
        access_groups = self.has_doc_access(doc_id, user_access_role)
        return self._set_doc_access(doc_id, access_groups, new_access_groups)
    
    
    def delete_doc_access(self, doc_id, user_access_role):
        access_groups = self.has_doc_access(doc_id, user_access_role)
        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, 'w') as f:
                del self.docs_access[doc_id]
                json.dump({
                    'docs_access': self.docs_access,
                    'access_groups': self.valid_access_groups
                }, f)
        return access_groups
        
    
    def _set_doc_access(self, doc_id, old_access_groups, new_access_groups):
        # Validate new_access_groups
        new_access_groups = set(new_access_groups or [])
        new_access_groups.add('admin') # Always add admin
        invalid_access_groups = [access_group for access_group in new_access_groups if access_group not in self.valid_access_groups]
        
        if len(invalid_access_groups) > 0:
            raise HTTPException(409, f"Unregistered access_groups {invalid_access_groups}. Register them at '/createAccessGroup' first.")
        
        # Set new_access_groups
        if new_access_groups == old_access_groups:
            return new_access_groups

        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, 'w') as f:
                self.docs_access[doc_id] = new_access_groups
                json.dump({
                    'docs_access': self.docs_access,
                    'access_groups': self.valid_access_groups
                }, f)
        return list(new_access_groups)