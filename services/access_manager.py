import os
import json
from dotenv import load_dotenv
from fastapi import HTTPException
from filelock import FileLock
from pymongo import MongoClient
from rag_service import OKResponse


load_dotenv()
MONGO_DB_URL = os.getenv('MONGO_DB_URL')


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
        self.access_table_path = f'./{company_id}/access_table.json'
        
        # Connect to admin database (requires admin privileges)
        self.db_client = MongoClient(MONGO_DB_URL)

        if not os.path.exists(self.access_table_path):
            self.access_rights = {}
            return

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, "r") as f:
                self.access_rights = json.load(f)


    def has_doc_access(self, doc_id, user_access_role):
        access_roles = self.access_rights.get(doc_id)
        if access_roles is None:
            raise AccessNotFoundError(doc_id)

        if not user_access_role in access_roles:
            raise InsufficientAccessError(user_access_role, doc_id)
        return access_roles
    

    # Creates or overrides file
    def create_doc_access(self, doc_id, new_access_groups, user_access_role):
        try:
            # If the file already exists and the user has permission to override, it will be overwritten
            access_roles = self.has_doc_access(doc_id, user_access_role)
            # Overrides are allowed in the access_table
        except AccessNotFoundError as e:
            pass # Expected and correct behavior

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                self.access_rights[doc_id] = new_access_groups
                json.dump(self.access_rights, f)
        return access_roles


    # Updates access
    def update_doc_access(self, doc_id, new_access_groups, user_access_role):
        # If the access is missing a error will be raised. The file should first be created.
        access_roles = self.has_doc_access(doc_id, user_access_role)

        if not new_access_groups or new_access_groups == access_roles:
            return access_roles

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                self.access_rights[doc_id] = new_access_groups
                json.dump(self.access_rights, f)
        return access_roles
    
    
    def delete_doc_access(self, doc_id, user_access_role):
        access_roles = self.has_doc_access(doc_id, user_access_role)
        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[doc_id]
                json.dump(self.access_rights, f)
        return access_roles
        
    
    # TODO: Create a access_groups file !
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
        
        return OKResponse(f'Successfully added {access_group}')