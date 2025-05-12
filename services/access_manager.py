import os
import json
from get_env_var import get_env_var
from fastapi import HTTPException
from filelock import FileLock
from pymongo import MongoClient
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class AccessManager:
    def __init__(self, company_id):
        self.company_id = company_id
        self.access_data_path = f'./companies/{company_id}/access_data.json'
        
        # Connect to admin database (requires admin privileges)
        self.db_client = MongoClient(MONGO_DB_URL)

        if not os.path.exists(self.access_data_path):
            self.valid_access_groups = {'admin'}
            return

        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, "r") as f:
                file_data = json.load(f)
                self.valid_access_groups = file_data['access_groups']


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
                            'accessGroups': { '$in': [access_group] }
                        }
                    }]
                })

        # Configuration
        username = f'llm_user_{self.company_id}_{access_group}'
        password = get_env_var(f'LLM_USER_{self.company_id}_PW')

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
                    'access_groups': self.valid_access_groups
                }, f)

        return OKResponse(f'Successfully added {access_group}')


    def has_doc_access(self, doc_id, user_access_role):
        doc = self.db_client[self.company_id]['docs'].find_one({ '_id': doc_id })
        if not doc:
            raise DocumentNotFoundError(doc_id)
        
        if not user_access_role in doc['accessGroups']:
            raise InsufficientAccessError(user_access_role)
        
        return doc
    
    
    def validate_new_access_groups(self, access_groups):
        # Validate new_access_groups
        access_groups = set(access_groups or [])
        access_groups.add('admin') # Always add admin
        invalid_access_groups = [access_group for access_group in access_groups if access_group not in self.valid_access_groups]
        
        if len(invalid_access_groups) > 0:
            raise HTTPException(409, f"Unregistered access_groups {invalid_access_groups}. Register them with POST '/accessGroups' first.")
        
        return list(access_groups)
    

access_managers = {}
def get_access_manager(company_id):
    access_manager = access_managers.get(company_id)
    if access_manager:
        return access_manager
    
    access_manager = AccessManager(company_id)
    access_managers[company_id] = access_manager
    return access_manager