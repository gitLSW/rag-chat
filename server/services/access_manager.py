from utils import get_env_var
from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from jsonschema import validate, ValidationError
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

db_client = MongoClient(MONGO_DB_URL)


class User:
    def __init__(self, user_data, company_id):
        self.user_id = user_data['id']
        self.user_roles = user_data['accessRoles']
        self.company_id = company_id


    def assert_admin(self):
        if 'admin' not in self.user_roles:
            raise InsufficientAccessError(self.user_roles, "Insufficient access rights, permission denied. Admin rights required")
        

    def has_doc_access(self, doc_id):
        doc = db_client[self.company_id]['docs'].find_one({ '_id': doc_id })
        if not doc:
            raise DocumentNotFoundError(doc_id)
        
        doc_access_groups = doc.get('accessGroups')
        if not doc_access_groups:
            return doc # Allow everybody for a doc with access_groups None
        
        if not any(user_access_role in doc_access_groups for user_access_role in self.user_roles):
            raise InsufficientAccessError(self.user_roles)
        
        return doc


class AccessManager:
    def __init__(self, company_id):
        self.company_id = company_id
        
        # Connect to admin database (requires admin privileges)
        self.company_db = db_client[self.company_id]
        
        # Get valid_access_groups from all users
        users_db = self.company_db['users']
        result = list(users_db.aggregate([
            {'$unwind': '$accessRoles'},
            {'$group': {'_id': None, 'allRoles': {'$addToSet': '$accessRoles'}}}
        ]))
        
        self.valid_access_groups = set(result[0]['allRoles']) if result else set()
        self.valid_access_groups.add('admin')


    def create_update_user(self, user_data, curr_user):
        """
        Creates or updates a MongoDB‐backed LLM user and a single per-user view,
        which filters docs to any of the user's roles. Also upserts user metadata
        and tracks all seen accessGroups.

        Args:
            user_data: dict with 'id' and list of 'accessRoles'
            curr_user: the invoking user, must be 'admin'
        """
        curr_user.assert_admin() # Require admin rights

        user = User(user_data, self.company_id)

        self.company_db = db_client[self.company_id]
        view_name = f'access_view_{user.user_id}'

        # Drop existing view (cannot 'create' over it) and recreate with new match
        if view_name in self.company_db.list_collection_names():
            self.company_db[view_name].drop()

        self.company_db.command({
            'create': view_name,
            'viewOn': 'docs',
            'pipeline': [{
                '$match': {
                    '$or': [
                        { 'accessGroups': { '$in': user.user_roles } },
                        { 'accessGroups': None },
                        { 'accessGroups': { '$exists': False } }
                    ]
                }
            }]
        })  # :contentReference[oaicite:0]{index=0}

        # Create the LLM user only if missing; catch the 'already exists' error
        username = f'llm_user_{self.company_id}_{user.user_id}'
        password = get_env_var(f'LLM_USER_{self.company_id}_PW')
        try:
            db_client['admin'].command({
                'createUser': username,
                'pwd': password,
                'roles': [{
                    'role': 'read',
                    'db': self.company_id,
                    'collection': view_name
                }]
            })
        except OperationFailure as e:
            if e.code == 51003:
                pass # User already exists — safe to ignore
            else:
                raise e

        # The 'users' collection is not currently needed, but may be useful in later versions of the server
        # Overwrite user
        self.company_db['users'].replace_one(
            {'id': user.user_id},
            user_data,
            upsert=True
        )
        
        self.valid_access_groups.update(user.user_roles) # Update all unique entries

        return OKResponse(f"User {user.user_id} successfully created or updated.")
    
    
    def validate_new_access_groups(self, access_groups):
        # Validate new_access_groups
        if not access_groups:
            return None
        
        access_groups = set(access_groups)
        access_groups.add('admin') # Always add admin
        invalid_access_groups = []
        for access_group in access_groups:
            if access_group not in self.valid_access_groups:
                invalid_access_groups.append(access_group)
        
        if len(invalid_access_groups) > 0:
            raise HTTPException(409, f"Unknown access_groups {invalid_access_groups}. Register them with POST /accessGroups first.")
        
        return list(access_groups)
    

access_managers = {}
def get_access_manager(company_id):
    access_manager = access_managers.get(company_id)
    if access_manager:
        return access_manager
    
    access_manager = AccessManager(company_id)
    access_managers[company_id] = access_manager
    return access_manager