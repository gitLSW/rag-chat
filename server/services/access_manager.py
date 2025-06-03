import logging
from utils import get_env_var
from fastapi import HTTPException
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from jsonschema import validate, ValidationError
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

db_client = MongoClient(MONGO_DB_URL)

logger = logging.getLogger(__name__)


class User:
    def __init__(self, user_data, company_id):
        self.id = user_data.get('id')
        self.access_roles = user_data.get('accessRoles')
        self.company_id = company_id
        
        if not self.id or not self.access_roles or not company_id:
            raise HTTPException(400, "User must contain an 'id' and 'accessRoles'")


    def assert_admin(self):
        if 'admin' not in self.access_roles:
            raise InsufficientAccessError(self.access_roles, "Insufficient access rights, permission denied. Admin rights required")
        

    def has_doc_access(self, doc_id):
        doc = db_client[self.company_id]['docs'].find_one({ '_id': doc_id })
        if not doc:
            raise DocumentNotFoundError(doc_id)
        
        doc_access_groups = doc.get('accessGroups')
        if not doc_access_groups:
            return doc # Allow everybody for a doc with access_groups None
        
        if not any(user_access_role in doc_access_groups for user_access_role in self.access_roles):
            raise InsufficientAccessError(self.access_roles)
        
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


    def create_overwrite_user(self, user_data, curr_user):
        """
        Creates or updates a MongoDB‐backed LLM user and a single per-user view,
        which filters docs to any of the user's roles. Also upserts user metadata
        and tracks all seen accessGroups.

        Args:
            user_data: dict with 'id' and list of 'accessRoles'
            curr_user: the invoking user, must be 'admin'
        """
        curr_user.assert_admin() # Require admin rights

        user = User(user_data, self.company_id) # validates userData
        
        view_name = f'access_view_{user.id}'

        # Drop existing view (cannot 'create' over it) and recreate with new match
        if view_name in self.company_db.list_collection_names():
            self.company_db[view_name].drop()

        self.company_db.command({
            'create': view_name,
            'viewOn': 'docs',
            'pipeline': [{
                '$match': {
                    '$or': [
                        { 'accessGroups': { '$in': user.access_roles } },
                        { 'accessGroups': None },
                        { 'accessGroups': { '$exists': False } }
                    ]
                }
            }]
        })  # :contentReference[oaicite:0]{index=0}

        # Create the LLM user only if missing; catch the 'already exists' error
        llm_username = f'llm_user_{self.company_id}_{user.id}'
        password = get_env_var(f'LLM_USER_PW')
        try:
            db_client['admin'].command({
                'createUser': llm_username,
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

        # Create or overwrite user
        self.company_db['users'].replace_one({ '_id': user.id }, user_data, upsert=True)
        
        self.valid_access_groups.update(user.access_roles) # Update all unique entries

        logger.info(f"User '{curr_user.id}' at '{curr_user.company_id}' created or overwrote user '{user.id}'")

        return OKResponse(f"User {user.id} successfully created or updated.", user_data)
    

    def delete_user(self, user_id, curr_user):
        """
        Deletes a MongoDB‐backed LLM user and their per-user view.

        Args:
            user_id: The ID of the user to delete.
            curr_user: The invoking user, must be 'admin'.
        """
        curr_user.assert_admin()  # Require admin rights

        # Drop the user's view if it exists
        view_name = f'access_view_{user_id}'
        if view_name in self.company_db.list_collection_names():
            self.company_db[view_name].drop()

        # Drop the LLM user from MongoDB admin db
        llm_username = f'llm_user_{self.company_id}_{user_id}'
        try:
            db_client['admin'].command({'dropUser': llm_username})
        except OperationFailure as e:
            if e.code == 11:  # UserNotFound
                pass  # It's okay, user already gone
            else:
                raise e

        # Delete the user metadata from the 'users' collection
        res = self.company_db['users'].delete_one({'id': user_id})
        if res.deleted_count == 0:
            raise HTTPException(404, f"No user with id {user_id} found.")

        logger.info(f"User '{curr_user.id}' at '{curr_user.company_id}' deleted user '{user_id}'")

        return OKResponse(f"User {user_id} successfully deleted.")

    
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