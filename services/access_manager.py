import os
import json
from utils import get_env_var, get_company_path
from fastapi import HTTPException
from filelock import FileLock
from pymongo import MongoClient
from pymongo.errors import OperationFailure
from jsonschema import validate, ValidationError
from services.api_responses import OKResponse, InsufficientAccessError, DocumentNotFoundError

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

USER_SCHEMA = {
    "type": "object",
    "properties": {
        "userId": { "type": "string" },
        "userRoles": {
            "type": "array",
            "items": { "type": "string" },
            "minItems": 1
        }
    },
    "required": ["userId", "userRoles"],
    "additionalProperties": False
}


class AccessManager:
    def __init__(self, company_id):
        self.company_id = company_id
        self.access_data_path = get_company_path(company_id, 'access_data.json')
        
        # Connect to admin database (requires admin privileges)
        self.db_client = MongoClient(MONGO_DB_URL)

        if not os.path.exists(self.access_data_path):
            self.valid_access_groups = {'admin'}
            return

        lock = FileLock(self.access_data_path)
        with lock:
            with open(self.access_data_path, "r") as f:
                file_data = json.load(f)
                self.valid_access_groups = file_data['accessGroups']


    def create_update_user(self, user_data, curr_user_roles):
        """
        Creates or updates a MongoDB‐backed LLM user and a single per-user view,
        which filters docs to any of the user's roles. Also upserts user metadata
        and tracks all seen accessGroups.

        Args:
            user_data: dict with 'userId' and list of 'userRoles'
            curr_user_roles: list of roles of the invoking user, must include 'admin'
        """
        # Require admin rights
        if 'admin' not in curr_user_roles:
            raise InsufficientAccessError(curr_user_roles, "Admin role required")

        # Validate user_data
        try:
            validate(instance=user_data, schema=USER_SCHEMA)
        except ValidationError as e:
            raise ValueError(f"Invalid user data: {e.message}")

        user_id = user_data["userId"]
        user_roles = user_data["userRoles"]

        company_db = self.db_client[self.company_id]
        view_name = f"access_view_{user_id}"

        # 3. Drop existing view (cannot 'create' over it) and recreate with new match
        if view_name in company_db.list_collection_names():
            company_db[view_name].drop()

        company_db.command({
            "create": view_name,
            "viewOn": "docs",
            "pipeline": [
                { "$match": { "accessGroups": { "$in": user_roles } } }
            ]
        })  # :contentReference[oaicite:0]{index=0}

        # 4. Create the LLM user only if missing; catch the 'already exists' error
        username = f"llm_user_{self.company_id}_{user_id}"
        password = get_env_var(f"LLM_USER_{self.company_id}_PW")
        try:
            self.db_client['admin'].command({
                "createUser": username,
                "pwd": password,
                "roles": [{
                    "role": "read",
                    "db": self.company_id,
                    "collection": view_name
                }]
                    })
        except OperationFailure as e:
            if e.code == 51003:
                pass # User already exists — safe to ignore
            else:
                raise e

        # The 'users' collection is not currently needed, but may be useful in later versions of the server
        # Overwrite user
        company_db['users'].replace_one(
            {"userId": user_id},
            user_data,
            upsert=True
        )

        # Persist all-new roles into the global access_groups set
        lock = FileLock(self.access_data_path)
        with lock:
            self.valid_access_groups.update(user_roles) # Update all unique entries
            with open(self.access_data_path, 'w') as f:
                json.dump({"accessGroups": list(self.valid_access_groups)}, f)

        return OKResponse(f"User {user_id} successfully created or updated.")


    def has_doc_access(self, doc_id, user_access_roles):
        doc = self.db_client[self.company_id]['docs'].find_one({ '_id': doc_id })
        if not doc:
            raise DocumentNotFoundError(doc_id)
        
        if not any(user_access_role in doc['accessGroups'] for user_access_role in user_access_roles):
            raise InsufficientAccessError(user_access_roles)
        
        return doc
    
    
    def validate_new_access_groups(self, access_groups):
        # Validate new_access_groups
        access_groups = set(access_groups or [])
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