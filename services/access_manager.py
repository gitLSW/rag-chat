import os
import json
from fastapi import HTTPException
from filelock import FileLock

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