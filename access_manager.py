import os
import json
from filelock import FileLock
from api_responses import ApiError, AccessNotFoundError, InsufficientAccessError, ApiResponse, OKResponse

class AccessManager:
    def __init__(self, company):
        self.company = company
        self.access_table_path = f'./{company}/access_table.json'
        if not os.path.exists(self.access_table_path):
            self.access_rights = {}
            return

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, "r") as f:
                self.access_rights = json.load(f)

    def has_file_access(self, file_name, user_access_role):
        standard_name = file_name.split('.')[0]
        path = f'./{self.company}/{standard_name}'

        access_roles = self.access_rights.get(path)
        if access_roles is None:
            raise AccessNotFoundError(file_path=path)

        if not user_access_role in access_roles:
            raise InsufficientAccessError(user_access_role=user_access_role, file_path=path)
        
        return OKResponse(data=path)
    
    # Creates or overrides file
    def create_file_access(self, new_file_name, new_access_groups, user_access_role):
        try:
            self.has_file_access(new_file_name, user_access_role).data
        except AccessNotFoundError as e:
            pass # Expected and correct behaviour
        except InsufficientAccessError as e:
            raise e # The file already exists and the user has no permission to override
        
        standard_name = new_file_name.split('.')[0]
        new_path = f'./{self.company}/{standard_name}'

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                self.access_rights[new_path] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=new_path)

    # Updates access
    def update_file_access(self, old_file_name, new_file_name, new_access_groups, user_access_role):
        try:
            old_path = self.has_file_access(old_file_name, user_access_role).data
        except Exception as e:
            raise e # If the access is missing, the file should first be created
        
        new_standard_name = new_file_name.split('.')[0]
        new_path = f'./{self.company}/{new_standard_name}'

        if os.path.exists(new_path):
            raise FileExistsError()

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[old_path]
                self.access_rights[new_path] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=new_path)
    
    
    def delete_file_access(self, file_name, user_access_role):
        try:
            path = self.has_file_access(file_name, user_access_role).data
        except Exception as e:
            raise e
        
        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[path]
                json.dump(self.access_rights, f)
        
        return OKResponse(data=path)