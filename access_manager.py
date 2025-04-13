import os
import json
from pathlib import Path
from filelock import FileLock
from api_responses import * 

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


    def has_file_access(self, path, user_access_role):
        standard_path = self._get_company_path(path)
        access_roles = self.access_rights.get(standard_path)
        if access_roles is None:
            raise AccessNotFoundError(file_path=standard_path)

        if not user_access_role in access_roles:
            raise InsufficientAccessError(user_access_role=user_access_role, file_path=standard_path)
        
        return OKResponse(data=standard_path)
    

    # Creates or overrides file
    def create_file_access(self, path, new_access_groups, user_access_role):
        try:
            # If the file already exists the user has no permission to override and a InsufficientAccessError gets raised
            standard_path = self.has_file_access(path, user_access_role).data
        except AccessNotFoundError as e:
            pass # Expected and correct behaviour

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                self.access_rights[standard_path] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=standard_path)


    # Updates access
    def update_file_access(self, old_path, new_path, new_access_groups, user_access_role):
        # If the access is missing a error will be raised. The file should first be created.
        old_standard_path = self.has_file_access(old_path, user_access_role).data
        new_standard_path = self._get_company_path(new_path)

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[old_standard_path]
                self.access_rights[new_standard_path] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=new_standard_path)
    
    
    def delete_file_access(self, path, user_access_role):
        standard_path = self.has_file_access(path, user_access_role).data
        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[standard_path]
                json.dump(self.access_rights, f)
        
        return OKResponse(data=standard_path)
    

    def _get_company_path(self, path):
        return get_relative_path(f"./{self.company_id}", path)


def get_relative_path(root, path):
    # Forbid these characters in paths (strictest security)
    BANNED_CHARS = {'\\', ':', '*', '?', '"', '<', '>', '|', '\0'}
    
    if any(char in path for char in BANNED_CHARS):
        raise ValueError("Invalid path")
    
    root = Path(root).resolve()

    path = Path(path)
    file_name = path.stem.split('.')[0] + path.suffix  # remove extra extensions
    path = path.with_name(file_name)
    
    full_path = (root / path).resolve(strict=False)
    
    if not str(full_path).startswith(str(root)):
        raise ValueError("Path escapes root directory")
    
    return full_path