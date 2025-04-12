import os
import json
from pathlib import Path
from filelock import FileLock
from api_responses import AccessNotFoundError, InsufficientAccessError, OKResponse

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
        standard_name = AccessManager._normalize_filename(file_name)
        access_roles = self.access_rights.get(standard_name)
        if access_roles is None:
            raise AccessNotFoundError(file_path=standard_name)

        if not user_access_role in access_roles:
            raise InsufficientAccessError(user_access_role=user_access_role, file_path=standard_name)
        
        return OKResponse(data=standard_name)
    

    # Creates or overrides file
    def create_file_access(self, file_name, new_access_groups, user_access_role):
        try:
            # If the file already exists the user has no permission to override and a InsufficientAccessError gets raised
            standard_name = self.has_file_access(file_name, user_access_role).data
        except AccessNotFoundError as e:
            pass # Expected and correct behaviour

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                self.access_rights[standard_name] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=standard_name)


    # Updates access
    def update_file_access(self, old_file_name, new_file_name, new_access_groups, user_access_role):
        # If the access is missing a error will be raised. The file should first be created.
        old_standard_name = self.has_file_access(old_file_name, user_access_role).data
        new_standard_name = AccessManager._normalize_filename(old_standard_name)

        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[old_standard_name]
                self.access_rights[new_standard_name] = new_access_groups
                json.dump(self.access_rights, f)

        return OKResponse(data=new_standard_name)
    
    
    def delete_file_access(self, file_name, user_access_role):
        standard_name = self.has_file_access(file_name, user_access_role).data
        lock = FileLock(self.access_table_path)
        with lock:
            with open(self.access_table_path, 'w') as f:
                del self.access_rights[standard_name]
                json.dump(self.access_rights, f)
        
        return OKResponse(data=standard_name)
    

    @staticmethod
    def _normalize_filename(path):
        path = Path(path)
        name = path.stem.split('.')[0]  # remove extra extensions
        return name + path.suffix