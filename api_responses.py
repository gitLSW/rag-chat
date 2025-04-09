from fastapi import HTTPException

class AccessNotFoundError(HTTPException):
    def __init__(self, file_path, detail='File not found'):
        super().__init__(404, detail)
        self.file_path = file_path

class InsufficientAccessError(HTTPException):
    def __init__(self, user_access_role, file_path, detail='Insufficient access rights, permission denied'):
        super().__init__(403, detail)
        self.user_access_role = user_access_role
        self.file_path = file_path
    
    
class ApiResponse:
    def __init__(self, status, detail, data):
        self.status = status
        self.detail = detail
        self.data = data

class OKResponse(ApiResponse):
    def __init__(self, detail='Success', data=None):
        super().__init__(200, detail, data)