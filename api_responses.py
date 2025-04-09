class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status

class AccessNotFoundError(ApiError):
    def __init__(self, file_path, message='File not found'):
        super().__init__(404, message)
        self.file_path = file_path

class InsufficientAccessError(ApiError):
    def __init__(self, user_access_role, file_path, message='Insufficient access rights, permission denied'):
        super().__init__(403, message)
        self.user_access_role = user_access_role
        self.file_path = file_path
    
    
class ApiResponse:
    def __init__(self, status, message, data):
        self.status = status
        self.message = message
        self.data = data

class OKResponse(ApiResponse):
    def __init__(self, message='Success', data=None):
        super().__init__(200, message, data)