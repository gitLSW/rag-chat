import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class OKResponse(JSONResponse):
    def __init__(self, detail='Success', data=None):
        self.data = data
        self.detail = detail
        super().__init__(status_code=200, content=json.dumps({
            "detail": detail,
            "data": json.dumps(data)
        }))


class AccessNotFoundError(HTTPException):
    def __init__(self, file_path, detail='File not found'):
        super().__init__(status_code=404, detail=detail)
        self.file_path = file_path

class InsufficientAccessError(HTTPException):
    def __init__(self, user_access_role, detail='Insufficient access rights, permission denied'):
        super().__init__(status_code=403, detail=detail)
        self.user_access_role = user_access_role

class InsufficientFileAccessError(InsufficientAccessError):
    def __init__(self, user_access_role, file_path, detail='Insufficient file access rights, permission denied'):
        super().__init__(user_access_role, detail)
        self.file_path = file_path