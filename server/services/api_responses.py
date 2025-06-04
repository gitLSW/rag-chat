from fastapi import HTTPException
from fastapi.responses import JSONResponse


class OKResponse(JSONResponse):
    def __init__(self, detail="Success", data=None):
        self.data = data
        self.detail = detail
        super().__init__(status_code=200, content={
            'detail': detail,
            'data': data
        })


class ValidationError(JSONResponse):
    def __init__(self, detail, invalid_json):
        super().__init__(status_code=422, content={
                'detail': detail,
                'invalidJSON': invalid_json,
            })


class DocumentNotFoundError(HTTPException):
    def __init__(self, doc_id, detail=None):
        super().__init__(404, detail if detail else f"Doc {doc_id} doesn't exist! Create it with POST /documents first.")
        self.doc_id = doc_id


class InsufficientAccessError(HTTPException):
    def __init__(self, user_access_roles, detail="Insufficient access rights, permission denied"):
        super().__init__(status_code=403, detail=detail)
        self.user_access_roles = user_access_roles