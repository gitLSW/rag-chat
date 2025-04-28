# auth_middleware.py
import jwt
from fastapi import Request
from fastapi.responses import JSONResponse
from api_responses import AccessNotFoundError, InsufficientAccessError
from pymongo import MongoClient
import os

SECRET_KEY = os.getenv("SECRET_KEY", "super-secret")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

class AuthMiddleware:
    def __init__(self, app):
        self.app = app
        self.client = MongoClient(MONGO_URI)
        self.db = self.client["honesty"]  # Example DB name, adjust as needed
        self.collection = self.db["auth_tokens"]

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            token = request.headers.get("Authorization")
            if not token:
                response = JSONResponse(status_code=403, content={"detail": "Unauthorized: Missing token"})
                await response(scope, receive, send)
                return
            try:
                decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
                user_id = decoded.get("sub")
                if not user_id:
                    raise AccessNotFoundError(file_path="user_id not found in token")
                user_data = self.collection.find_one({"user_id": user_id})
                if not user_data:
                    raise AccessNotFoundError(file_path="user not found")
                scope["auth"] = {
                    "user_id": user_id,
                    "company_id": user_data.get("company_id"),
                    "role": user_data.get("role")
                }
            except jwt.ExpiredSignatureError:
                response = JSONResponse(status_code=403, content={"detail": "Token expired"})
                await response(scope, receive, send)
                return
            except Exception as e:
                response = JSONResponse(status_code=403, content={"detail": f"Unauthorized: {str(e)}"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)