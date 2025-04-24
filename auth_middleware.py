from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from fastapi.exceptions import HTTPException

class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive=receive)
            token = request.headers.get("Authorization")
            if not token or not self._verify_token(token):
                response = JSONResponse(status_code=403, content={"detail": "Unauthorized"})
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)

    def _verify_token(self, token):
        # Dummy logic - replace with real validation
        return token == "expected-secure-token"