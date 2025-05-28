from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, HTTPException
from utils import get_env_var

IS_PROD_ENV = get_env_var('IS_PROD_ENV')


class APIAccessMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, api_key, allowed_ips, exempt_paths=set()):
        super().__init__(app)
        self.api_key = api_key
        self.allowed_ips = allowed_ips
        self.exempt_paths = exempt_paths


    async def dispatch(self, request: Request, call_next):
        # Skip verification for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Verify IP address
        client_ip = request.client.host
        if client_ip not in self.allowed_ips and IS_PROD_ENV:
            raise HTTPException(403, "Forbidden - IP address not allowed")

        # Verify API key
        api_key = request.headers.get("x-api-key")
        if api_key != self.api_key:
            raise HTTPException(403, "Forbidden - Invalid API key")

        return await call_next(request)