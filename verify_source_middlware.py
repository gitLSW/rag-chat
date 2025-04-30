from fastapi import Request, HTTPException
from fastapi.middleware import Middleware
from typing import Set, Optional

class VerifySourceMiddleware:
    """
    Secures all endpoints (except the exempt ones) by
    verifying if the requests come from whitelisted IPs and
    if they have the correct API key in the header.
    """
    def __init__(
        self,
        allowed_ips: Set[str],
        api_key: str,
        exempt_paths: Optional[Set[str]] = None
    ):
        self.allowed_ips = allowed_ips
        self.api_key = api_key
        self.exempt_paths = exempt_paths or {"/health", "/docs", "/openapi.json"}

    async def __call__(self, request: Request, call_next):
        # Skip verification for exempt paths
        if request.url.path in self.exempt_paths:
            return await call_next(request)

        # Verify IP address
        client_ip = request.client.host
        if client_ip not in self.allowed_ips:
            raise HTTPException(
                status_code=403,
                detail="Forbidden - IP address not allowed"
            )

        # Verify API key
        api_key = request.headers.get("x-api-key")
        if api_key != self.api_key:
            raise HTTPException(
                status_code=403,
                detail="Forbidden - Invalid API key"
            )

        return await call_next(request)