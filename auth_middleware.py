# auth_middleware.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from starlette.middleware.base import BaseHTTPMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

AUTH_PUBLIC_KEY = os.getenv("AUTH_PUBLIC_KEY")

# PUBLIC_ROUTES = ["/route"]

# Security scheme for Bearer token
security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT tokens on every request."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes (optional)
        # if request.url.path in PUBLIC_ROUTES:
        #    return await call_next(request)
        
        # Extract token from header
        credentials: HTTPAuthorizationCredentials = await security(request)
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing Authorization header")
        
        try:
            # Verify token
            token = credentials.credentials
            payload = jwt.decode(
                token,
                AUTH_PUBLIC_KEY,
                algorithms=["RS256"]
            )
            
            # Attach user data to request state (optional)
            request.state.user_id = payload.get("user_id")
            request.state.role = payload.get("role")
            
        except JWTError as e:
            raise HTTPException(
                status_code=401,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return await call_next(request)