import time
import requests
from jose import jwt, JWTError
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

# PUBLIC_ROUTES = ["/route"]

security = HTTPBearer()


class TokenMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT tokens on every request."""

    def __init__(self, app, public_key_url):
        super().__init__(app)
        self.public_key_url = public_key_url
        self.public_key = None
        self._public_key_last_updated = 0  # Unix timestamp
        self._update_public_key()


    async def dispatch(self, request: Request, call_next):
        # Skip auth for public routes (optional)
        # if request.url.path in PUBLIC_ROUTES:
        #    return await call_next(request)

        # Extract token from header
        credentials: HTTPAuthorizationCredentials = await security(request)
        if not credentials:
            raise HTTPException(status_code=401, detail="Missing Authorization header")

        token = credentials.credentials

        try:
            payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
        except JWTError as e1:
            # Attempt to refetch public key and retry decoding if it's older than 1 hour
            try:
                current_time = time.time()
                if current_time - self._public_key_last_updated > 3600:
                    self._update_public_key()
                    payload = jwt.decode(token, self.public_key, algorithms=["RS256"])
                else:
                    raise e1
            except JWTError as e2:
                raise HTTPException(
                    status_code=401,
                    detail=f"Invalid token after key refresh: {str(e2)}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        request.state.company_id = payload.get("company_id")
        request.state.user_role = payload.get("user_role")

        return await call_next(request)


    def _update_public_key(self):
        """Fetch the public key from remote URL."""
        try:
            response = requests.get(self.public_key_url)
            response.raise_for_status()
            self.public_key = response.text
            self._public_key_last_updated = time.time()
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to fetch public key: {e}")