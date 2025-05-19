import time
import requests
from utils import get_env_var
from jose import jwt, JWTError
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from pymongo import MongoClient
from services.access_manager import User

IS_PROD_ENV = get_env_var('IS_PROD_ENV')
if not IS_PROD_ENV:
    JWT_PUBLIC_KEY = get_env_var('JWT_PUBLIC_KEY')

MONGO_DB_URL = get_env_var('MONGO_DB_URL')

db_client = MongoClient(MONGO_DB_URL)
security = HTTPBearer()


class TokenMiddleware(BaseHTTPMiddleware):
    """Middleware to validate JWT tokens on every request."""

    def __init__(self, app, public_key_url):
        super().__init__(app)
        self.public_key_url = public_key_url
        self.public_key = None
        self._public_key_last_updated = 0  # Unix timestamp
        self._update_public_key()


    async def dispatch(self, req: Request, call_next):
        # Skip auth for public routes (optional)
        # if req.url.path in PUBLIC_ROUTES:
        #    return await call_next(req)

        # Extract token from header
        credentials: HTTPAuthorizationCredentials = await security(req)
        if not credentials:
            raise HTTPException(401, "Missing Authorization header")

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
                    detail=f"Invalid token after key refresh: {e2}",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        company_id = payload.get("companyId")
        if not company_id:
            raise HTTPException(400, f"JWT Token was decoded, but the payload was missing the companyId.")
        
        user_id = payload.get("userId")
        if not user_id:
            raise HTTPException(400, f"JWT Token was decoded, but the payload was missing the userId.")
        
        user = db_client[company_id]['users'].find_one({ '_id': user_id })
        if not user:
            raise HTTPException(404, f"No user found for id {user_id}. Register new users first.")
        
        req.state.company_id = company_id
        req.state.user = User(user, self.company_id)

        return await call_next(req)


    def _update_public_key(self):
        """Fetch the public key from remote URL."""
        if not IS_PROD_ENV:
            self.public_key = JWT_PUBLIC_KEY
            self._public_key_last_updated = time.time()
            return

        try:
            response = requests.get(self.public_key_url)
            response.raise_for_status()
            self.public_key = response.text
            self._public_key_last_updated = time.time()
        except requests.RequestException as e:
            raise HTTPException(500, f"Failed to fetch public key: {e}")