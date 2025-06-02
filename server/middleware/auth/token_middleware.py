import time
import requests
import jwt # pyjwt
from jwt import InvalidTokenError
from utils import get_env_var
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

    def __init__(self, app, public_key_url):
        super().__init__(app)
        self.public_key_url = public_key_url
        self.public_key = None
        self._public_key_last_updated = 0
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
            payload = jwt.decode(token, self.public_key, algorithms=['RS256'])
        except InvalidTokenError as e1:
            try:
                if time.time() - self._public_key_last_updated > 3600:
                    self._update_public_key()
                    payload = jwt.decode(token, self.public_key, algorithms=['RS256'])
                else:
                    raise e1
            except InvalidTokenError as e2:
                raise HTTPException(
                    status_code=401,
                    detail=e2,
                    headers={'WWW-Authenticate': 'Bearer'},
                )

        company_id = payload.get('companyId')
        user_id = payload.get('userId')

        if not company_id or not user_id:
            raise HTTPException(400, "Missing companyId or userId in token payload.")

        if user_id == 'superuser':
            # The super admin was used. Without a super admin, it wouldn't be possible to register any users (including other admins)
            user_data = { 'id': 'superuser', 'accessRoles': ['admin'] }
        else:
            user_data = db_client[company_id]['users'].find_one({ '_id': user_id })
            if not user_data:
                raise HTTPException(404, f"No user found for id {user_id}.")

        req.state.user = User(user_data, company_id)
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