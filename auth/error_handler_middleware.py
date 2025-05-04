import logging
import os
from functools import wraps
from typing import Callable, Optional

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse
from pymongo import errors as mongo_errors
from chromadb import errors as chroma_errors
from openai import errors as openai_errors
from sentence_transformers.util import pytorch_cos_sim
import jsonschema
import json
import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('error_handler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ErrorHandlerMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        self._register_exception_handlers()
        self._register_middleware()

    def _register_exception_handlers(self):
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            logger.warning(
                f"HTTPException: {exc.status_code} - {exc.detail}",
                extra={"path": request.url.path, "status_code": exc.status_code}
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            logger.warning(
                f"RequestValidationError: {str(exc)}",
                extra={"path": request.url.path, "errors": exc.errors()}
            )
            return JSONResponse(
                status_code=422,
                content={"detail": exc.errors(), "body": exc.body},
            )

        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            return await self._handle_generic_error(request, exc)

    def _register_middleware(self):
        @self.app.middleware("http")
        async def middleware(request: Request, call_next: Callable) -> Response:
            try:
                return await call_next(request)
            except Exception as exc:
                return await self._handle_generic_error(request, exc)

    async def _handle_generic_error(self, request: Request, exc: Exception) -> JSONResponse:
        # Environment variable errors - let these crash the system
        if isinstance(exc, (KeyError,)) and "env" in str(exc).lower():
            raise exc

        # Determine log level and response based on error type
        error_info = self._classify_error(exc)
        
        logger.log(
            error_info["log_level"],
            f"{error_info['type']}: {str(exc)}",
            exc_info=exc,
            extra={
                "path": request.url.path,
                "method": request.method,
                "error_type": error_info["type"],
                "error_details": str(exc)
            }
        )

        return JSONResponse(
            status_code=error_info["status_code"],
            content={
                "detail": error_info["message"],
                "error_type": error_info["type"]
            }
        )

    def _classify_error(self, exc: Exception) -> dict:
        """Classify error and return appropriate response details"""
        error_map = {
            # File operations
            FileNotFoundError: {
                "type": "FileNotFound",
                "log_level": logging.WARNING,
                "status_code": 404,
                "message": "Requested resource not found"
            },
            PermissionError: {
                "type": "PermissionDenied",
                "log_level": logging.ERROR,
                "status_code": 403,
                "message": "Permission denied for operation"
            },
            json.JSONDecodeError: {
                "type": "InvalidJSON",
                "log_level": logging.WARNING,
                "status_code": 400,
                "message": "Invalid JSON data"
            },
            OSError: {
                "type": "FileSystemError",
                "log_level": logging.ERROR,
                "status_code": 500,
                "message": "Filesystem operation failed"
            },
            
            # Database operations
            mongo_errors.ConnectionFailure: {
                "type": "DatabaseConnectionError",
                "log_level": logging.CRITICAL,
                "status_code": 503,
                "message": "Database connection failed"
            },
            mongo_errors.OperationFailure: {
                "type": "DatabaseOperationError",
                "log_level": logging.ERROR,
                "status_code": 500,
                "message": "Database operation failed"
            },
            mongo_errors.AutoReconnect: {
                "type": "DatabaseAutoReconnect",
                "log_level": logging.WARNING,
                "status_code": 503,
                "message": "Temporary database issue - please retry"
            },
            mongo_errors.DuplicateKeyError: {
                "type": "DuplicateKeyError",
                "log_level": logging.WARNING,
                "status_code": 409,
                "message": "Duplicate key violation"
            },
            chroma_errors.NoDatapointsException: {
                "type": "NoDatapointsError",
                "log_level": logging.WARNING,
                "status_code": 404,
                "message": "No data points found for query"
            },
            chroma_errors.IDAlreadyExistsError: {
                "type": "DuplicateIDError",
                "log_level": logging.WARNING,
                "status_code": 409,
                "message": "Document ID already exists"
            },
            
            # External services
            openai_errors.APIError: {
                "type": "LLMAPIError",
                "log_level": logging.ERROR,
                "status_code": 502,
                "message": "AI service API error"
            },
            openai_errors.APIConnectionError: {
                "type": "LLMConnectionError",
                "log_level": logging.ERROR,
                "status_code": 503,
                "message": "Could not connect to AI service"
            },
            openai_errors.RateLimitError: {
                "type": "LLMRateLimit",
                "log_level": logging.WARNING,
                "status_code": 429,
                "message": "AI service rate limit exceeded"
            },
            openai_errors.Timeout: {
                "type": "LLMTimeout",
                "log_level": logging.WARNING,
                "status_code": 504,
                "message": "AI service timeout"
            },
            aiohttp.ClientError: {
                "type": "HTTPClientError",
                "log_level": logging.ERROR,
                "status_code": 502,
                "message": "External service communication failed"
            },
            
            # Document processing
            jsonschema.exceptions.ValidationError: {
                "type": "SchemaValidationError",
                "log_level": logging.WARNING,
                "status_code": 422,
                "message": "Data validation failed"
            },
            ValueError: {
                "type": "ValueError",
                "log_level": logging.WARNING,
                "status_code": 400,
                "message": "Invalid input value"
            },
            TypeError: {
                "type": "TypeError",
                "log_level": logging.WARNING,
                "status_code": 400,
                "message": "Invalid input type"
            },
            
            # Resource errors
            MemoryError: {
                "type": "MemoryError",
                "log_level": logging.CRITICAL,
                "status_code": 500,
                "message": "System resource limit reached"
            },
            RuntimeError: {
                "type": "RuntimeError",
                "log_level": logging.ERROR,
                "status_code": 500,
                "message": "Unexpected runtime error"
            },
            
            # Default catch-all
            Exception: {
                "type": "UnexpectedError",
                "log_level": logging.ERROR,
                "status_code": 500,
                "message": "An unexpected error occurred"
            }
        }

        # Find the most specific matching error class
        for error_class, info in error_map.items():
            if isinstance(exc, error_class):
                return info
        
        return error_map[Exception]

    @staticmethod
    def critical_env_vars(*env_vars: str) -> None:
        """Decorator to ensure critical environment variables are present"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                missing = [var for var in env_vars if not os.getenv(var)]
                if missing:
                    raise EnvironmentError(
                        f"Missing critical environment variables: {', '.join(missing)}"
                    )
                return await func(*args, **kwargs)
            return wrapper
        return decorator