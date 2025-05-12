import logging
import json
import aiohttp
import jsonschema
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.exceptions import RequestValidationError
from pymongo import errors as mongo_errors
from chromadb import errors as chroma_errors
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class ErrorHandlerMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app
        self._register_exception_handlers()

    def _register_exception_handlers(self):
        self.app.add_exception_handler(HTTPException, self.http_exception_handler)
        self.app.add_exception_handler(RequestValidationError, self.validation_exception_handler)
        self.app.add_exception_handler(Exception, self.global_exception_handler)

    async def http_exception_handler(self, request: Request, exc: HTTPException):
        logger.warning(
            f"HTTPException: {exc.status_code} - {exc.detail}",
            extra={"path": request.url.path, "status_code": exc.status_code}
        )
        return HTTPException(
            status_code=exc.status_code,
            detail=exc.detail
        )

    async def validation_exception_handler(self, request: Request, exc: RequestValidationError):
        logger.warning(
            f"RequestValidationError: {str(exc)}",
            extra={"path": request.url.path, "errors": exc.errors()}
        )
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=exc.errors()
        )

    async def global_exception_handler(self, request: Request, exc: Exception):
        return await self._handle_error(request, exc)

    async def _handle_error(self, request: Request, exc: Exception):
        if isinstance(exc, KeyError) and "env" in str(exc).lower():
            raise exc

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

        return HTTPException(
            status_code=error_info["status_code"],
            detail=error_info["message"]
        )

    def _classify_error(self, exc: Exception) -> dict:
        error_map = {
            FileNotFoundError: {
                "type": "FileNotFound",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_404_NOT_FOUND,
                "message": "Requested resource not found"
            },
            PermissionError: {
                "type": "PermissionDenied",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_403_FORBIDDEN,
                "message": "Permission denied for operation"
            },
            json.JSONDecodeError: {
                "type": "InvalidJSON",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "message": "Invalid JSON data"
            },
            OSError: {
                "type": "FileSystemError",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Filesystem operation failed"
            },
            mongo_errors.ConnectionFailure: {
                "type": "DatabaseConnectionError",
                "log_level": logging.CRITICAL,
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "message": "Database connection failed"
            },
            mongo_errors.OperationFailure: {
                "type": "DatabaseOperationError",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Database operation failed"
            },
            mongo_errors.AutoReconnect: {
                "type": "DatabaseAutoReconnect",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_503_SERVICE_UNAVAILABLE,
                "message": "Temporary database issue - please retry"
            },
            mongo_errors.DuplicateKeyError: {
                "type": "DuplicateKeyError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_409_CONFLICT,
                "message": "Duplicate key violation"
            },
            chroma_errors.NoDatapointsException: {
                "type": "NoDatapointsError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_404_NOT_FOUND,
                "message": "No data points found for query"
            },
            chroma_errors.IDAlreadyExistsError: {
                "type": "DuplicateIDError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_409_CONFLICT,
                "message": "Document ID already exists"
            },
            aiohttp.ClientError: {
                "type": "HTTPClientError",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_502_BAD_GATEWAY,
                "message": "External service communication failed"
            },
            jsonschema.exceptions.ValidationError: {
                "type": "SchemaValidationError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
                "message": "Data validation failed"
            },
            ValueError: {
                "type": "ValueError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "message": "Invalid input value"
            },
            TypeError: {
                "type": "TypeError",
                "log_level": logging.WARNING,
                "status_code": status.HTTP_400_BAD_REQUEST,
                "message": "Invalid input type"
            },
            MemoryError: {
                "type": "MemoryError",
                "log_level": logging.CRITICAL,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "System resource limit reached"
            },
            RuntimeError: {
                "type": "RuntimeError",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "Unexpected runtime error"
            },
            Exception: {
                "type": "UnexpectedError",
                "log_level": logging.ERROR,
                "status_code": status.HTTP_500_INTERNAL_SERVER_ERROR,
                "message": "An unexpected error occurred"
            }
        }

        for error_class, info in error_map.items():
            if isinstance(exc, error_class):
                return info
        return error_map[Exception]


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, error_handler: ErrorHandlerMiddleware):
        super().__init__(app)
        self.error_handler = error_handler

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as exc:
            return await self.error_handler._handle_error(request, exc)