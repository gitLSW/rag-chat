import io
import logging
import json
import jsonschema
import asyncio
import aiofiles
import aiorwlock
import aiohttp
from fastapi import Request, status
from fastapi.responses import JSONResponse
from pymongo import errors as mongo_errors
from chromadb import errors as chroma_errors

logger = logging.getLogger(__name__)

def get_exc_message(exc, default_msg):
    msg = str(exc)
    return msg if msg else default_msg

def register_exception_handlers(app):
    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning(f"FileNotFoundError: {exc}")
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": get_exc_message(exc, "Requested resource not found")})

    @app.exception_handler(PermissionError)
    async def permission_error_handler(request: Request, exc: PermissionError):
        logger.error(f"PermissionError: {exc}")
        return JSONResponse(status_code=status.HTTP_403_FORBIDDEN, content={"detail": get_exc_message(exc, "Permission denied for operation")})

    @app.exception_handler(json.JSONDecodeError)
    async def json_decode_error_handler(request: Request, exc: json.JSONDecodeError):
        logger.warning(f"JSONDecodeError: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Invalid JSON data")})

    @app.exception_handler(OSError)
    async def os_error_handler(request: Request, exc: OSError):
        logger.error(f"OSError: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Filesystem operation failed")})

    @app.exception_handler(mongo_errors.ConnectionFailure)
    async def mongo_connection_failure_handler(request: Request, exc: mongo_errors.ConnectionFailure):
        logger.critical(f"MongoDB ConnectionFailure: {exc}")
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": get_exc_message(exc, "Database connection failed")})

    @app.exception_handler(mongo_errors.ServerSelectionTimeoutError)
    async def mongo_server_selection_timeout_handler(request: Request, exc: mongo_errors.ServerSelectionTimeoutError):
        logger.critical(f"MongoDB ServerSelectionTimeoutError: {exc}")
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": get_exc_message(exc, "Database connection timed out")})

    @app.exception_handler(mongo_errors.OperationFailure)
    async def mongo_operation_failure_handler(request: Request, exc: mongo_errors.OperationFailure):
        logger.error(f"MongoDB OperationFailure: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Database operation failed")})

    @app.exception_handler(mongo_errors.AutoReconnect)
    async def mongo_auto_reconnect_handler(request: Request, exc: mongo_errors.AutoReconnect):
        logger.warning(f"MongoDB AutoReconnect: {exc}")
        return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content={"detail": get_exc_message(exc, "Temporary database issue - please retry")})

    @app.exception_handler(mongo_errors.DuplicateKeyError)
    async def mongo_duplicate_key_error_handler(request: Request, exc: mongo_errors.DuplicateKeyError):
        logger.warning(f"MongoDB DuplicateKeyError: {exc}")
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": get_exc_message(exc, "Duplicate key violation")})

    @app.exception_handler(chroma_errors.IDAlreadyExistsError)
    async def chroma_id_already_exists_handler(request: Request, exc: chroma_errors.IDAlreadyExistsError):
        logger.warning(f"ChromaDB IDAlreadyExistsError: {exc}")
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": get_exc_message(exc, "Document ID already exists")})

    @app.exception_handler(aiohttp.ClientError)
    async def aiohttp_client_error_handler(request: Request, exc: aiohttp.ClientError):
        logger.error(f"Aiohttp ClientError: {exc}")
        return JSONResponse(status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": get_exc_message(exc, "External service communication failed")})

    @app.exception_handler(jsonschema.exceptions.ValidationError)
    async def jsonschema_validation_error_handler(request: Request, exc: jsonschema.exceptions.ValidationError):
        logger.warning(f"JSONSchema ValidationError: {exc}")
        return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"detail": get_exc_message(exc, "Data validation failed")})

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        logger.warning(f"ValueError: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Invalid input value")})

    @app.exception_handler(TypeError)
    async def type_error_handler(request: Request, exc: TypeError):
        logger.warning(f"TypeError: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Invalid input type")})
    
    @app.exception_handler(aiofiles.os.error)
    async def aiofiles_os_error_handler(request: Request, exc: aiofiles.os.error):
        logger.error(f"aiofiles OS Error: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Filesystem operation failed")})
    
    @app.exception_handler(IsADirectoryError)
    async def is_a_directory_error_handler(request: Request, exc: IsADirectoryError):
        logger.warning(f"IsADirectoryError: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Path is a directory, not a file")})
    
    @app.exception_handler(NotADirectoryError)
    async def not_a_directory_error_handler(request: Request, exc: NotADirectoryError):
        logger.warning(f"NotADirectoryError: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Path is not a directory")})
    
    @app.exception_handler(TimeoutError)
    async def timeout_error_handler(request: Request, exc: TimeoutError):
        logger.error(f"TimeoutError: {exc}")
        return JSONResponse(status_code=status.HTTP_504_GATEWAY_TIMEOUT, content={"detail": get_exc_message(exc, "Operation timed out")})
    
    @app.exception_handler(asyncio.TimeoutError)
    async def asyncio_timeout_error_handler(request: Request, exc: asyncio.TimeoutError):
        logger.error(f"Asyncio TimeoutError: {exc}")
        return JSONResponse(status_code=status.HTTP_504_GATEWAY_TIMEOUT, content={"detail": get_exc_message(exc, "Async operation timed out")})
    
    @app.exception_handler(BlockingIOError)
    async def blocking_io_error_handler(request: Request, exc: BlockingIOError):
        logger.error(f"BlockingIOError: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Non-blocking operation would block")})
    
    @app.exception_handler(io.UnsupportedOperation)
    async def io_unsupported_operation_handler(request: Request, exc: io.UnsupportedOperation):
        logger.warning(f"UnsupportedOperation: {exc}")
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": get_exc_message(exc, "Unsupported file operation")})
    
    @app.exception_handler(aiorwlock.Exc)
    async def aiorwlock_error_handler(request: Request, exc: aiorwlock.Exc):
        logger.error(f"RWLock Error: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Locking mechanism failed")})
        
    @app.exception_handler(MemoryError)
    async def memory_error_handler(request: Request, exc: MemoryError):
        logger.critical(f"MemoryError: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "System resource limit reached")})

    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError):
        logger.error(f"RuntimeError: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "Unexpected runtime error")})

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled Exception: {exc}")
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": get_exc_message(exc, "An unexpected error occurred")})