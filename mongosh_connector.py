import json
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from typing import Dict, Any, Union, List

LLM_USER_CREDENTIALS = {
    "username": "llm_readonly",
    "password": "StrongPassword123!"
}

class MongoshConnector:
    """
    Safely execute MongoDB commands for the LLM within the company's own database,
    using a read-only MongoDB user.
    """

    def __init__(self, company_id: str):
        """
        Connect to the company's database using read-only credentials.
        """
        self.company_id = company_id
        
        self.client = MongoClient(
            f"mongodb://{LLM_USER_CREDENTIALS['username']}:{LLM_USER_CREDENTIALS['password']}@localhost:27017/",
            authSource=company_id,
            authMechanism="SCRAM-SHA-256",
            tls=True,
            tlsAllowInvalidCertificates=False,
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000,
            connectTimeoutMS=5000,
            retryWrites=False,
            readPreference="secondaryPreferred"
        )
        
        self.db = self.client[self.company_id]

        # Settings for safe query execution
        self.max_time_ms = 5000
        self.max_docs = 100

    def _sanitize_aggregate_pipeline(self, pipeline: List[Dict[str, Any]]):
        """
        Validate aggregation pipeline to prevent dangerous operations like $out, $merge, or cross-database $lookup.
        """
        for stage in pipeline:
            if not isinstance(stage, dict):
                raise ValueError("Invalid aggregation pipeline stage format.")

            for key, value in stage.items():
                # Disallow stages that write
                if key in ("$out", "$merge"):
                    raise ValueError(f"Aggregation stage '{key}' is not allowed.")

                # Validate $lookup stages
                if key == "$lookup":
                    if not isinstance(value, dict):
                        raise ValueError("Invalid $lookup format.")
                    from_collection = value.get("from", "")
                    if "." in from_collection:
                        raise ValueError("Cross-database $lookup is not allowed.")
    
                # Disallow $function usage
                if key == "$project" or key == "$addFields":
                    if isinstance(value, dict):
                        if any("$function" in v for v in value.values() if isinstance(v, dict)):
                            raise ValueError("$function usage inside pipeline is not allowed.")

    def _safe_arguments(self, operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Safely adjust and validate arguments for supported operations.
        """
        safe_args = arguments.copy()

        # Enforce maxTimeMS
        if operation in ("find", "find_one", "aggregate", "distinct", "count_documents"):
            safe_args.setdefault("maxTimeMS", self.max_time_ms)

        # Enforce limits on cursor-returning operations
        if operation == "find":
            safe_args.setdefault("limit", self.max_docs)
        if operation == "aggregate":
            pipeline = safe_args.get("pipeline", [])
            if not isinstance(pipeline, list):
                raise ValueError("'pipeline' must be a list.")
            self._sanitize_aggregate_pipeline(pipeline)

        return safe_args

    def run(self, command: Dict[str, Any]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Execute a safe MongoDB command.
        """
        try:
            collection_name = command.get("collection")
            operation = command.get("operation")
            arguments = command.get("arguments", {})

            if not collection_name or not operation:
                raise ValueError("Missing required fields: 'collection' and 'operation'.")

            # Validate collection name (no cross-database access)
            if "." in collection_name:
                raise ValueError("Cross-database access is not allowed.")

            # Access collection
            collection = self.db[collection_name]

            # Check operation exists
            if not hasattr(collection, operation):
                raise ValueError(f"Operation '{operation}' not available.")

            method = getattr(collection, operation)

            if not isinstance(arguments, dict):
                raise ValueError("'arguments' must be a dictionary.")

            safe_args = self._safe_arguments(operation, arguments)

            result = method(**safe_args)

            # If cursor, enforce limits
            if hasattr(result, "limit") and hasattr(result, "max_time_ms"):
                result = result.limit(self.max_docs).max_time_ms(self.max_time_ms)
                return list(result)

            # Otherwise, return result directly
            return result

        except PyMongoError:
            return {"error": "Database query failed."}
        except Exception:
            return {"error": "Invalid query or unsafe operation detected."}

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()