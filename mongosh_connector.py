import json
from pymongo import MongoClient
from pymongo.errors import PyMongoError, OperationFailure
from typing import Dict, Any, Union, List





import re
import json
from typing import Dict, Any, Union, List
from pymongo import MongoClient
from pymongo.errors import PyMongoError, OperationFailure

LLM_USER_CREDENTIALS = {
    "username": "llm_readonly",
    "password": "StrongPassword123!"
}

# Map JS-style operation names to PyMongo method names
JS_TO_PY_OPS = {
    "find":         "find",
    "findOne":      "find_one",
    "aggregate":    "aggregate",
    "distinct":     "distinct",
    "count":        "count_documents",
}

# Allowed aggregation stages
ALLOWED_STAGES = {
    "$match", "$project", "$group", "$sort", "$limit", "$skip",
    "$lookup", "$graphLookup", "$unionWith"
}

# Operators that allow code execution
FORBIDDEN_OPERATORS = {
    "$where", "$expr", "$function", "$accumulator"
}

class MongoshConnector:
    """
    Safely execute MongoDB commands for the LLM within the company's own database,
    using a read-only MongoDB user, with deep sanitization and explicit JS→Python op mapping.
    """

    def __init__(self, company_id: str):
        # Validate company_id to prevent URI injection
        if not re.match(r'^[A-Za-z0-9_]+$', company_id):
            raise ValueError("Invalid company_id format.")
        self.company_id = company_id

        # Connect with separate authSource parameter
        self.client = MongoClient(
            host="localhost",
            port=27017,
            username=LLM_USER_CREDENTIALS["username"],
            password=LLM_USER_CREDENTIALS["password"],
            authSource=company_id,
            authMechanism="SCRAM-SHA-256",
            tls=False,
            serverSelectionTimeoutMS=5000,
            socketTimeoutMS=30000,
            connectTimeoutMS=5000,
            retryWrites=False,
            readPreference="secondaryPreferred"
        )
        self.db = self.client[self.company_id]

        # Safety caps
        self.max_time_ms = 5000
        self.max_docs = 100
        
        try:
            # Create user in the target database (only if it does not exist)
            result = self.db.command("createUser",
                LLM_USER_CREDENTIALS['username'],
                pwd=LLM_USER_CREDENTIALS['password'],
                roles=[{"role": "read", "db": company_id}] # read-only on company's DB
            )
            print("User created successfully:", result)
            
        except OperationFailure as e:
            print(f"Operation failed: {e.details.get('errmsg', 'Unknown error')}")
        except Exception as e:
            print(f"Error: Could not connect to MongoDB or create user. {str(e)}")


    def close(self):
        self.client.close()
        

    def _check_forbidden(self, doc: Any):
        """Recursively scan for forbidden operators that execute code."""
        if isinstance(doc, dict):
            for key, val in doc.items():
                if key in FORBIDDEN_OPERATORS:
                    raise ValueError(f"Use of '{key}' is not allowed.")
                # Recurse into both key and value
                self._check_forbidden(val)
        elif isinstance(doc, list):
            for item in doc:
                self._check_forbidden(item)


    def _sanitize_aggregate_pipeline(self, pipeline: List[Dict[str, Any]]):
        if not isinstance(pipeline, list):
            raise ValueError("Pipeline must be a list.")
        for stage in pipeline:
            if not isinstance(stage, dict) or len(stage) != 1:
                raise ValueError("Each pipeline stage must be a single-key dict.")
            op, spec = next(iter(stage.items()))
            if op not in ALLOWED_STAGES:
                raise ValueError(f"Stage '{op}' is not permitted.")
            # Deep-scan for code operators
            self._check_forbidden(spec)

            # Additional checks for multi-collection stages
            if op in ("$lookup", "$graphLookup"):
                from_coll = spec.get("from")
                if not isinstance(from_coll, str) or "." in from_coll:
                    raise ValueError(f"Invalid 'from' in {op} stage.")
            if op == "$unionWith":
                # value may be string or dict
                if isinstance(spec, str):
                    if "." in spec:
                        raise ValueError("Invalid collection in $unionWith.")
                elif isinstance(spec, dict):
                    coll = spec.get("coll") or spec.get("from")
                    if not isinstance(coll, str) or "." in coll:
                        raise ValueError("Invalid 'coll' in $unionWith dict.")
                    if "pipeline" in spec:
                        self._sanitize_aggregate_pipeline(spec["pipeline"])


    def _safe_arguments(self, py_op: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        # Always override time and size limits
        args: Dict[str, Any] = {}
        if py_op == "find":
            filt = arguments.get("filter", {})
            self._check_forbidden(filt)
            args["filter"] = filt
            if "projection" in arguments:
                args["projection"] = arguments["projection"]
            if "sort" in arguments:
                args["sort"] = arguments["sort"]
            args["limit"] = self.max_docs
            args["maxTimeMS"] = self.max_time_ms

        elif py_op == "find_one":
            filt = arguments.get("filter", {})
            self._check_forbidden(filt)
            args["filter"] = filt
            if "projection" in arguments:
                args["projection"] = arguments["projection"]
            if "sort" in arguments:
                args["sort"] = arguments["sort"]
            args["maxTimeMS"] = self.max_time_ms

        elif py_op == "aggregate":
            pipeline = arguments.get("pipeline", [])
            self._sanitize_aggregate_pipeline(pipeline)
            args["pipeline"] = pipeline
            args["maxTimeMS"] = self.max_time_ms

        elif py_op == "distinct":
            # key and filter
            key = arguments.get("key")
            filt = arguments.get("filter", {})
            if not isinstance(key, str):
                raise ValueError("distinct requires a string 'key'")
            self._check_forbidden(filt)
            args["key"] = key
            args["filter"] = filt
            args["maxTimeMS"] = self.max_time_ms

        elif py_op == "count_documents":
            filt = arguments.get("filter", {})
            self._check_forbidden(filt)
            args["filter"] = filt
            args["maxTimeMS"] = self.max_time_ms

        else:
            # Should never happen if mapping is correct
            raise ValueError(f"Unsupported operation '{py_op}'")

        return args
        

    def run(self, command: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        try:
            commands = json.loads(command)
            coll_name = commands.get("collection")
            js_op = commands.get("operation")
            arguments = commands.get("arguments", {})

            if not coll_name or not js_op:
                raise ValueError("Must specify 'collection' and 'operation'.")

            # No cross-db access
            if "." in coll_name or "/" in coll_name:
                raise ValueError("Invalid collection name.")

            # Map JS op → Py op
            py_op = JS_TO_PY_OPS.get(js_op)
            if not py_op:
                raise ValueError(f"Unsupported operation '{js_op}'.")

            # Get the collection
            coll = self.db[coll_name]

            # Validate and build safe args
            if not isinstance(arguments, dict):
                raise ValueError("'arguments' must be a dict.")
            safe_args = self._safe_arguments(py_op, arguments)

            # Dispatch explicitly
            if py_op == "find":
                cursor = coll.find(**safe_args)
                return list(cursor)
            elif py_op == "find_one":
                return coll.find_one(**safe_args)
            elif py_op == "aggregate":
                cursor = coll.aggregate(**safe_args)
                return list(cursor)
            elif py_op == "distinct":
                return coll.distinct(**safe_args)
            elif py_op == "count_documents":
                return coll.count_documents(**safe_args)

        except json.JSONDecodeError:
            return {"error": "Invalid JSON format."}
        except (ValueError, OperationFailure):
            # Generic error messages to avoid leaking internals
            return {"error": "Invalid or disallowed query."}
        except PyMongoError:
            return {"error": "Database query failed."}