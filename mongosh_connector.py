import re
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from typing import Dict, Any
from pymongo import MongoClient
from pymongo.errors import OperationFailure, DuplicateKeyError, ConnectionFailure

LLM_USER_CREDENTIALS = {
    "username": "llm_readonly",
    "password": "StrongPassword123!"
}

class MongoshConnector:
    """
    A connector that executes arbitrary read-only MongoDB commands safely
    by using a dedicated read-only database user.
    """
    
    def __init__(self, company_id: str):
        """
        Initialize with read-only credentials.
        
        Args:
            company_id: The database name to connect to
        """
        # Connect with read-only user credentials
        self.client = MongoClient(
            f"mongodb://{LLM_USER_CREDENTIALS.username}:{LLM_USER_CREDENTIALS.password}@localhost:27017/",
            authSource=company_id,
            authMechanism="SCRAM-SHA-256",
            # Important security settings
            connectTimeoutMS=5000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=5000,
            retryWrites=False,
            readPreference='secondaryPreferred'  # Reduce load on primary
        )
        
        self.db = self.client[company_id]
        
        # Settings for safe execution
        self.max_time_ms = 5000  # Max query execution time
        self.max_docs = 100      # Max documents to return

        try:
            # Create user in the target database
            result = self.db.command("createUser",
                LLM_USER_CREDENTIALS.username,
                pwd=LLM_USER_CREDENTIALS.password,
                roles=[{"role": "read", "db": comapny_id}] # read-only on company's DB
            )
            print("User created successfully:", result)
            
        except OperationFailure as e:
            print(f"Operation failed: {e.details['errmsg']}")
        except DuplicateKeyError:
            print("Error: User already exists")
        except ConnectionFailure:
            print("Error: Could not connect to MongoDB")

    
    def validate(self, command: str) -> bool:
        # Use whitelist pattern
        allowed_pattern = r'^db\.\w+\.(find|aggregate|count|distinct)\(.*\)(\.\w+\(.*\))*$'
        return re.fullmatch(allowed_pattern, command) is not None


    def run(self, command: str) -> Union[Dict, List[Dict], str]:
        if not self.validate(command):
            raise ValueError("Invalid command structure")
        
        try:
            # Parse command instead of using eval()
            parts = command.split('.')
            collection = self.db[parts[1]]
            method_chain = parts[2:]
            
            result = collection
            for call in method_chain:
                if '(' in call:
                    method_name, args = call.split('(', 1)
                    args = args.rstrip(')')
                    # Safe argument parsing
                    parsed_args = json.loads(f'[{args}]')
                    result = getattr(result, method_name)(*parsed_args)
                else:
                    result = getattr(result, call)
            
            # Always enforce limits
            if isinstance(result, Cursor):
                return list(result.limit(self.max_docs).max_time_ms(self.max_time_ms))
            
            return result
            
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()