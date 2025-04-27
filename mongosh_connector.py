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
                roles=[{"role": "read", "db": "customer_company_db"}]
            )
            print("User created successfully:", result)
            
        except OperationFailure as e:
            print(f"Operation failed: {e.details['errmsg']}")
        except DuplicateKeyError:
            print("Error: User already exists")
        except ConnectionFailure:
            print("Error: Could not connect to MongoDB")

    
    def validate(self, command: str) -> bool:
        """
        Validate that the command is safe to execute in mongosh context.
        
        Args:
            command: The mongosh command to validate
            
        Returns:
            bool: True if command is valid, False otherwise
        """
        # Disallowed patterns that could potentially cause harm or access other databases
        disallowed_patterns = [
            r'\.\.',  # Parent directory references
            r'process\.',  # Access to process object
            r'load\s*\(',  # Loading external files
            r'eval\s*\(',  # eval function
            r'new\s+Date\s*\(',  # Potential date manipulation
            r'system\.',  # System access
            r'sleep\s*\(',  # Sleep/delay operations
            r'db\.getSiblingDB\s*\(',  # Accessing other databases
            r'db\.adminCommand\s*\(',  # Admin commands
            r'db\.\w+\.(insert|update|remove|delete|drop|create|rename)',  # Write operations
            r'shutdown\s*\(',  # Shutdown commands
            r'while\s*\(',  # Potential infinite loops
            r'for\s*\(',  # Potential heavy loops
            r'function\s*\(',  # Function declaration
        ]
        
        # Check for disallowed patterns
        for pattern in disallowed_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False
                
        # Check if command tries to access a different database
        if re.search(r'db\s*=\s*(?!db\b)[\w\.]+', command):
            return False
            
        return True

    def run(self, command: str) -> Union[Dict[str, Any], List[Dict[str, Any]], str]:
        """
        Execute a mongosh command safely after validation.
        
        Args:
            command: The mongosh command to execute
            
        Returns:
            The result of the command execution or an error message
            
        Raises:
            ValueError: If command is invalid or unsafe
            PyMongoError: For MongoDB-related errors
        """
        # First, validate the command
        if not self.validate(command):
            raise ValueError("Command contains potentially unsafe operations")
        
        try:
            # Special handling for find() operations to apply limits
            if '.find(' in command and not ('.limit(' in command or '.toArray(' in command):
                # If it's a find command without limit, we'll add our own
                modified_command = command.replace('.find(', f'.find().limit({self.max_docs})', 1)
                command = modified_command
            
            # Execute the command in eval context (simulating mongosh)
            # Note: In production, you might want to parse and use proper MongoDB API calls instead
            result = self.db.eval(f"function() {{ return {command}; }}", max_time_ms=self.max_time_ms)
            
            # Convert MongoDB cursor to list if needed
            if hasattr(result, 'alive'):
                result = list(result)[:self.max_docs]
                
            return result
            
        except OperationFailure as e:
            # Handle MongoDB operation errors
            if 'not authorized' in str(e):
                return "Error: Permission denied - read-only access"
            return f"MongoDB operation error: {e.details.get('errmsg', str(e))}"
        except PyMongoError as e:
            return f"MongoDB error: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
        
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()