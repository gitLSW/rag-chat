import subprocess
import json
import re
from pymongo import MongoClient
from pymongo.errors import OperationFailure, DuplicateKeyError, ConnectionFailure


class MongoDBConnector:
    def __init__(self, mongo_db):
        self.uri = uri
        self.db = mongo_db
        try:
            # Create user in the target db
            result = db.command("createUser",
                "llm_user",
                pwd="StrongPassword123!",
                roles=[{"role": "read", "db": mongo_db.name}]
            )
            print("User created successfully:", result)
        except OperationFailure as e:
            print(f"Operation failed: {e.details['errmsg']}")
        except DuplicateKeyError:
            print("Error: User already exists")
        except ConnectionFailure:
            print("Error: Could not connect to MongoDB")
        finally:
            admin_client.close()
        
    def execute_mongosh_command(self, command: str) -> Optional[dict]:
        """
        Execute a mongosh command after validating it only targets the allowed database
        """
        # Validate the command doesn't try to access other databases
        if not self._validate_command(command):
            return {"error": "Command attempts to access unauthorized databases"}
            
        try:
            # Construct the full command with JSON output
            full_cmd = f'mongosh "{self.uri}" --quiet --eval \'EJSON.stringify({command})\''
            
            # Execute the command
            result = subprocess.run(
                full_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Parse and return the result
            return json.loads(result.stdout)
            
        except subprocess.CalledProcessError as e:
            return {"error": f"MongoDB command failed: {e.stderr}"}
        except json.JSONDecodeError:
            return {"error": "Failed to parse MongoDB response"}
            
    
    def _validate_command(self, command: str) -> bool:
        """
        Validate that the command only references the allowed database
        """
        # Check for explicit db references
        forbidden_patterns = [
            r"use\s+[^'\s]+",  # use other_db
            r"db\.getSiblingDB\([^)]+\)",  # db.getSiblingDB()
            r"db\.getMongo\(\)\.getDB\([^)]+\)"  # db.getMongo().getDB()
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, command):
                return False
                
        # If we get here, the command is probably safe
        return True
    


# Connection URI with the restricted user
MONGO_URI = "mongodb://llm_user:secure_password@localhost:27017/customer_company_db?authSource=admin"
ALLOWED_DB = "customer_company_db"

connector = MongoDBConnector(MONGO_URI, ALLOWED_DB)

# Example query from LLM
query_result = connector.execute_mongosh_command(
    "db.customers.find({status: 'active'}).limit(5)"
)

print(query_result)