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

    
    
    async def run(self, commands: str) -> Dict[str, Any]:
        """
        Execute mongosh commands safely using read-only permissions.
        
        Args:
            commands: One or more mongosh commands separated by semicolons
            
        Returns:
            Dictionary with:
            - success: bool
            - results: list of command results if successful
            - error: error message if failed
        """
        try:
            # Split commands while handling semicolons in strings/objects
            command_list = self._split_commands(commands)
            results = []
            
            for cmd in command_list:
                try:
                    # First try eval-style execution (for most commands)
                    result = await self._execute_eval(cmd)
                    results.append({
                        'command': cmd,
                        'result': result
                    })
                except OperationFailure as e:
                    # If eval fails, try specific handling for certain commands
                    if 'eval' in str(e).lower():
                        result = await self._execute_special(cmd)
                        results.append({
                            'command': cmd,
                            'result': result
                        })
                    else:
                        raise
                        
            return {
                'success': True,
                'results': results
            }
            
        except PyMongoError as e:
            return {
                'success': False,
                'error': f'MongoDB Error: {str(e)}'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Execution Error: {str(e)}'
            }
    
    
    async def _execute_eval(self, command: str) -> Any:
        """
        Execute command using MongoDB's eval (with safety checks).
        """
        # Add safety wrappers to the command
        wrapped = f"""
        (function() {{
            // Safety limits
            DBQuery.shellBatchSize = {self.max_docs};
            
            // Execute the command
            try {{
                return {command};
            }} catch (e) {{
                return {{error: e.message}};
            }}
        }})()
        """
        
        # Execute with timeout
        result = await self.db.command(
            'eval',
            wrapped,
            maxTimeMS=self.max_time_ms
        )
        
        if 'retval' in result and 'error' in result['retval']:
            raise OperationFailure(result['retval']['error'])
            
        return result.get('retval')
    
    
    async def _execute_special(self, command: str) -> Any:
        """
        Handle special cases that don't work well with eval.
        """
        # Match basic CRUD patterns
        if match := re.match(r'db\.(\w+)\.(\w+)\((.*)\)', command):
            collection, method, args = match.groups()
            coll = self.db[collection]
            
            # Parse arguments safely
            try:
                parsed_args = eval(f'[{args}]', {'__builtins__': None}, {})
            except:
                parsed_args = []
                
            # Execute with appropriate method
            if method == 'find':
                query = parsed_args[0] if len(parsed_args) > 0 else {}
                projection = parsed_args[1] if len(parsed_args) > 1 else None
                cursor = coll.find(query, projection).limit(self.max_docs)
                return list(cursor)
                
            elif method == 'aggregate':
                pipeline = parsed_args[0] if len(parsed_args) > 0 else []
                if not any('$limit' in stage for stage in pipeline):
                    pipeline.append({'$limit': self.max_docs})
                return list(coll.aggregate(pipeline))
                
            elif method in ('count', 'countDocuments'):
                query = parsed_args[0] if len(parsed_args) > 0 else {}
                return coll.count_documents(query)
                
            elif method == 'distinct':
                field = parsed_args[0] if len(parsed_args) > 0 else None
                query = parsed_args[1] if len(parsed_args) > 1 else {}
                return coll.distinct(field, query)
                
        raise OperationFailure(f"Unsupported command format: {command}")
    
    
    def _split_commands(self, commands: str) -> List[str]:
        """
        Split commands by semicolons, ignoring those inside strings/objects.
        """
        parts = []
        current = []
        in_string = False
        in_object = 0
        
        for char in commands:
            if char == ';' and not in_string and in_object == 0:
                cmd = ''.join(current).strip()
                if cmd:
                    parts.append(cmd)
                current = []
            else:
                current.append(char)
                if char in ('"', "'"):
                    in_string = not in_string
                elif char == '{':
                    in_object += 1
                elif char == '}':
                    in_object -= 1
                    
        # Add final command
        final_cmd = ''.join(current).strip()
        if final_cmd:
            parts.append(final_cmd)
            
        return parts
        
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()