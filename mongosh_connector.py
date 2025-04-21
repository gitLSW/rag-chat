import re
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from typing import List, Dict, Any

class MongoshConnector:
    """
    A connector class that allows read-only MongoDB operations to be executed safely.
    """
    def __init__(self, mongo_db):
        """
        Initialize the connector with a specific company database.
        
        Args:
            company_id (str): The company ID used as the database name
        """
        self.db = mongo_db
        
        # List of prohibited operations/commands
        self.prohibited_commands = [
            'insert', 'update', 'delete', 'remove', 
            'create', 'drop', 'rename', 'shutdown',
            'eval', '$where', 'mapReduce'
        ]
        
        # Maximum results to return (prevent large data transfers)
        self.max_results = 100
    
    
    def _is_safe_command(self, command: str) -> bool:
        """
        Check if a command is safe (read-only and not prohibited).
        
        Args:
            command (str): The MongoDB command/query
            
        Returns:
            bool: True if safe, False if prohibited
        """
        command_lower = command.lower()
        
        # Check for prohibited operations
        for prohibited in self.prohibited_commands:
            if prohibited in command_lower:
                return False
                
        # Check for potentially dangerous operations
        dangerous_patterns = [
            r'db\.\w+\.(insert|update|delete|remove)\(',
            r'db\.(create|drop|rename)',
            r'db\.adminCommand\(',
            r'db\.shutdownServer\('
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return False
                
        return True
    
    
    def run(self, commands: str) -> Dict[str, Any]:
        """
        Execute read-only MongoDB commands safely.
        
        Args:
            commands (str): The mongosh commands to execute
            
        Returns:
            dict: A dictionary with:
                - success: bool indicating if execution was successful
                - results: list of query results (if successful)
                - error: error message (if not successful)
        """
        try:
            # Split commands by semicolons (basic command separation)
            command_list = [cmd.strip() for cmd in commands.split(';') if cmd.strip()]
            
            results = []
            
            for cmd in command_list:
                if not self._is_safe_command(cmd):
                    return {
                        'success': False,
                        'error': f'Prohibited operation in command: {cmd}'
                    }
                
                # Try to execute the command
                try:
                    # Handle find queries
                    if cmd.startswith('db.'):
                        # Extract collection and query parts
                        parts = cmd.split('.')
                        if len(parts) < 3 or not parts[2].startswith('find'):
                            return {
                                'success': False,
                                'error': f'Only find queries are allowed: {cmd}'
                            }
                            
                        collection_name = parts[1]
                        query_part = '.'.join(parts[2:])
                        
                        # Parse the query (very basic parsing)
                        if '(' in query_part and query_part.endswith(')'):
                            query_str = query_part[query_part.find('(')+1:-1]
                            
                            # Try to parse the query as JSON
                            try:
                                query = eval(query_str, {'__builtins__': None}, {})
                            except:
                                query = {}
                                
                            # Execute the query with limit
                            result = list(
                                self.db[collection_name]
                                .find(query)
                                .limit(self.max_results)
                            )
                            results.append({
                                'collection': collection_name,
                                'query': query_str,
                                'results': result
                            })
                        else:
                            return {
                                'success': False,
                                'error': f'Invalid query syntax: {cmd}'
                            }
                    else:
                        return {
                            'success': False,
                            'error': f'Only collection queries are allowed: {cmd}'
                        }
                
                except OperationFailure as e:
                    return {
                        'success': False,
                        'error': f'MongoDB operation failed: {str(e)}'
                    }
                except PyMongoError as e:
                    return {
                        'success': False,
                        'error': f'MongoDB error: {str(e)}'
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Error executing command: {str(e)}'
                    }
            
            return {
                'success': True,
                'results': results
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Unexpected error: {str(e)}'
            }