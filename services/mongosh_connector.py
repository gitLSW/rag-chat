import os
from pymongo import MongoClient
from dotenv import load_dotenv


class MongoDBConnector:
    def __init__(self, company_id):
        """Initialize MongoDB connection with company-specific credentials."""
        load_dotenv()
        
        self.company_id = company_id
        self.user_access_level = None  # Will be set in run()
        
        # Construct credentials
        username = f'llm_user_{company_id}'
        password = os.getenv(f'LLM_USER_{company_id}_PW')
        
        if not password:
            raise ValueError(f"Password not found for company {company_id}")
        
        # Connect to MongoDB
        client = MongoClient(
            host='localhost',
            port=27017,
            username=username,
            password=password,
            authSource='admin'  # Assuming admin is the auth database
        )
        
        self.db = client[company_id]
        
        # Ensure views exist for all access levels (0-9 as per your example)
        for level in range(1, 11):  # Assuming access levels 0-9
            view_name = f'access_level_{level}'
            if view_name not in self.db.list_collection_names():
                # Create view that filters documents up to this access level
                self.db.command({
                    'create': view_name,
                    'viewOn': self.base_collection,
                    'pipeline': [{
                        '$match': {
                            'access_level': {'$gte': level} # lower access_level means more access privileges 
                        }
                    }]
                })


    def run(self, json_cmd, user_access_level):
        """
        Execute a MongoDB command with proper access level restrictions.
        
        Args:
            json_cmd (dict): The MongoDB command in JSON format
            user_access_level (int): The access level of the current user
            
        Returns:
            The result of the MongoDB command execution
        """
        self.user_access_level = user_access_level
        
        # Determine the appropriate view to query
        view_name = f'access_level_{level}'
        
        try:
            # Modify the command to use the view instead of base collection
            modified_cmd = self._rewrite_command(json_cmd, view_name)
            
            # Execute the command
            result = self.db.command(modified_cmd)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Error executing MongoDB command: {str(e)}")


    def _rewrite_command(self, original_cmd, view_name):
        """
        Rewrite the MongoDB command to use the appropriate view and ensure it's read-only.
        
        Args:
            original_cmd (dict): Original MongoDB command
            view_name (str): Name of the view to use
            
        Returns:
            dict: Modified command using the view
            
        Raises:
            ValueError: If command is not read-only or can't be safely rewritten
        """
        # First make a deep copy to avoid modifying the original
        cmd = original_cmd.copy()
        
        # List of allowed read-only command types
        READ_ONLY_COMMANDS = {
            'find', 'aggregate', 'count', 'countDocuments',
            'estimatedDocumentCount', 'distinct', 'mapReduce'
        }
        
        # Get the command type (first key in the dict)
        command_type = next(iter(cmd.keys())) if cmd else None
        
        if command_type not in READ_ONLY_COMMANDS:
            raise ValueError(f"Command type '{command_type}' is not read-only or not supported")
        
        # Handle each command type appropriately
        if command_type == 'find':
            # Simple find command
            cmd['find'] = view_name
        elif command_type == 'aggregate':
            # Aggregate command - collection can be in different places
            if isinstance(cmd['aggregate'], str):
                # Standard format: {aggregate: "collection", pipeline: [...]}
                cmd['aggregate'] = view_name
            elif isinstance(cmd['aggregate'], int):
                # Rare case where aggregate might be a number (1 for $cmd)
                # We should probably reject this as it's unusual
                raise ValueError("Unsupported aggregate command format")
        elif command_type in ('count', 'countDocuments'):
            # Count commands
            cmd[command_type] = view_name
        elif command_type == 'estimatedDocumentCount':
            # Doesn't take a collection parameter, but operates on current collection
            # We'll need to modify this to work with our view approach
            raise ValueError("estimatedDocumentCount not supported with views - use countDocuments")
        elif command_type == 'distinct':
            # Distinct command
            cmd['distinct'] = view_name
        elif command_type == 'mapReduce':
            # MapReduce - though generally you should use aggregation instead
            if cmd.get('out'):
                raise ValueError("mapReduce with output collection not allowed")
            cmd['mapReduce'] = view_name
        
        # Additional safety checks
        if 'writeConcern' in cmd:
            raise ValueError("writeConcern not allowed in read-only operations")
        if 'bypassDocumentValidation' in cmd:
            raise ValueError("bypassDocumentValidation not allowed in read-only operations")
        
        return cmd
    

    def close(self):
        """Close the MongoDB connection."""
        self.client.close()