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
        Rewrite the MongoDB command to use the appropriate view.
        
        For find operations, we modify the collection name.
        For other operations, we might need different handling.
        """
        # This is a simplified version - you may need to expand based on your commands
        if 'find' in cmd:
            # For find operations, just change the collection name
            cmd['find'] = view_name
            return cmd
        elif 'aggregate' in original_cmd:
            # For aggregate operations
            cmd['aggregate'] = view_name
            return cmd
        else:
            # For other commands, you might want different handling
            # This is a safety measure - you might want to restrict certain commands
            raise ValueError(f"Unsupported command type: {original_cmd}")


    def close(self):
        """Close the MongoDB connection."""
        self.client.close()