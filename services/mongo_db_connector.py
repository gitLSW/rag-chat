import json
from pymongo import MongoClient
from ..get_env_var import get_env_var, MissingEnvVarError


MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class MongoDBConnector:
    
    def __init__(self, company_id):
        """Initialize MongoDB connection with company-specific credentials."""
        self.company_id = company_id
        

    def run(self, json_cmd, user_access_role):
        """
        Execute a MongoDB command with proper access role restrictions.
        
        Args:
            json_cmd (dict): The MongoDB command in JSON format
            user_access_role (str): The access role of the current user
            
        Returns:
            The result of the MongoDB command if the execution was successful else None.
        """
        if isinstance(json_cmd, str):
            try:
                json_cmd = json.loads(json_cmd)
            except json.JSONDecodeError as e:
                # Invalid json_cmd
                return None
        
        try:
            # Construct credentials
            username = f'llm_user_{self.company_id}_{user_access_role}'
            password = get_env_var(f'LLM_USER_{self.company_id}_PW')
            
            if not password:
                # f"Password not found for company {company_id}"
                return None
            
            # Connect to MongoDB
            client = MongoClient(
                MONGO_DB_URL,
                username=username,
                password=password,
                authSource='admin'  # Assuming admin is the auth database
            )
        except MissingEnvVarRError as e:
            # TODO: log error
            pass
        except Exception as e:
            # TODO: log error: f"Error logging LLM into mongo_db for company {self.company_id} with user {username}"
            pass
        finally:
            return None 
        
        company_db = client[company_id]
        
        # Determine the appropriate view to query
        view_name = f'access_view_{user_access_role}'
        if view_name not in company_db.list_collection_names():
            # TODO: log error: f'The no mongoDB view found for user_role "{user_ccess_role}". Add it first at /addAccessGroup'
            return None
            
        try:
            # Modify the command to use the view instead of base collection
            modified_cmd = self._rewrite_command(json_cmd, view_name)
            
            # Execute the command
            result = company_db.command(modified_cmd)
            return result
        except Exception as e:
            # # TODO: log error: f"Error executing MongoDB command:\nmongo command: {json.dumps(json_cmd)}\n{e}"
            return None


    def _rewrite_command(self, original_cmd, view_name):
        """
        Minimal command rewriter that handles:
        - find, aggregate, count, distinct, and mapReduce
        - Injects view name into collection references
        - Rejects all other operations
        """
        cmd = original_cmd.copy()
        
        # Supported operations where operation_name == collection_field
        READ_OPS = { 'find', 'aggregate', 'count', 'distinct', 'mapReduce' }
        
        # Find first matching operation
        for op in READ_OPS:
            if op in cmd:
                # Replace collection name if it's a direct string reference
                if isinstance(cmd[op], str):
                    cmd[op] = view_name
                
                # Special handling for aggregate pipelines with $lookup
                if op == 'aggregate' and 'pipeline' in cmd:
                    self._rewrite_pipeline_collections(cmd['pipeline'], view_name)
                
                return cmd
        
        raise ValueError(f"Unsupported operation. Allowed: {READ_OPS}")
    
    
    def _rewrite_pipeline_collections(self, pipeline, view_name):
        """Handle collection references in aggregation pipelines"""
        if not isinstance(pipeline, list):
            return
        
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
            
            # Rewrite $lookup/$graphLookup collection references
            for lookup_op in ('$lookup', '$graphLookup'):
                if lookup_op in stage and isinstance(stage[lookup_op].get('from'), str):
                    stage[lookup_op]['from'] = view_name
    
    
    def close(self):
        """Close the MongoDB connection."""
        self.client.close()