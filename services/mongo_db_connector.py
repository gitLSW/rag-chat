import json
import logging
from pymongo import MongoClient
from utils import get_env_var, MissingEnvVarError

logger = logging.getLogger(__name__)

MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class MongoDBConnector:
    
    def __init__(self, company_id):
        """Initialize MongoDB connection with company-specific credentials."""
        self.company_id = company_id
        

    def run(self, json_cmd, user_access_roles):
        """
        Execute a MongoDB command with proper access role restrictions.
        
        Args:
            json_cmd (dict): The MongoDB command in JSON format
            user_access_roles (List[str]): The access roles of the current user
            
        Returns:
            The result of the MongoDB command if the execution was successful else None.
        """
        if isinstance(json_cmd, str):
            try:
                json_cmd = json.loads(json_cmd)
            except json.JSONDecodeError as e:
                # Invalid json_cmd
                return None, None

        try:
            # Construct credentials
            username = f'llm_user_{self.company_id}'
            password = get_env_var(f'LLM_USER_{self.company_id}_PW')
            
            # Connect to MongoDB
            client = MongoClient(
                MONGO_DB_URL,
                username=username,
                password=password,
                authSource='admin'  # Assuming admin is the auth database
            )
        except MissingEnvVarError as e:
            logger.critical(f"LLM User password not found for company {self.company_id}")
            return None, None
        except Exception as e:
            logger.error(f"Error connecting LLM User to mongo db for company {self.company_id} with user {username}")
            return None, None
        
        company_db = client[self.company_id]
            
        try:
            # Modify the command to use the view instead of base collection
            modified_cmd = self._rewrite_command(json_cmd, user_access_roles)
            
            # Execute the command
            return modified_cmd, company_db.command(modified_cmd)
        except Exception as e:
            logger.warning(f"Error executing MongoDB command:\nmongo command: {json.dumps(json_cmd)}\n{e}")
            return None, None


    def _rewrite_command(self, original_cmd, user_access_roles):
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
                if op == 'aggregate' and 'pipeline' in cmd:
                    # Inject access control filter at the top level
                    cmd['pipeline'] = self._inject_access_filter_pipeline(cmd['pipeline'], user_access_roles)

                    # Inject access control inside $lookup stages
                    self._rewrite_pipeline_collections(cmd['pipeline'], user_access_roles)

                elif op == 'find':
                    cmd['filter'] = self._inject_access_filter_find(cmd.get('filter', {}), user_access_roles)

                return cmd

        raise ValueError(f"Unsupported operation. Allowed: {READ_OPS}")


    def _inject_access_filter_pipeline(self, pipeline, user_access_roles):
        # Inject $match filter based on access roles at the beginning
        access_filter = {"accessGroup": {"$in": user_access_roles}}
        return [{"$match": access_filter}] + pipeline


    def _inject_access_filter_find(self, existing_filter, user_access_roles):
        access_filter = {"accessGroup": {"$in": user_access_roles}}
        if existing_filter:
            return {"$and": [existing_filter, access_filter]}
        return access_filter


    
    def _rewrite_pipeline_collections(self, pipeline, user_access_roles):
        """Inject accessGroup filter into $lookup pipeline stages"""
        if not isinstance(pipeline, list):
            return
    
        for stage in pipeline:
            if not isinstance(stage, dict):
                continue
    
            if '$lookup' in stage:
                lookup = stage['$lookup']
    
                # Only rewrite if 'from' is a string and no 'pipeline' exists
                if isinstance(lookup.get('from'), str) and 'pipeline' not in lookup:
                    local_field = lookup.get('localField')
                    foreign_field = lookup.get('foreignField')
                    if local_field and foreign_field:
                        stage['$lookup'] = {
                            'from': lookup['from'],
                            'let': {f'lf': f'${local_field}'},
                            'pipeline': [
                                {
                                    '$match': {
                                        '$expr': {'$eq': [f'${foreign_field}', '$$lf']},
                                        'accessGroup': {'$in': user_access_roles}
                                    }
                                }
                            ],
                            'as': lookup['as']
                        }
    
                # If already using a pipeline, we can inject access control directly
                elif 'pipeline' in lookup:
                    lookup['pipeline'].insert(0, {
                        '$match': {
                            'accessGroup': {'$in': user_access_roles}
                        }
                    })


    def close(self):
        """Close the MongoDB connection."""
        self.client.close()