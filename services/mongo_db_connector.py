import json
import logging
from pymongo import MongoClient
from get_env_var import get_env_var, MissingEnvVarError

logger = logging.getLogger(__name__)

MONGO_DB_URL = get_env_var('MONGO_DB_URL')


class MongoDBConnector:
    
    def __init__(self, company_id):
        """Initialize MongoDB connection with company-specific credentials."""
        self.company_id = company_id


    def run(self, json_cmd, user_id):
        """
        Execute a MongoDB command with proper access restrictions based on user-specific view.
        
        Args:
            json_cmd (dict or str): The MongoDB command in JSON format.
            user_id (str): The ID of the current user.
            
        Returns:
            A tuple (modified_cmd, result) or (None, None) on error.
        """
        if isinstance(json_cmd, str):
            try:
                json_cmd = json.loads(json_cmd)
            except json.JSONDecodeError:
                logger.warning("Failed to decode json_cmd string.")
                return None, None

        try:
            username = f'llm_user_{self.company_id}_{user_id}'
            password = get_env_var(f'LLM_USER_{self.company_id}_PW')

            client = MongoClient(
                MONGO_DB_URL,
                username=username,
                password=password,
                authSource='admin'  # Assuming 'admin' DB is used for authentication
            )
        except MissingEnvVarError:
            logger.critical(f"Missing password env var for user {username}")
            return None, None
        except Exception as e:
            logger.error(f"Error connecting to MongoDB for user {username}: {e}")
            return None, None

        company_db = client[self.company_id]
        view_name = f'access_view_{user_id}'

        if view_name not in company_db.list_collection_names():
            logger.critical(f"No MongoDB view found for user {user_id}. Integrity issue.")
            return None, None

        try:
            modified_cmd = self._rewrite_command(json_cmd, view_name)
            return modified_cmd, company_db.command(modified_cmd)
        except Exception as e:
            logger.warning(f"Error executing MongoDB command:\n{json.dumps(json_cmd)}\nError: {e}")
            return None, None


    def _rewrite_command(self, original_cmd, view_name):
        """
        Rewrites the command to use the per-user view.
        """
        cmd = original_cmd.copy()
        READ_OPS = { 'find', 'aggregate', 'count', 'distinct', 'mapReduce' }

        for op in READ_OPS:
            if op in cmd:
                if isinstance(cmd[op], str):
                    cmd[op] = view_name
                if op == 'aggregate' and 'pipeline' in cmd:
                    self._rewrite_pipeline_collections(cmd['pipeline'], view_name)
                return cmd

        raise ValueError(f"Unsupported operation. Allowed: {READ_OPS}")


    def _rewrite_pipeline_collections(self, pipeline, view_name):
        """Rewrites collection names in aggregation pipelines."""
        if not isinstance(pipeline, list):
            return

        for stage in pipeline:
            if not isinstance(stage, dict):
                continue

            for lookup_op in ('$lookup', '$graphLookup'):
                if lookup_op in stage:
                    lookup = stage[lookup_op]
                    if isinstance(lookup.get('from'), str):
                        lookup['from'] = view_name


    def close(self):
        """Close MongoDB connection (placeholder if connection pooling is later used)."""
        pass  # Kept for symmetry; not actively managing client state