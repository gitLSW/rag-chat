import os
from dotenv import load_dotenv

# Get the root directory of the project
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Get current file, go up two

data_path = os.path.join(project_root, 'data')
server_path = os.path.join(project_root, 'server')


# Load environment variables once at startup
load_dotenv(os.path.join(project_root, '.env'))


# Custom error class
class MissingEnvVarError(Exception):
    """Raised when a required environment variable is missing"""
    def __init__(self, var_name):
        super().__init__(f"Missing required environment variable: {var_name}")
        self.var_name = var_name


def get_env_var(env_var_name, default=None):
    """
    Get an environment variable or raise MissingEnvVarError if not found.
    
    Args:
        env_var_name: Name of the environment variable to get
        default: Optional default value if the variable is not required
        
    Returns:
        The value of the environment variable
        
    Raises:
        MissingEnvVarError: If the variable is not set and no default is provided
    """
    value = os.getenv(env_var_name, default)

    if value is None:
        raise MissingEnvVarError(env_var_name)
    
    try:
        return int(value)
    except ValueError:
        pass
    
    try:
        return float(value)
    except ValueError:
        pass
    
    normalized_value = value.lower()
    boolean_descriptions = ['true', 'false']
    if normalized_value in boolean_descriptions:
        return normalized_value == 'true'

    return value


def get_company_path(company_id: str, sub_path: str) -> str:
    """
    Returns the absolute path to a company's subdirectory, safely and normalized.
    
    Args:
        company_id (str): ID of the company.
        sub_path (str): Subpath within the company"s directory.

    Returns:
        str: Absolute normalized path to the requested subdirectory/file.
        
    Raises:
        ValueError: If the resulting path is outside the company"s root directory.
    """
    # Remove leading slashes from sub_path to prevent it from being treated as absolute
    sub_path = sub_path.lstrip("/\\")

    # Build the base path for the company
    company_root = os.path.join(data_path, 'companies', company_id)

    # Join and normalize the full path
    target_path = os.path.normpath(os.path.join(company_root, sub_path))

    # Ensure the path is within the company's directory
    if not os.path.commonpath([company_root, target_path]) == company_root:
        raise ValueError("Invalid path: Attempted to access outside of the company's directory.")

    return target_path


def apply_defaults(instance: dict, schema: dict):
    if not isinstance(instance, dict) or not isinstance(schema, dict):
        return

    properties = schema.get('properties', {})
    for prop, subschema in properties.items():
        if prop not in instance or instance[prop] is None:
            if 'default' in subschema:
                instance[prop] = subschema['default']
        elif subschema.get('type') == 'object' and isinstance(instance.get(prop), dict):
            # Recurse into nested object
            apply_defaults(instance[prop], subschema)
        elif subschema.get('type') == 'array' and isinstance(instance.get(prop), list):
            item_schema = subschema.get('items')
            if item_schema and isinstance(item_schema, dict):
                for item in instance[prop]:
                    if isinstance(item, dict):
                        apply_defaults(item, item_schema)

