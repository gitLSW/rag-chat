import os
import json
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
    Supports int, float, bool, JSON arrays/objects.
    """
    value = os.getenv(env_var_name, default)

    if value is None:
        raise MissingEnvVarError(env_var_name)
    
    # Try to parse int
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    
    # Try to parse float
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    
    # Try to parse bool
    normalized_value = str(value).lower()
    if normalized_value == 'true':
        return True
    elif normalized_value == 'false':
        return False

    # Try to parse JSON (arrays or objects)
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (list, dict)):
            return parsed
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: return string as is
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

