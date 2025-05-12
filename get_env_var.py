import os
from dotenv import load_dotenv

# Load environment variables once at startup
load_dotenv()


# Custom error class
class MissingEnvVarError(Exception):
    """Raised when a required environment variable is missing"""
    def __init__(self, var_name):
        super().__init__(f"Missing required environment variable: {var_name}")
        self.var_name = var_name


def get_env_var(env_var_name: str, default: any = None) -> str:
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
    
    normalized_value = value.lower()
    boolean_descriptions = ['true', 'false']
    if normalized_value in boolean_descriptions:
        return normalized_value == 'true'

    return value