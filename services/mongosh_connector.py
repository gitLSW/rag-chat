import os
import re
from dotenv import load_dotenv
import subprocess
import urllib.parse
from typing import List, Dict
from pymongo import MongoClient

load_dotenv()

DANGEROUS_PATTERNS = [
    'eval', 'function', 'system', 'load', 'sleep',
    'while', 'for', 'insert', 'update', 'delete',
    'remove', 'drop', 'shutdown', 'adminCommand'
]


class MongoshConnector:
    def __init__(self, company_id: str, access_level: int):
        """
        Initialize the connector for a specific company and access level.
        """
        if not 0 <= access_level <= 10:
            raise ValueError("Access level must be between 0 and 10")

        self.company_id = company_id
        self.access_level = access_level
        self.username = f"llm_{company_id}_level{access_level}"
        self.password = os.getenv(f"LLM_{self.company_id}_PW")
         
    
    def run(self, mongosh_cmd: str) -> str:
        """
        Run a mongosh command with the appropriate access level restrictions.
        
        :param mongosh_cmd: MongoDB shell command
        :return: Output or error message
        """
        try:
            self._validate_command(mongosh_cmd)
            
            username = urllib.parse.quote_plus(self.username)
            password = urllib.parse.quote_plus(self.password)
            uri = f"mongodb://{username}:{password}@localhost:27017/{self.company_id}?authSource=admin"

            cmd = [
                "mongosh",
                uri,
                "--quiet",
                "--eval",
                mongosh_cmd
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            if process.returncode != 0:
                raise RuntimeError(f"mongosh error: {process.stderr.strip()}")

            return process.stdout.strip()

        except Exception as e:
            return f"Error executing mongosh command: {str(e)}"


    def _validate_command(self, mongosh_cmd: str) -> None:
        """Validate the command for security."""
        if not isinstance(mongosh_cmd, str):
            raise ValueError("mongosh_cmd must be a string.")
        if len(mongosh_cmd) > 5000:
            raise ValueError("mongosh_cmd is too long.")
        
        # Prevent dangerous operations
        for cmd in DANGEROUS_PATTERNS:
            if re.search(rf'\b{cmd}\b', mongosh_cmd, re.IGNORECASE):
                raise ValueError(f"Potentially dangerous command detected: '{cmd}'")