import subprocess
import urllib.parse
import re

# LLM password for all company users
LLM_USER_PASSWORD = "StrongPassword123!"  # or better: load from ENV variable

DANGEROUS_JS_PATTERNS = [
    r"\bwhile\s*\(",
    r"\bfor\s*\(",
    r"\beval\s*\(",
    r"\bfunction\s*\(",
    r"\bsleep\s*\(",
    r"new\s+Function",
    r"load\s*\(",
    r"system\s*\(",
]

class MongoshConnector:

    def __init__(self, company_id):
        """
        Initialize the connector for a specific company.
        Assumes the MongoDB username is 'llm_<company_id>'.
        """
        if not self._validate_company_id(company_id):
            raise ValueError(f"Invalid company_id: {company_id}")

        self.company_id = company_id
        

    def run(self, mongosh_cmd):
        """
        Run a mongosh command for this company's database.

        :param mongosh_cmd: MongoDB shell command
        :return: Output or error message
        """
        try:
            self._validate_command(mongosh_cmd)
            
            username = urllib.parse.quote_plus("llm_read_only")
            password = urllib.parse.quote_plus(LLM_USER_PASSWORD)
            uri = f"mongodb://{username}:{password}@localhost:27017/?authSource={self.company_id}"

            cmd = [
                "mongosh",
                uri,
                "--quiet",
                "--eval",
                f"use {self.company_id}; {mongosh_cmd}"
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


    def _validate_company_id(self, company_id):
        # Only allow alphanumeric + underscores
        return bool(re.fullmatch(r"[a-zA-Z0-9_]+", company_id))


    def _validate_command(self, mongosh_cmd):
        if not isinstance(mongosh_cmd, str):
            raise ValueError("mongosh_cmd must be a string.")

        if len(mongosh_cmd) > 5000:
            raise ValueError("mongosh_cmd is too long.")

        for pattern in DANGEROUS_JS_PATTERNS:
            if re.search(pattern, mongosh_cmd, flags=re.IGNORECASE):
                raise ValueError(f"Potentially dangerous JavaScript detected: '{pattern}'")