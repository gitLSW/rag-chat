import subprocess
import urllib.parse
import re

# LLM user credentials
LLM_USER_CREDENTIALS = {
    "username": "llm_readonly",
    "password": "StrongPassword123!"
}

DANGEROUS_JS_PATTERNS = [
    r"\bwhile\s*\(",    # while loops
    r"\bfor\s*\(",      # for loops
    r"\beval\s*\(",     # eval()
    r"\bfunction\s*\(", # function definitions
    r"\bsleep\s*\(",    # sleep functions (if allowed)
    r"new\s+Function",  # new Function()
    r"load\s*\(",       # loading external scripts
    r"system\s*\(",     # try to access system shell
]

class MongoshConnector:
    
    def __init__(self, host, port, database, auth_source="admin"):
        self.host = host
        self.port = port
        self.database = database
        self.auth_source = auth_source
        self.uri = self._build_uri()

    def _build_uri(self):
        username = urllib.parse.quote_plus(LLM_USER_CREDENTIALS["username"])
        password = urllib.parse.quote_plus(LLM_USER_CREDENTIALS["password"])
        return f"mongodb://{username}:{password}@{self.host}:{self.port}/?authSource={self.auth_source}"

    def _validate_command(self, mongosh_cmd):
        if not isinstance(mongosh_cmd, str):
            raise ValueError("mongosh_cmd must be a string.")

        if len(mongosh_cmd) > 5000:
            raise ValueError("mongosh_cmd is too long.")

        for pattern in DANGEROUS_JS_PATTERNS:
            if re.search(pattern, mongosh_cmd, flags=re.IGNORECASE):
                raise ValueError(f"Potentially dangerous JavaScript detected: '{pattern}'")

    def run(self, mongosh_cmd):
        try:
            self._validate_command(mongosh_cmd)

            cmd = [
                "mongosh",
                self.uri,
                "--quiet",
                "--eval",
                f"use {self.database}; {mongosh_cmd}"
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