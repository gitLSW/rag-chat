#!/bin/bash

# This script prepares and registers a systemd service for your RAG FastAPI server.
# It should be placed in your project_root_dir/install/ directory.

# Assumptions:
# 1. You are running this script from your project_root_dir/install/ directory.
# 2. Your virtual environment 'rag-env' is located at project_root_dir/rag-env.
# 3. Your FastAPI server script is at project_root_dir/server/api_server.py.
# 4. MongoDB is managed as a separate systemd service (e.g., 'mongod.service').

# --- Configuration ---
# Name of your systemd service
SERVICE_NAME="rag-server"
# Full path where the service file will be created
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

# Determine project root directory:
# This script is located at project_root_dir/install/register_systemd.sh
# So, going up one level from 'install' gives us the project root.
PROJECT_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Full path to the Python executable within your virtual environment
VENV_PYTHON="${PROJECT_ROOT_DIR}/rag-env/bin/python"
# Full path to your FastAPI server script
SERVER_SCRIPT="${PROJECT_ROOT_DIR}/server/api_server.py"

# Directory for service logs (if you choose file logging over journalctl)
# Note: Using journalctl (StandardOutput=journal) is generally preferred for systemd services.
LOG_DIR="${PROJECT_ROOT_DIR}/logs"

# Get the current user's username. The service will run under this user.
# This is important for file permissions and access to your project's directories.
RUN_AS_USER=$(whoami)

# --- Pre-checks and Validations ---
echo "--- Validating paths and environment ---"

# Ensure the log directory exists (if you decide to use file logging)
mkdir -p "${LOG_DIR}"

# Check if the virtual environment directory exists
if [ ! -d "${PROJECT_ROOT_DIR}/rag-env" ]; then
    echo "Error: Virtual environment 'rag-env' not found at ${PROJECT_ROOT_DIR}/rag-env."
    echo "Please ensure it exists and is correctly set up, or update PROJECT_ROOT_DIR in the script."
    exit 1
fi

# Check if the main server script exists
if [ ! -f "${SERVER_SCRIPT}" ]; then
    echo "Error: Server script 'api_server.py' not found at ${SERVER_SCRIPT}."
    echo "Please ensure it exists or update SERVER_SCRIPT in the script."
    exit 1
fi

# Check if the Python executable within the virtual environment exists
if [ ! -f "${VENV_PYTHON}" ]; then
    echo "Error: Python executable not found in virtual environment at ${VENV_PYTHON}."
    echo "Please ensure the virtual environment is correctly set up (e.g., by running 'python3 -m venv rag-env' and 'source rag-env/bin/activate')."
    exit 1
fi

echo "All paths validated successfully."

# --- Create Systemd Service File ---
echo "--- Creating systemd service file: ${SERVICE_FILE} ---"

# Use 'sudo bash -c' to write the file as root, as /etc/systemd/system requires root privileges.
# The 'EOF' block allows for multi-line string creation.
sudo bash -c "cat > ${SERVICE_FILE} <<EOF
[Unit]
Description=RAG FastAPI Server
# Ensure network is up before starting this service.
# This service is configured to start after MongoDB, which is managed as 'mongod.service'
# as per your 'Install MongoDB' and 'Start and enable mongoDB' steps in install.md.
After=network.target mongod.service

[Service]
# User under which the service will run. Important for permissions.
User=${RUN_AS_USER}
# Working directory for the service. All relative paths in your Python script will be relative to this.
WorkingDirectory=${PROJECT_ROOT_DIR}
# The command to execute when the service starts.
# It uses the full path to the Python interpreter inside your virtual environment
# and then the full path to your FastAPI server script.
ExecStart=${VENV_PYTHON} ${SERVER_SCRIPT}
# Automatically restart the service if it crashes or stops unexpectedly.
Restart=always
# Direct standard output and error to the systemd journal for centralized logging.
StandardOutput=journal
StandardError=journal
# If you prefer logs to be written to files instead of journalctl, uncomment the lines below
# and comment out the 'StandardOutput=journal' and 'StandardError=journal' lines:
# StandardOutput=append:${LOG_DIR}/stdout.log
# StandardError=append:${LOG_DIR}/stderr.log

[Install]
# This ensures the service starts automatically when the system boots up.
WantedBy=multi-user.target
EOF"

# Check if the service file creation was successful
if [ $? -ne 0 ]; then
    echo "Error: Failed to create systemd service file. Do you have sudo permissions?"
    exit 1
fi

echo "Systemd service file created successfully."

# --- Reload Systemd, Enable, and Start Service ---
echo "--- Reloading systemd daemon to recognize the new service ---"
sudo systemctl daemon-reload

echo "--- Service Setup Complete ---"
echo ""
echo "Your RAG FastAPI server should now be running as a systemd service."
echo "It will automatically restart if it crashes and will start on system boot."
echo ""
echo "--- Useful Commands ---"
echo "To check the service status:"
echo "  sudo systemctl status ${SERVICE_NAME}"
echo ""
echo "To view logs (most recent first):"
echo "  journalctl -u ${SERVICE_NAME} --since '5 minutes ago' -f"
echo "  (Use '-f' to follow logs in real-time)"
echo ""
echo "To start the service:"
echo "  sudo systemctl start ${SERVICE_NAME}"
echo ""
echo "To stop the service:"
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo ""
echo "To restart the service:"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo ""
echo "To disable the service from starting on boot:"
echo "  sudo systemctl disable ${SERVICE_NAME}"
echo ""
echo "--- Important Note on MongoDB ---"
echo "This setup assumes MongoDB is running as a separate systemd service (e.g., 'mongod.service')."
echo "Ensure you have completed the 'Install MongoDB' and 'Start and enable mongoDB' steps in your 'install.md' before running this script."