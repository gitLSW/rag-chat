#!/bin/bash

# This script is used to start your RAG FastAPI server and MongoDB
# as systemd services. It assumes that both services have been
# registered and enabled via systemd (e.g., using install.md and register_systemd.sh).

# --- Configuration ---
# Name of your FastAPI server systemd service
FASTAPI_SERVICE_NAME="rag-server"
# Name of the MongoDB systemd service (standard name)
MONGO_SERVICE_NAME="mongod"

echo "--- Starting Services ---"

# 1. Start MongoDB service
echo "[INFO] Starting MongoDB service (${MONGO_SERVICE_NAME})..."
sudo systemctl start "${MONGO_SERVICE_NAME}"
if [ $? -eq 0 ]; then
    echo "[INFO] MongoDB service started successfully."
else
    echo "[ERROR] Failed to start MongoDB service. Please check its status with 'sudo systemctl status ${MONGO_SERVICE_NAME}'."
    # Exit if MongoDB fails to start, as the FastAPI server depends on it.
    exit 1
fi

# 2. Start FastAPI server service
echo "[INFO] Starting RAG FastAPI server service (${FASTAPI_SERVICE_NAME})..."
sudo systemctl start "${FASTAPI_SERVICE_NAME}"
if [ $? -eq 0 ]; then
    echo "[INFO] RAG FastAPI server service started successfully."
else
    echo "[ERROR] Failed to start RAG FastAPI server service. Please check its status with 'sudo systemctl status ${FASTAPI_SERVICE_NAME}'."
    exit 1
fi

echo ""
echo "--- Service Status ---"
echo "You can check the status of your services using the following commands:"
echo "  MongoDB:          sudo systemctl status ${MONGO_SERVICE_NAME}"
echo "  FastAPI Server:   sudo systemctl status ${FASTAPI_SERVICE_NAME}"
echo ""
echo "To view logs for the FastAPI server:"
echo "  journalctl -u ${FASTAPI_SERVICE_NAME} --since '5 minutes ago' -f"
echo ""
echo "To stop the FastAPI server:"
echo "  sudo systemctl stop ${FASTAPI_SERVICE_NAME}"
echo ""
echo "To restart the FastAPI server:"
echo "  sudo systemctl restart ${FASTAPI_SERVICE_NAME}"