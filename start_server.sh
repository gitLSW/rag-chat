#!/bin/bash

MONGO_PORT=27017
DB_PATH="./data/databases/mongo_db"

mkdir -p $DB_PATH

echo "[INFO] Starting MongoDB on port $MONGO_PORT..."
mongod --port $MONGO_PORT --dbpath $DB_PATH --bind_ip 127.0.0.1 &
MONGO_PID=$!

echo "[INFO] Starting FastAPI server..."
uvicorn server.api_server:app --host 0.0.0.0 --port 7500

# Shutdown MongoDB when done
kill $MONGO_PID