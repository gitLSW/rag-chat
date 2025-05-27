#!/bin/bash
MONGO_PORT=27017
DB_PATH="./data/databases/mongo_db"

mkdir -p $DB_PATH

echo "[INFO] Starting MongoDB on port $MONGO_PORT..."
mongod --port $MONGO_PORT --dbpath $DB_PATH --bind_ip 127.0.0.1 &
MONGO_PID=$!

echo "[INFO] Starting FastAPI server..."
python server/api_server.py

# Shutdown MongoDB when done
kill $MONGO_PID