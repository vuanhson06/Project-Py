#!/bin/bash
echo "🔧 Starting AmongSMS Backend Server..."

# Kill any existing processes on port 8002
echo "Cleaning up port 8002..."
pkill -f "uvicorn.*8002" 2>/dev/null || true

# Wait a moment
sleep 1

# Start the server
echo "🚀 Starting server on port 8002..."
uvicorn Backend:app --host 0.0.0.0 --port 8002

