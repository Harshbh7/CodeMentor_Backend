#!/bin/sh
set -e

# Set python path to current directory so that 'app' module can be found
export PYTHONPATH=.

# Run database migrations
alembic upgrade head

# Start the FastAPI application on the port specified by Render, falling back to 8000
PORT_TO_USE=${PORT:-8000}
uvicorn app.main:app --host 0.0.0.0 --port "$PORT_TO_USE" --workers 1
