#!/bin/bash

# PyNucleus Flask Server Runner
# Sets environment variables and launches the Flask API server

set -e  # Exit on any error

# Set Flask app location
export FLASK_APP=src.pynucleus.api.app:app

# Set Flask environment
export FLASK_ENV=development

# Launch Flask server on all interfaces, port 5000
echo "Starting PyNucleus Flask API server..."
echo "Server will be available at: http://localhost:5000"
echo "Health check: http://localhost:5000/health"
echo ""

flask run --host=0.0.0.0 --port=5000 