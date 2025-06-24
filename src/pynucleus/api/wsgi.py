"""
WSGI entry point for PyNucleus Flask API.

This module provides the application factory for production deployment with Gunicorn.
"""

import os
import sys
from pathlib import Path

# Add project paths to Python path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
SRC_PATH = BASE_DIR.parent.parent

for path in [str(PROJECT_ROOT), str(SRC_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from app import create_app

# Create application instance for WSGI
app = create_app()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False) 