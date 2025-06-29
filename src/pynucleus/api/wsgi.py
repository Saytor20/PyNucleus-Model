"""
WSGI Entry Point for PyNucleus Flask API

Production entry point for gunicorn and other WSGI servers.
Usage: gunicorn --bind 0.0.0.0:5001 src.pynucleus.api.wsgi:app
"""

import os
import sys
from pathlib import Path

# Ensure proper path setup
app_root = Path(__file__).parent.parent.parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

# Import and create app
from src.pynucleus.api.app import create_app

# Production configuration
config = {
    'DEBUG': False,
    'TESTING': False,
    'SECRET_KEY': os.getenv('SECRET_KEY', 'change-this-in-production'),
}

# Create application instance
app = create_app(config)

if __name__ == "__main__":
    # Fallback for direct execution
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5001))) 