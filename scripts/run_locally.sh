#!/bin/bash
"""
PyNucleus Local Development Server Launch Script
===============================================
Launches the Flask app using Gunicorn with development-friendly settings.
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 PyNucleus Local Development Server${NC}"
echo -e "${BLUE}====================================${NC}"

# Check if we're in the right directory
if [ ! -f "src/pynucleus/api/wsgi.py" ]; then
    echo -e "${RED}❌ Error: Run this script from the PyNucleus project root directory${NC}"
    echo -e "   Expected to find: src/pynucleus/api/wsgi.py"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠️  Warning: No virtual environment detected${NC}"
    echo -e "   Consider activating your virtual environment first:"
    echo -e "   ${BLUE}source .venv/bin/activate${NC}"
    echo ""
fi

# Check if gunicorn is installed
if ! command -v gunicorn &> /dev/null; then
    echo -e "${RED}❌ Error: gunicorn not found${NC}"
    echo -e "   Install with: ${BLUE}pip install gunicorn${NC}"
    exit 1
fi

# Set environment variables for development
export FLASK_ENV=development
export PYTHONUNBUFFERED=1

# Change to API directory
cd src/pynucleus/api || exit 1

echo -e "${GREEN}✅ Starting PyNucleus Flask API with Gunicorn...${NC}"
echo -e "   📍 URL: ${BLUE}http://localhost:5001${NC}"
echo -e "   🔄 Auto-reload: ${GREEN}enabled${NC}"
echo -e "   📋 Workers: ${YELLOW}1${NC}"
echo -e "   🛑 Press Ctrl+C to stop"
echo ""

# Launch gunicorn with development settings
exec gunicorn \
    --reload \
    --bind 0.0.0.0:5001 \
    --workers 1 \
    --timeout 60 \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --capture-output \
    "wsgi:app" 