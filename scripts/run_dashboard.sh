#!/bin/bash

# PyNucleus Dashboard Launcher
# Simple shell script to run the unified Streamlit dashboard

set -e

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting PyNucleus Dashboard..."
echo "📁 Project root: $PROJECT_ROOT"
echo "🌐 Opening in browser at http://localhost:8501"
echo "💡 Press Ctrl+C to stop the dashboard"
echo "----------------------------------------"

# Change to project root
cd "$PROJECT_ROOT"

# Run the dashboard
python scripts/run_dashboard.py 