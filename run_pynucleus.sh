#!/bin/bash
# PyNucleus CLI Wrapper Script
# Automatically activates virtual environment and runs the CLI

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d "pynucleus_env" ]; then
    echo "❌ Virtual environment 'pynucleus_env' not found!"
    echo "Please run: python -m venv pynucleus_env"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source pynucleus_env/bin/activate

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import rich, typer, chromadb, torch, transformers" 2>/dev/null || {
    echo "❌ Missing dependencies detected!"
    echo "Installing required packages..."
    pip install rich typer chromadb torch transformers pandas rank-bm25 sentence-transformers pydantic-settings rapidfuzz accelerate
}

# Run the CLI
echo "🚀 Starting PyNucleus CLI..."
exec python pynucleus "$@" 