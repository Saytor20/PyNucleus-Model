#!/bin/bash
# Activate the virtual environment and run the given Python command

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../pynucleus_env"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
  echo "❌ Virtual environment not found at $VENV_DIR"
  echo "Please create it with: python3 -m venv pynucleus_env"
  exit 1
fi

# Activate the venv
source "$VENV_DIR/bin/activate"

# Run the Python command with all arguments
python "$@" 