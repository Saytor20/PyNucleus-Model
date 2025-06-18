#!/usr/bin/env python3
"""
PyNucleus CLI entry point module.

This module provides the main entry point for the PyNucleus CLI application.
"""

import sys
from pathlib import Path

# Add project root to Python path to access run_pipeline.py
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the typer app from run_pipeline.py
from run_pipeline import app


def main():
    """Main entry point for the PyNucleus CLI."""
    app()


if __name__ == "__main__":
    main() 