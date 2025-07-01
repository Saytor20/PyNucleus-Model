#!/usr/bin/env python3
"""
PyNucleus Dashboard Launcher

Simple script to launch the Flask dashboard.

Usage:
    python scripts/launch_dashboard.py
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Flask dashboard."""
    
    print("🚀 PyNucleus Dashboard")
    print("=" * 30)
    print("Starting Flask dashboard...")
    print("Dashboard: http://localhost:5001/dashboard")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        # Launch Flask app
        cmd = [
            sys.executable, "-m", "flask", 
            "--app", "src/pynucleus/api/app:create_app",
            "run", 
            "--host=0.0.0.0", 
            "--port=5001", 
            "--debug"
        ]
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 