#!/usr/bin/env python3
"""
Chemical Engineering Terminal Dashboard Launcher

Simple launcher for the redesigned terminal dashboard with black background.
"""

import sys
import subprocess
from pathlib import Path

def main():
    """Launch the Chemical Engineering Terminal Dashboard."""
    
    # Get the project root directory
    root_dir = Path(__file__).parent.parent
    dashboard_path = root_dir / "src" / "pynucleus" / "terminal_dashboard.py"
    
    if not dashboard_path.exists():
        print("Error: Dashboard file not found!")
        print(f"Expected location: {dashboard_path}")
        sys.exit(1)
    
    print("Chemical Engineering Terminal Dashboard")
    print("=" * 50)
    print("Starting dashboard...")
    print()
    
    try:
        # Run the dashboard
        subprocess.run([sys.executable, str(dashboard_path)], cwd=root_dir)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except Exception as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 