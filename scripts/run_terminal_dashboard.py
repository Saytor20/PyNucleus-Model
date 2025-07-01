#!/usr/bin/env python3
"""
PyNucleus Terminal Dashboard Launcher

Simple script to launch the Flask-based terminal dashboard.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Run the PyNucleus terminal dashboard"""
    
    # Get the project root directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Add src to Python path
    src_path = str(project_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    # Set environment variables
    os.environ['PYTHONPATH'] = f"{src_path}:{os.environ.get('PYTHONPATH', '')}"
    
    # Dashboard file path
    dashboard_path = project_root / "src" / "pynucleus" / "terminal_dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Terminal dashboard file not found: {dashboard_path}")
        sys.exit(1)
    
    print("🚀 Starting PyNucleus Terminal Dashboard...")
    print(f"📁 Dashboard path: {dashboard_path}")
    print(f"🌐 Will try ports: 5000, 5001, 8080, 8000, 3000")
    print("💡 Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run the Flask app
        subprocess.run([
            sys.executable, str(dashboard_path)
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Terminal dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running terminal dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 