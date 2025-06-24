#!/usr/bin/env python3
"""
Smart Auto-Restart Script for PyNucleus Web Server
==================================================
This script automatically:
1. Kills any existing server processes on port 5001
2. Starts the server fresh with enhanced statistics
3. Handles cleanup on shutdown
4. Can optionally watch for file changes and auto-restart

Usage:
    python restart_web_server.py           # Start server with auto-restart
    python restart_web_server.py --watch   # Start with file watching
    python restart_web_server.py --kill    # Just kill existing servers
"""

import os
import sys
import time
import signal
import subprocess
import atexit
from pathlib import Path

def kill_existing_servers():
    """Kill any existing PyNucleus web servers"""
    print("🔄 Killing existing PyNucleus web servers...")
    
    # Kill processes using port 5001
    try:
        result = subprocess.run(['lsof', '-ti:5001'], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    subprocess.run(['kill', '-9', pid], capture_output=True)
                    print(f"   ✅ Killed process {pid}")
    except Exception as e:
        print(f"   ⚠️ Error killing port processes: {e}")
    
    # Kill PyNucleus web app processes
    try:
        subprocess.run(['pkill', '-f', 'run_web_app'], capture_output=True)
        print("   ✅ Killed PyNucleus web app processes")
    except Exception as e:
        print(f"   ⚠️ Error killing web app processes: {e}")
    
    # Small delay to ensure cleanup
    time.sleep(1)

def start_server():
    """Start the PyNucleus Flask web server using the correct entry point"""
    print("🚀 Starting PyNucleus web server...")
    
    # Change to the correct directory
    os.chdir(Path(__file__).parent)

    # Set environment variables for Flask
    env = os.environ.copy()
    env["FLASK_APP"] = "src/pynucleus/api/app.py"
    env["FLASK_ENV"] = "development"
    env["PYTHONUNBUFFERED"] = "1"

    # Start the Flask server using gunicorn for better production compatibility
    process = subprocess.Popen(
        ["gunicorn", "--reload", "--bind", "0.0.0.0:5001", "--workers", "1", "wsgi:app"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=env,
        cwd=Path(__file__).parent / "src" / "pynucleus" / "api"
    )
    return process

def cleanup():
    """Cleanup function called on exit"""
    print("\n🛑 Shutting down PyNucleus web server...")
    kill_existing_servers()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PyNucleus Web Server Auto-Restart')
    parser.add_argument('--kill', action='store_true', help='Just kill existing servers and exit')
    parser.add_argument('--watch', action='store_true', help='Watch for file changes and auto-restart')
    parser.add_argument('--no-kill', action='store_true', help='Don\'t kill existing servers first')
    
    args = parser.parse_args()
    
    if args.kill:
        kill_existing_servers()
        print("✅ Existing servers killed. Exiting.")
        return
    
    # Register cleanup function
    atexit.register(cleanup)
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal...")
        cleanup()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Kill existing servers unless told not to
        if not args.no_kill:
            kill_existing_servers()
        
        # Start the server
        process = start_server()
        
        print("=" * 60)
        print("🎯 PyNucleus Web Server Auto-Restart Active!")
        print("📱 Visit: http://localhost:5001/")
        print("🔧 Enhanced Statistics Buttons Available!")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 60)
        
        if args.watch:
            print("👀 File watching enabled - server will auto-restart on changes")
            # TODO: Add file watching functionality if needed
        
        # Stream server output
        try:
            for line in process.stdout:
                print(line, end='')
        except KeyboardInterrupt:
            pass
        
    except Exception as e:
        print(f"❌ Error: {e}")
        cleanup()
        sys.exit(1)

if __name__ == "__main__":
    main() 