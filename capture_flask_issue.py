#!/usr/bin/env python3
"""
Comprehensive Flask UI Issue Capture Script
===========================================
Captures complete system state for Flask UI issue reproduction and analysis.
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def capture_system_info():
    """Capture comprehensive system information"""
    info = {
        "timestamp": datetime.now().isoformat(),
        "working_directory": os.getcwd(),
        "python_version": sys.version,
        "platform": sys.platform,
        "environment_variables": dict(os.environ),
        "current_user": os.getenv("USER", "unknown"),
        "hostname": os.uname().nodename if hasattr(os, 'uname') else "unknown"
    }
    return info

def capture_pip_freeze():
    """Capture pip freeze output"""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "freeze"], 
                              capture_output=True, text=True, timeout=30)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

def capture_gunicorn_startup():
    """Capture gunicorn startup with app:create_app()"""
    print("🚀 Starting gunicorn with app:create_app()...")
    
    # Store original directory
    original_dir = os.getcwd()
    
    # First, let's check if we need to create a create_app function
    app_file = Path("src/pynucleus/api/app.py")
    if not app_file.exists():
        print("❌ app.py not found")
        return {"success": False, "error": "app.py not found"}
    
    with open(app_file, 'r') as f:
        content = f.read()
        
    if "def create_app()" not in content:
        print("⚠️  No create_app() function found, creating one temporarily...")
        
        # Add create_app function to the end of the file
        create_app_code = """

def create_app():
    \"\"\"Application factory function for gunicorn.\"\"\"
    return app
"""
        
        # Create temporary app file with create_app function
        temp_app_file = Path("src/pynucleus/api/app_temp.py")
        with open(temp_app_file, 'w') as f:
            f.write(content + create_app_code)
        
        gunicorn_app = "app_temp:create_app()"
        app_module = "app_temp"
    else:
        gunicorn_app = "app:create_app()"
        app_module = "app"
    
    # Change to the API directory
    api_dir = Path("src/pynucleus/api")
    if not api_dir.exists():
        return {"success": False, "error": "API directory not found"}
    
    os.chdir(api_dir)
    
    # Set environment variables
    env = os.environ.copy()
    env["FLASK_APP"] = f"{app_module}.py"
    env["FLASK_ENV"] = "development"
    env["PYTHONUNBUFFERED"] = "1"
    
    # Check if gunicorn is available
    try:
        subprocess.run(["gunicorn", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        os.chdir(original_dir)
        return {"success": False, "error": "gunicorn not installed"}
    
    # Start gunicorn process
    try:
        print(f"🚀 Starting: gunicorn {gunicorn_app} --bind 0.0.0.0:5001")
        process = subprocess.Popen(
            ["gunicorn", gunicorn_app, "--bind", "0.0.0.0:5001", "--workers", "1", "--timeout", "30", "--preload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env,
            cwd=api_dir
        )
        
        # Capture output for 15 seconds to give more time
        stdout_lines = []
        stderr_lines = []
        start_time = time.time()
        
        while time.time() - start_time < 15:
            if process.poll() is not None:
                # Process has ended
                break
            
            # Read stdout with timeout
            import select
            if hasattr(select, 'select'):
                ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)
                if process.stdout in ready:
                    stdout_line = process.stdout.readline()
                    if stdout_line:
                        stdout_lines.append(stdout_line.strip())
                        print(f"STDOUT: {stdout_line.strip()}")
                
                if process.stderr in ready:
                    stderr_line = process.stderr.readline()
                    if stderr_line:
                        stderr_lines.append(stderr_line.strip())
                        print(f"STDERR: {stderr_line.strip()}")
            else:
                # Fallback for systems without select
                stdout_line = process.stdout.readline()
                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    print(f"STDOUT: {stdout_line.strip()}")
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    print(f"STDERR: {stderr_line.strip()}")
        
        # Get remaining output
        try:
            remaining_stdout, remaining_stderr = process.communicate(timeout=2)
            if remaining_stdout:
                stdout_lines.extend(remaining_stdout.strip().split('\n'))
            if remaining_stderr:
                stderr_lines.extend(remaining_stderr.strip().split('\n'))
        except subprocess.TimeoutExpired:
            pass
        
        # Terminate process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        
        # Return to original directory
        os.chdir(original_dir)
        
        # Clean up temp file if created
        if gunicorn_app.startswith("app_temp:"):
            temp_app_file.unlink(missing_ok=True)
        
        return {
            "success": process.returncode == 0 if process.returncode is not None else False,
            "stdout": "\n".join(stdout_lines),
            "stderr": "\n".join(stderr_lines),
            "return_code": process.returncode,
            "gunicorn_app": gunicorn_app,
            "process_ended": process.poll() is not None
        }
        
    except Exception as e:
        # Return to original directory on error
        os.chdir(original_dir)
        
        # Clean up temp file if created
        if 'temp_app_file' in locals():
            temp_app_file.unlink(missing_ok=True)
        
        return {"success": False, "error": str(e)}

def run_diagnostic_script(script_name):
    """Run a diagnostic script and capture output"""
    print(f"🔍 Running {script_name}...")
    
    script_path = Path("scripts") / script_name
    if not script_path.exists():
        return {"success": False, "error": f"Script {script_name} not found"}
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=Path.cwd()
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Script timed out after 120 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Main capture function"""
    print("🔍 Starting comprehensive Flask UI issue capture...")
    
    # Store the original working directory and use absolute paths
    original_dir = Path.cwd()
    
    # Ensure logs directory exists with absolute path
    logs_dir = original_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Capture timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize capture data
    capture_data = {
        "capture_timestamp": datetime.now().isoformat(),
        "system_info": capture_system_info(),
        "pip_freeze": capture_pip_freeze(),
        "gunicorn_startup": capture_gunicorn_startup(),
        "diagnostics": {}
    }
    
    # Run diagnostic scripts
    diagnostic_scripts = [
        "comprehensive_system_diagnostic.py",
        "system_validator.py"
    ]
    
    for script in diagnostic_scripts:
        print(f"🔍 Running {script}...")
        capture_data["diagnostics"][script] = run_diagnostic_script(script)
    
    # Write failure reproduction log with absolute path
    failure_log_path = logs_dir / f"failure_reproduction_{timestamp}.log"
    with open(failure_log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("FLASK UI ISSUE REPRODUCTION CAPTURE\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SYSTEM INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(json.dumps(capture_data["system_info"], indent=2))
        f.write("\n\n")
        
        f.write("PIP FREEZE OUTPUT:\n")
        f.write("-" * 40 + "\n")
        pip_data = capture_data["pip_freeze"]
        f.write(f"Success: {pip_data['success']}\n")
        if pip_data['success']:
            f.write(pip_data['stdout'])
        else:
            f.write(f"Error: {pip_data.get('error', 'Unknown error')}\n")
        f.write("\n\n")
        
        f.write("GUNICORN STARTUP ATTEMPT:\n")
        f.write("-" * 40 + "\n")
        gunicorn_data = capture_data["gunicorn_startup"]
        f.write(f"Success: {gunicorn_data['success']}\n")
        f.write(f"Gunicorn App: {gunicorn_data.get('gunicorn_app', 'N/A')}\n")
        f.write(f"Return Code: {gunicorn_data.get('return_code', 'N/A')}\n\n")
        
        f.write("STDOUT:\n")
        f.write(gunicorn_data.get('stdout', 'No output'))
        f.write("\n\nSTDERR:\n")
        f.write(gunicorn_data.get('stderr', 'No output'))
        
        # Add error details if available
        if 'error' in gunicorn_data:
            f.write(f"\n\nERROR: {gunicorn_data['error']}")
        f.write("\n\n")
    
    # Write diagnostics log with absolute path
    diagnostics_log_path = logs_dir / f"diagnostics_output_{timestamp}.log"
    with open(diagnostics_log_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DIAGNOSTIC SCRIPTS OUTPUT\n")
        f.write("=" * 80 + "\n\n")
        
        for script_name, result in capture_data["diagnostics"].items():
            f.write(f"SCRIPT: {script_name}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Success: {result['success']}\n")
            f.write(f"Return Code: {result.get('return_code', 'N/A')}\n\n")
            
            if result['success']:
                f.write("STDOUT:\n")
                f.write(result.get('stdout', 'No output'))
                f.write("\n\nSTDERR:\n")
                f.write(result.get('stderr', 'No output'))
            else:
                f.write(f"Error: {result.get('error', 'Unknown error')}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    
    # Create symlink for easy access with absolute paths
    try:
        symlink_failure = logs_dir / "failure_reproduction.log"
        symlink_diagnostics = logs_dir / "diagnostics_output.log"
        
        if symlink_failure.exists():
            symlink_failure.unlink()
        if symlink_diagnostics.exists():
            symlink_diagnostics.unlink()
        
        symlink_failure.symlink_to(failure_log_path.name)
        symlink_diagnostics.symlink_to(diagnostics_log_path.name)
    except Exception as e:
        print(f"⚠️  Could not create symlinks: {e}")
    
    print(f"✅ Capture complete!")
    print(f"📄 Failure reproduction log: {failure_log_path}")
    print(f"📄 Diagnostics output log: {diagnostics_log_path}")
    
    return capture_data

if __name__ == "__main__":
    main() 