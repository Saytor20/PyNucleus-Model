#!/usr/bin/env python3
"""
DWSIM Workflow Module

Provides simplified functions for DWSIM integration in PyNucleus-Model.
Handles cases where DWSIM is not installed or configured.
"""

import os
import sys
from pathlib import Path


def check_dwsim_availability():
    """Check if DWSIM is properly configured."""
    try:
        import clr

        dwsim_path = os.getenv("DWSIM_DLL_PATH")
        if not dwsim_path or not Path(dwsim_path).exists():
            return False, "DWSIM_DLL_PATH not set or directory doesn't exist"
        return True, "DWSIM available"
    except ImportError:
        return (
            False,
            "pythonnet (clr) not available - install with: pip install pythonnet",
        )
    except Exception as e:
        return False, f"DWSIM check failed: {e}"


def run_dwsim_simulation(dwsim_file_path, output_csv_path=None):
    """
    Run a DWSIM simulation and export results to CSV.

    Args:
        dwsim_file_path (str): Path to the .dwsim file
        output_csv_path (str): Path for output CSV file

    Returns:
        str: Path to the output CSV file or None if failed
    """
    available, message = check_dwsim_availability()
    if not available:
        print(f"❌ DWSIM not available: {message}")
        return None

    try:
        # Add DWSIM integration here when DWSIM is properly installed
        from src.sim_bridge.dwsim_bridge import DWSIMBridge

        bridge = DWSIMBridge()
        result = bridge.run_simulation(dwsim_file_path, output_csv_path)
        return result

    except Exception as e:
        print(f"❌ DWSIM simulation failed: {e}")
        return None


def quick_dwsim_demo():
    """
    Run a quick DWSIM demo to test the integration.
    """
    print("🚀 Running DWSIM Quick Demo...")

    # Check if DWSIM is available
    available, message = check_dwsim_availability()
    if not available:
        print(f"❌ Unexpected error: {message}")
        print("\n💡 To use DWSIM integration:")
        print("   1. Install DWSIM on your system")
        print("   2. Set DWSIM_DLL_PATH environment variable")
        print("   3. Place a .dwsim file in examples/ directory")
        print("   4. Run: run_dwsim_simulation('your_file.dwsim')")
        return False

    # Look for example DWSIM files
    examples_dir = Path("examples")
    if not examples_dir.exists():
        examples_dir.mkdir(exist_ok=True)

    dwsim_files = list(examples_dir.glob("*.dwsim"))
    if not dwsim_files:
        print("❌ No .dwsim files found in examples/ directory")
        print("💡 Add a .dwsim file to examples/ to test the integration")
        return False

    # Run simulation on first found file
    dwsim_file = dwsim_files[0]
    print(f"🔧 Running simulation on: {dwsim_file}")

    output_path = run_dwsim_simulation(str(dwsim_file))
    if output_path:
        print(f"✅ Simulation completed! Results saved to: {output_path}")
        return True
    else:
        print("❌ Simulation failed")
        return False


def get_dwsim_status():
    """Get current DWSIM integration status."""
    available, message = check_dwsim_availability()
    status = {
        "available": available,
        "message": message,
        "dwsim_path": os.getenv("DWSIM_DLL_PATH"),
        "examples_dir_exists": Path("examples").exists(),
        "dwsim_files_count": (
            len(list(Path("examples").glob("*.dwsim")))
            if Path("examples").exists()
            else 0
        ),
    }
    return status


if __name__ == "__main__":
    # Quick test
    status = get_dwsim_status()
    print(f"DWSIM Status: {status}")
    quick_dwsim_demo()
