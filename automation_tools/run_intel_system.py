#!/usr/bin/env python3
"""
Entrypoint script for PyNucleus Model
"""

import os
import sys
from pathlib import Path


def main():
    print("🚀 Starting PyNucleus Model...")

    # Check DWSIM environment
    dwsim_path = os.getenv("DWSIM_DLL_PATH")
    if dwsim_path:
        print(f"✅ DWSIM DLL path found: {dwsim_path}")
        dll_path = Path(dwsim_path)
        if dll_path.exists():
            print(
                f"✅ DWSIM DLL directory exists with {len(list(dll_path.glob('*.dll')))} DLLs"
            )
        else:
            print("❌ DWSIM DLL directory does not exist")
    else:
        print("❌ DWSIM_DLL_PATH not set")

    # Check Python environment
    print("\n📦 Python Environment:")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    # Check if pythonnet is available
    try:
        import clr

        print("✅ pythonnet (clr) is available")
    except ImportError:
        print("❌ pythonnet (clr) is not available")

    print("\n✨ System check complete!")


if __name__ == "__main__":
    main()
