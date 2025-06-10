#!/usr/bin/env python3
"""
Test script to verify DWSIM integration
"""

import os
from pathlib import Path


def test_dwsim_environment():
    print("🔍 Testing DWSIM Environment...")

    # Check DWSIM DLL path
    dwsim_path = os.getenv("DWSIM_DLL_PATH")
    if not dwsim_path:
        print("❌ DWSIM_DLL_PATH not set")
        return False

    print(f"✅ DWSIM_DLL_PATH set to: {dwsim_path}")

    # Check DLL directory
    dll_path = Path(dwsim_path)
    if not dll_path.exists():
        print(f"❌ DWSIM DLL directory does not exist: {dwsim_path}")
        return False

    # Count DLLs
    dlls = list(dll_path.glob("*.dll"))
    print(f"✅ Found {len(dlls)} DLL files")
    if not dlls:
        print("❌ No DLL files found")
        return False

    # Try to import DWSIM
    try:
        import clr

        print("✅ pythonnet (clr) is available")

        # Try to load a DWSIM DLL
        try:
            clr.AddReference(str(dlls[0]))
            print(f"✅ Successfully loaded DLL: {dlls[0].name}")
        except Exception as e:
            print(f"❌ Failed to load DLL: {e}")
            return False

    except ImportError as e:
        print(f"❌ pythonnet (clr) is not available: {e}")
        return False

    print("\n✨ DWSIM environment check complete!")
    return True


if __name__ == "__main__":
    test_dwsim_environment()
