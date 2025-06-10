#!/usr/bin/env python3
"""
DWSIM Integration Test Script
Tests the DWSIM bridge functionality and DLL loading
"""

import os
import sys
from pathlib import Path
from typing import List, Optional


def test_dwsim_environment() -> bool:
    """Test the DWSIM environment setup."""
    print("\n🔍 Testing DWSIM Environment...")

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

    # Count and list DLLs
    dlls = list(dll_path.glob("*.dll"))
    print(f"✅ Found {len(dlls)} DLL files:")
    for dll in dlls:
        print(f"   - {dll.name}")

    if not dlls:
        print("❌ No DLL files found")
        return False

    return True


def test_python_environment() -> bool:
    """Test the Python environment setup."""
    print("\n🔍 Testing Python Environment...")

    # Check Python version
    print(f"✅ Python version: {sys.version}")

    # Check pythonnet
    try:
        import clr

        print("✅ pythonnet (clr) is available")

        # Try to load a DWSIM DLL
        try:
            from DWSIM.Interfaces import IFlowsheet

            print("✅ Successfully imported DWSIM.Interfaces")

            from DWSIM.Thermodynamics import PropertyPackages

            print("✅ Successfully imported DWSIM.Thermodynamics")

            from DWSIM.UnitOperations import UnitOperations

            print("✅ Successfully imported DWSIM.UnitOperations")

        except Exception as e:
            print(f"❌ Failed to import DWSIM types: {e}")
            return False

    except ImportError as e:
        print(f"❌ pythonnet (clr) is not available: {e}")
        return False

    return True


def test_dwsim_bridge() -> bool:
    """Test the DWSIM bridge functionality."""
    print("\n🔍 Testing DWSIM Bridge...")

    try:
        from sim_bridge.dwsim_bridge import DWSIMBridge

        with DWSIMBridge() as bridge:
            # Test creating a new flowsheet
            flowsheet = bridge.create_flowsheet()
            print("✅ Successfully created new flowsheet")

            # Test basic flowsheet operations
            print("✅ Successfully initialized DWSIM bridge")

        return True

    except Exception as e:
        print(f"❌ DWSIM bridge test failed: {e}")
        return False


def main() -> None:
    """Run all DWSIM integration tests."""
    print("🚀 Starting DWSIM Integration Tests...")

    # Run all tests
    tests = [
        ("Environment", test_dwsim_environment),
        ("Python", test_python_environment),
        ("Bridge", test_dwsim_bridge),
    ]

    results = []
    for name, test_func in tests:
        print(f"\n📋 Running {name} Test...")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))

    # Print summary
    print("\n📊 Test Summary:")
    print("─" * 40)
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:10} {status}")
    print("─" * 40)

    # Exit with appropriate code
    all_passed = all(result for _, result in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
