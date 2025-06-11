#!/usr/bin/env python3
"""
Tests for DWSIM bridge functionality
"""

import os
import pytest
from pathlib import Path
from pynucleus.simulation.dwsim_bridge import DWSIMBridge


def test_dwsim_environment():
    """Test DWSIM environment setup."""
    # Check DWSIM DLL path
    assert os.getenv("DWSIM_DLL_PATH"), "DWSIM_DLL_PATH not set"

    # Check DLL directory
    dll_path = Path(os.getenv("DWSIM_DLL_PATH"))
    assert dll_path.exists(), f"DWSIM DLL directory not found: {dll_path}"

    # Check DLL files
    dlls = list(dll_path.glob("*.dll"))
    assert len(dlls) > 0, "No DLL files found"

    # Check required DLLs
    required_dlls = [
        "DWSIM.Thermodynamics.dll",
        "DWSIM.UnitOperations.dll",
        "DWSIM.Interfaces.dll",
    ]
    for dll in required_dlls:
        assert any(dll in str(d) for d in dlls), f"Required DLL not found: {dll}"


def test_dwsim_bridge_initialization():
    """Test DWSIM bridge initialization."""
    with DWSIMBridge() as bridge:
        assert bridge is not None, "Failed to initialize DWSIM bridge"

        # Test creating a new flowsheet
        flowsheet = bridge.create_flowsheet()
        assert flowsheet is not None, "Failed to create flowsheet"


def test_dwsim_operations():
    """Test basic DWSIM operations."""
    with DWSIMBridge() as bridge:
        # Create a new flowsheet
        flowsheet = bridge.create_flowsheet()

        # Test adding a stream
        stream = bridge.add_stream(flowsheet, "TestStream")
        assert stream is not None, "Failed to add stream"

        # Test adding a unit operation
        unit = bridge.add_unit_operation(flowsheet, "TestUnit", "Mixer")
        assert unit is not None, "Failed to add unit operation"


def test_error_handling():
    """Test error handling in DWSIM bridge."""
    with pytest.raises(RuntimeError):
        # Test with invalid DLL path
        os.environ["DWSIM_DLL_PATH"] = "/invalid/path"
        DWSIMBridge()

    # Restore correct path
    os.environ["DWSIM_DLL_PATH"] = str(Path("dwsim_libs").absolute())
