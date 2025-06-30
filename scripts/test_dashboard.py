#!/usr/bin/env python3
"""
Test script for PyNucleus Dashboard functionality

This script tests the core components of the dashboard to ensure
they work correctly before deployment.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    try:
        from pynucleus.diagnostics.runner import DiagnosticRunner
        print("✅ DiagnosticRunner imported successfully")
    except ImportError as e:
        print(f"⚠️  DiagnosticRunner import failed: {e}")
        print("   This may be normal if PyNucleus isn't fully set up")
    
    try:
        from pynucleus.eval.confidence_calibration import ConfidenceCalibrator
        print("✅ ConfidenceCalibrator imported successfully")
    except ImportError as e:
        print(f"⚠️  ConfidenceCalibrator import failed: {e}")
        print("   This may be normal if PyNucleus isn't fully set up")
    
    return True

def test_dashboard_components():
    """Test dashboard component initialization"""
    print("\nTesting dashboard components...")
    
    try:
        from pynucleus.diagnostics.dashboard import DashboardDataManager, DashboardUI
        print("✅ Dashboard classes imported successfully")
        
        # Test data manager initialization
        data_manager = DashboardDataManager()
        print("✅ DashboardDataManager initialized")
        
        # Test UI initialization
        ui = DashboardUI(data_manager)
        print("✅ DashboardUI initialized")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard component test failed: {e}")
        return False

def test_log_parsing():
    """Test log file parsing functionality"""
    print("\nTesting log parsing...")
    
    try:
        from pynucleus.diagnostics.dashboard import DashboardDataManager
        
        data_manager = DashboardDataManager()
        
        # Create a test log entry
        test_log_line = "2024-06-24 16:33:00,325 INFO app: Question processed successfully in 0.12s"
        parsed = data_manager._parse_log_line(test_log_line)
        
        if parsed:
            print("✅ Log parsing works correctly")
            print(f"   Parsed: {parsed}")
            return True
        else:
            print("❌ Log parsing returned None")
            return False
            
    except Exception as e:
        print(f"❌ Log parsing test failed: {e}")
        return False

def test_file_structure():
    """Test that required files and directories exist"""
    print("\nTesting file structure...")
    
    dashboard_file = root_dir / "src" / "pynucleus" / "diagnostics" / "dashboard.py"
    if dashboard_file.exists():
        print("✅ Dashboard file exists")
    else:
        print("❌ Dashboard file not found")
        return False
    
    logs_dir = root_dir / "logs"
    if logs_dir.exists():
        print("✅ Logs directory exists")
    else:
        print("⚠️  Logs directory not found (will be created when needed)")
    
    launcher_script = root_dir / "scripts" / "launch_dashboard.sh"
    if launcher_script.exists():
        print("✅ Launcher script exists")
        if os.access(launcher_script, os.X_OK):
            print("✅ Launcher script is executable")
        else:
            print("⚠️  Launcher script is not executable")
    else:
        print("❌ Launcher script not found")
        return False
    
    return True

def test_requirements():
    """Test that required packages are installed"""
    print("\nTesting requirements...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'numpy'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🔬 PyNucleus Dashboard Test Suite")
    print("=" * 40)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Requirements", test_requirements),
        ("Imports", test_imports),
        ("Dashboard Components", test_dashboard_components),
        ("Log Parsing", test_log_parsing)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'=' * 20}")
        print(f"Running: {test_name}")
        print(f"{'=' * 20}")
        
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'=' * 40}")
    print("TEST SUMMARY")
    print(f"{'=' * 40}")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Dashboard is ready to deploy.")
        print("\nTo launch the dashboard:")
        print("  ./scripts/launch_dashboard.sh")
        print("  or")
        print("  streamlit run src/pynucleus/diagnostics/dashboard.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please resolve issues before deployment.")
        
        if not results.get("Requirements", True):
            print("\nTo install missing requirements:")
            print("  pip install streamlit plotly")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 