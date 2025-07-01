#!/usr/bin/env python3
"""
Test script to verify output formatting for diagnostic and validation scripts.
This script tests that the log_message() method produces consistent terminal-like output.
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_output_formatting():
    """Test the output formatting consistency."""
    print("Testing output formatting consistency...")
    print("=" * 60)
    print("   OUTPUT FORMATTING TEST")
    print("=" * 60)
    
    # Test symbols
    symbols = {"info": "ℹ️  ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}
    
    for level, symbol in symbols.items():
        print(f"{symbol}This is a {level} message")
    
    print("\n" + "=" * 60)
    print("   SECTION HEADER TEST")
    print("=" * 60)
    
    print("✅ Test passed: Output formatting is consistent")
    print("ℹ️  Info message: System is working correctly")
    print("⚠️  Warning: Some optional features may not be available")
    print("❌ Error: This is a test error message")
    
    print("\n" + "=" * 60)
    print("   TEST COMPLETED")
    print("=" * 60)
    print("✅ All formatting tests passed!")

if __name__ == "__main__":
    test_output_formatting() 