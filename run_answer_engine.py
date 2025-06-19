#!/usr/bin/env python3
"""
Correct way to run DSPyAnswerEngine
This script shows you how to use the answer engine without import errors
"""

import sys
from pathlib import Path

# Method 1: Add src to path and import
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from pynucleus.llm.answer_engine import DSPyAnswerEngine
    print("✅ Successfully imported DSPyAnswerEngine!")
    
    # Create and test the engine
    engine = DSPyAnswerEngine()
    
    # Get status
    status = engine.get_status()
    print(f"\n📊 Engine Status:")
    print(f"   - DSPy available: {status['dspy_available']}")
    print(f"   - DSPy configured: {status['dspy_configured']}")
    print(f"   - Local DSPy configured: {status['local_dspy_configured']}")
    print(f"   - Simple local configured: {status['simple_local_configured']}")
    print(f"   - Model ID: {status['model_id']}")
    
    # Test a query
    print(f"\n🧪 Testing query...")
    result = engine.answer_general("What is the purpose of a heat exchanger?")
    
    print(f"✅ Query successful!")
    print(f"📝 Answer: {result.get('answer', 'No answer')[:200]}...")
    print(f"⏱️  Time: {result.get('generation_time', 'N/A')} seconds")
    
    print(f"\n🎉 Everything works correctly!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n💡 This happens when you try to run the module directly.")
    print("   The correct ways to run this are:")
    print("   1. python run_answer_engine.py (this script)")
    print("   2. python -m src.pynucleus.llm.answer_engine")
    print("   3. python demo_answer_engine.py")
    print("   4. python test_answer_engine.py")
    
except Exception as e:
    print(f"❌ Runtime Error: {e}")
    print("   This might be due to missing dependencies or model files.") 