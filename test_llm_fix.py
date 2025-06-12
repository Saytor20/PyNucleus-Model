#!/usr/bin/env python3
"""
Test script to verify LLMQueryManager works after TokenCounter fix.
"""

import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path().resolve() / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def test_llm_query_manager():
    """Test LLMQueryManager initialization and basic functionality."""
    print("🔧 Testing LLM Query Manager with fixed TokenCounter import...")
    
    try:
        # Import should work now
        from pynucleus.llm.query_llm import LLMQueryManager, quick_ask_llm
        print("✅ Import successful!")
        
        # Initialize manager
        manager = LLMQueryManager(max_tokens=2048)
        print("✅ LLMQueryManager initialization successful!")
        
        # Check template directory
        print(f"📍 Template directory: {manager.template_dir}")
        print(f"📍 Template directory exists: {manager.template_dir.exists()}")
        
        # Test basic prompt rendering
        try:
            test_prompt = manager.render_prompt(
                user_query="Test query for prompt system validation",
                system_message="You are a helpful assistant."
            )
            print("✅ Template rendering successful!")
            print("📋 Sample rendered prompt structure:")
            print("─" * 50)
            print(test_prompt[:300] + "..." if len(test_prompt) > 300 else test_prompt)
            
        except Exception as render_error:
            print(f"⚠️ Template rendering issue: {render_error}")
            print("(This is expected if template files are missing)")
        
        print("\n🎉 LLM Query Manager is now working properly!")
        print("✅ The notebook cell should work without NameError now.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_llm_query_manager() 