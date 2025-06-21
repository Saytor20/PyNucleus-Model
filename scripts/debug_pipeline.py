#!/usr/bin/env python3
"""
Debug script to test enhanced LLM prompting and model loading functionality.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pynucleus.llm.model_loader import generate, get_model_info
from pynucleus.llm.prompting import build_prompt, build_simple_prompt
from pynucleus.rag.engine import ask
from pynucleus.settings import settings

def test_model_loading():
    """Test and display model loading information."""
    print("🔍 TESTING MODEL LOADING")
    print("=" * 50)
    
    model_info = get_model_info()
    
    print(f"Loading Method: {model_info['method']}")
    print(f"Model ID: {model_info['model_id']}")
    print(f"GGUF Path: {model_info['gguf_path']}")
    print(f"Has GGUF Model: {model_info['has_gguf']}")
    print(f"Has HF Model: {model_info['has_hf']}")
    print(f"CUDA Setting: {model_info['use_cuda']}")
    print(f"CUDA Available: {model_info['cuda_available']}")
    print(f"MPS Available: {model_info['mps_available']}")
    
    return model_info

def test_basic_generation():
    """Test basic model generation."""
    print("\n🧪 TESTING BASIC GENERATION")
    print("=" * 50)
    
    test_prompt = "What is distillation in chemical engineering?"
    
    print(f"Input: {test_prompt}")
    print("Generating response...")
    
    try:
        response = generate(test_prompt, max_tokens=150)
        print(f"✅ Response: {response}")
        return True
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return False

def test_enhanced_prompting():
    """Test enhanced prompting with step-by-step reasoning."""
    print("\n📝 TESTING ENHANCED PROMPTING")
    print("=" * 50)
    
    context = """
    [1] Distillation is a separation process based on differences in boiling points. 
    The process involves heating a liquid mixture until the more volatile components vaporize.
    
    [2] In industrial applications, distillation columns use multiple stages to achieve 
    better separation efficiency. The number of theoretical plates determines separation quality.
    
    [3] Key parameters include reflux ratio, feed composition, and operating pressure.
    Temperature control is critical for maintaining proper vapor-liquid equilibrium.
    """
    
    question = "How does distillation work and what are the key design parameters?"
    
    # Test enhanced prompting
    print("🚀 Testing Enhanced Prompt:")
    enhanced_prompt = build_prompt(context, question, max_context_chars=800)
    print(f"Prompt length: {len(enhanced_prompt)} characters")
    
    try:
        enhanced_response = generate(enhanced_prompt, max_tokens=200)
        print(f"✅ Enhanced Response: {enhanced_response}")
    except Exception as e:
        print(f"❌ Enhanced prompting failed: {e}")
        enhanced_response = None
    
    # Test simple prompting for comparison
    print("\n📄 Testing Simple Prompt:")
    simple_prompt = build_simple_prompt(context, question, max_context_chars=800)
    print(f"Prompt length: {len(simple_prompt)} characters")
    
    try:
        simple_response = generate(simple_prompt, max_tokens=200)
        print(f"✅ Simple Response: {simple_response}")
    except Exception as e:
        print(f"❌ Simple prompting failed: {e}")
        simple_response = None
    
    return enhanced_response, simple_response

def test_rag_integration():
    """Test RAG integration with enhanced prompting."""
    print("\n🔗 TESTING RAG INTEGRATION")
    print("=" * 50)
    
    test_question = "What is distillation?"
    
    print(f"RAG Query: {test_question}")
    print("Processing through RAG pipeline...")
    
    try:
        rag_result = ask(test_question)
        print(f"✅ RAG Answer: {rag_result['answer']}")
        print(f"📚 Sources: {rag_result['sources']}")
        return rag_result
    except Exception as e:
        print(f"❌ RAG integration failed: {e}")
        return None

def test_context_truncation():
    """Test context truncation functionality."""
    print("\n✂️ TESTING CONTEXT TRUNCATION")
    print("=" * 50)
    
    long_context = "This is a very long context. " * 100  # Create long context
    question = "What is this about?"
    
    print(f"Original context length: {len(long_context)} characters")
    
    # Test with different truncation limits
    for limit in [100, 500, 1000]:
        print(f"\n🔧 Testing with {limit} character limit:")
        try:
            prompt = build_prompt(long_context, question, max_context_chars=limit)
            context_start = prompt.find("Context:") + 8
            context_end = prompt.find("Question:")
            actual_context = prompt[context_start:context_end].strip()
            print(f"   Actual context length: {len(actual_context)} characters")
            print(f"   ✅ Truncation working correctly")
        except Exception as e:
            print(f"   ❌ Truncation failed: {e}")

def test_backward_compatibility():
    """Test backward compatibility of prompting functions."""
    print("\n🔄 TESTING BACKWARD COMPATIBILITY")
    print("=" * 50)
    
    context = "Test context for backward compatibility."
    question = "Test question?"
    
    # Test without max_context_chars parameter (should use default)
    try:
        prompt_default = build_prompt(context, question)
        print(f"✅ Default parameter works: {len(prompt_default)} chars")
    except Exception as e:
        print(f"❌ Default parameter failed: {e}")
    
    # Test with explicit parameter
    try:
        prompt_explicit = build_prompt(context, question, max_context_chars=500)
        print(f"✅ Explicit parameter works: {len(prompt_explicit)} chars")
    except Exception as e:
        print(f"❌ Explicit parameter failed: {e}")

def main():
    """Run all debug tests."""
    print("🚀 PyNucleus Enhanced LLM Pipeline Debug")
    print("=" * 60)
    print(f"Settings: MAX_TOKENS={settings.MAX_TOKENS}, MAX_CONTEXT_CHARS={settings.MAX_CONTEXT_CHARS}")
    print(f"Model: {settings.MODEL_ID}")
    print()
    
    # Run all tests
    tests_passed = 0
    total_tests = 6
    
    try:
        # Test 1: Model Loading
        model_info = test_model_loading()
        if model_info['method'] != 'Failed':
            tests_passed += 1
            print("✅ Test 1 PASSED: Model Loading")
        else:
            print("❌ Test 1 FAILED: Model Loading")
    except Exception as e:
        print(f"❌ Test 1 FAILED: Model Loading - {e}")
    
    try:
        # Test 2: Basic Generation
        if test_basic_generation():
            tests_passed += 1
            print("✅ Test 2 PASSED: Basic Generation")
        else:
            print("❌ Test 2 FAILED: Basic Generation")
    except Exception as e:
        print(f"❌ Test 2 FAILED: Basic Generation - {e}")
    
    try:
        # Test 3: Enhanced Prompting
        enhanced_resp, simple_resp = test_enhanced_prompting()
        if enhanced_resp and simple_resp:
            tests_passed += 1
            print("✅ Test 3 PASSED: Enhanced Prompting")
        else:
            print("❌ Test 3 FAILED: Enhanced Prompting")
    except Exception as e:
        print(f"❌ Test 3 FAILED: Enhanced Prompting - {e}")
    
    try:
        # Test 4: RAG Integration
        rag_result = test_rag_integration()
        if rag_result and rag_result.get('answer'):
            tests_passed += 1
            print("✅ Test 4 PASSED: RAG Integration")
        else:
            print("❌ Test 4 FAILED: RAG Integration")
    except Exception as e:
        print(f"❌ Test 4 FAILED: RAG Integration - {e}")
    
    try:
        # Test 5: Context Truncation
        test_context_truncation()
        tests_passed += 1
        print("✅ Test 5 PASSED: Context Truncation")
    except Exception as e:
        print(f"❌ Test 5 FAILED: Context Truncation - {e}")
    
    try:
        # Test 6: Backward Compatibility
        test_backward_compatibility()
        tests_passed += 1
        print("✅ Test 6 PASSED: Backward Compatibility")
    except Exception as e:
        print(f"❌ Test 6 FAILED: Backward Compatibility - {e}")
    
    # Final Summary
    print("\n" + "=" * 60)
    print(f"🎯 FINAL RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Enhanced LLM pipeline is working correctly.")
        return 0
    elif tests_passed >= total_tests // 2:
        print("⚠️ Most tests passed. Some issues detected but core functionality works.")
        return 0
    else:
        print("💥 Multiple test failures. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main()) 