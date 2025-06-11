#!/usr/bin/env python3
"""
Usage example for PyNucleus Token Utilities.

This script demonstrates how to use the token counting functionality.
"""

import sys
from pathlib import Path

# Add the current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from token_utils import TokenCounter, count_tokens, get_available_cache_info, clear_all_caches

def demonstrate_token_counting():
    """Demonstrate basic token counting functionality."""
    print("🚀 PyNucleus Token Utilities Demo")
    print("=" * 50)
    
    # Sample texts
    texts = [
        "Hello, world!",
        "This is a longer sentence with more tokens to demonstrate the utility.",
        "Chemical processes involve complex molecular interactions that require precise modeling.",
        "PyNucleus provides comprehensive RAG and simulation capabilities for chemical engineering."
    ]
    
    print("1. Basic Token Counting with Default Model (GPT-2)")
    print("-" * 50)
    
    for i, text in enumerate(texts, 1):
        count = count_tokens(text)
        print(f"Text {i}: '{text}'")
        print(f"Tokens: {count}")
        print()
    
    print("2. Using TokenCounter Class for Advanced Features")
    print("-" * 50)
    
    counter = TokenCounter("gpt2")
    text = "Advanced tokenization with detailed token inspection."
    
    # Get token count
    token_count = counter.count_tokens(text)
    # Get actual tokens
    tokens = counter.get_tokens(text)
    
    print(f"Text: '{text}'")
    print(f"Token count: {token_count}")
    print(f"Actual tokens: {tokens}")
    print()
    
    print("3. Batch Processing with Lists")
    print("-" * 50)
    
    batch_texts = [
        "Chemical reactor design",
        "Distillation column optimization", 
        "Heat exchanger efficiency"
    ]
    
    batch_counts = counter.count_tokens(batch_texts)
    print("Batch token counts:")
    for text, count in zip(batch_texts, batch_counts):
        print(f"  '{text}': {count} tokens")
    print()
    
    print("4. Caching Performance Demo")
    print("-" * 50)
    
    # Show cache performance
    import time
    
    test_text = "Performance testing for cached tokenization operations."
    
    # First call (cache miss)
    start = time.time()
    count1 = count_tokens(test_text)
    first_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    count2 = count_tokens(test_text)
    second_time = time.time() - start
    
    print(f"Text: '{test_text}'")
    print(f"First call: {first_time:.6f}s (cache miss)")
    print(f"Second call: {second_time:.6f}s (cache hit)")
    print(f"Both calls returned: {count1} tokens")
    print()
    
    print("5. Cache Information")
    print("-" * 50)
    
    cache_info = get_available_cache_info()
    print("Cache statistics:")
    for cache_type, stats in cache_info.items():
        print(f"  {cache_type}:")
        print(f"    Hits: {stats['hits']}")
        print(f"    Misses: {stats['misses']}")
        print(f"    Current size: {stats['current_size']}")
        print(f"    Max size: {stats['max_size']}")
        if stats['hits'] + stats['misses'] > 0:
            hit_rate = stats['hits'] / (stats['hits'] + stats['misses']) * 100
            print(f"    Hit rate: {hit_rate:.1f}%")
    print()
    
    print("6. Different Models Comparison")
    print("-" * 50)
    
    test_text = "Compare tokenization across different models."
    models = ["gpt2", "distilbert-base-uncased"]
    
    for model in models:
        try:
            model_count = count_tokens(test_text, model)
            print(f"Model '{model}': {model_count} tokens")
        except Exception as e:
            print(f"Model '{model}': Error - {e}")
    print()
    
    print("✅ Demo completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        demonstrate_token_counting()
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        sys.exit(1) 