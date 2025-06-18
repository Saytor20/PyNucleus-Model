#!/usr/bin/env python3
"""Test script for FAISS integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pynucleus.rag.vector_store import RealFAISSVectorStore

def test_faiss_integration():
    """Test the real FAISS vector store integration."""
    print("🔍 Testing Real FAISS Vector Store...")
    
    try:
        store = RealFAISSVectorStore()
        
        print(f"FAISS Loaded: {store.loaded}")
        if store.loaded:
            stats = store.get_index_stats()
            print(f"Vector Count: {stats.get('total_vectors', 'Unknown')}")
            print(f"Dimensions: {stats.get('dimensions', 'Unknown')}")
            
            # Test search with different thresholds and queries
            test_queries = [
                ("modular chemical plants", 0.1),
                ("distillation optimization", 0.1), 
                ("process efficiency", 0.1),
                ("reactor design", 0.1)
            ]
            
            for query, threshold in test_queries:
                print(f"\n🔍 Testing search: '{query}' (threshold: {threshold})")
                results = store.search(query, top_k=3, similarity_threshold=threshold)
                print(f"Search Results: {len(results)}")
                
                if results:
                    for i, result in enumerate(results):
                        print(f"  Result {i+1}:")
                        print(f"    Score: {result['score']:.3f}")
                        print(f"    Source: {result['source']}")
                        print(f"    Text preview: {result['text'][:100]}...")
                else:
                    print(f"  No results found for '{query}'")
        else:
            print("❌ FAISS not loaded - check logs for details")
            
    except Exception as e:
        print(f"❌ Error testing FAISS integration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_faiss_integration() 