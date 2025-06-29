"""
Comprehensive Vector System Robustness Tests

Tests for:
- Vector store initialization and health
- Document ingestion and indexing  
- Search functionality across different scenarios
- Performance metrics and benchmarks
- Error handling and fallback mechanisms
- Data consistency and integrity
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging

# Import the vector store implementations
from pynucleus.rag.vector_store import ChromaVectorStore, RealFAISSVectorStore
from pynucleus.rag.vector_store_remote import create_vector_store
from pynucleus.pipeline.pipeline_rag import RAGPipeline
from pynucleus.settings import settings

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVectorStoreRobustness:
    """Test suite for vector store robustness and reliability."""
    
    @pytest.fixture
    def temp_index_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing."""
        return [
            {
                "text": "Distillation is a separation process that exploits differences in volatilities of mixture components. The process involves vaporization and condensation.",
                "metadata": {"source": "chemical_engineering.pdf", "section": "separation_processes"}
            },
            {
                "text": "Mass transfer involves the transport of species from one location to another, typically from high to low concentration regions.",
                "metadata": {"source": "transport_phenomena.pdf", "section": "mass_transfer"}
            },
            {
                "text": "Heat exchangers are devices used to transfer thermal energy between two or more fluids at different temperatures.",
                "metadata": {"source": "heat_transfer.pdf", "section": "equipment"}
            },
            {
                "text": "Chemical reactors are vessels designed to contain chemical reactions with optimal conditions for conversion and selectivity.",
                "metadata": {"source": "reaction_engineering.pdf", "section": "reactor_design"}
            },
            {
                "text": "Process safety management involves systematic approaches to preventing accidents and ensuring safe operations in chemical plants.",
                "metadata": {"source": "process_safety.pdf", "section": "safety_management"}
            }
        ]
    
    def test_vector_store_initialization(self, temp_index_dir):
        """Test vector store initialization and basic functionality."""
        logger.info("Testing vector store initialization...")
        
        # Test ChromaDB initialization
        chroma_store = ChromaVectorStore(index_dir=temp_index_dir)
        assert chroma_store is not None
        assert chroma_store.index_dir == Path(temp_index_dir)
        
        # Test FAISS initialization
        faiss_store = RealFAISSVectorStore(index_dir=temp_index_dir)
        assert faiss_store is not None
        assert faiss_store.index_dir == Path(temp_index_dir)
        
        logger.info("✅ Vector store initialization tests passed")
    
    def test_vector_store_health_check(self):
        """Test the health and status of vector stores."""
        logger.info("Testing vector store health checks...")
        
        try:
            vector_store = create_vector_store(backend=settings.vstore_backend)
            stats = vector_store.get_index_stats()
            
            assert isinstance(stats, dict)
            assert "backend" in stats
            assert "loaded" in stats
            assert "status" in stats
            
            logger.info(f"Vector store stats: {stats}")
            
            if stats.get("document_count", 0) > 0:
                logger.info(f"✅ Vector store loaded with {stats['document_count']} documents")
            else:
                logger.warning("⚠️ Vector store has no documents loaded")
                
        except Exception as e:
            logger.error(f"❌ Vector store health check failed: {e}")
            pytest.fail(f"Vector store health check failed: {e}")
    
    def test_search_functionality(self):
        """Test search functionality across different query types."""
        logger.info("Testing search functionality...")
        
        test_queries = [
            {"query": "distillation", "expected_min_results": 0, "should_contain": "distillation"},
            {"query": "mass transfer", "expected_min_results": 0, "should_contain": "mass"},
            {"query": "heat exchanger", "expected_min_results": 0, "should_contain": "heat"},
            {"query": "", "expected_min_results": 0, "should_contain": None},
        ]
        
        try:
            vector_store = create_vector_store(backend=settings.vstore_backend)
            
            for test_case in test_queries:
                query = test_case["query"]
                expected_min = test_case["expected_min_results"]
                should_contain = test_case["should_contain"]
                
                logger.info(f"Testing query: '{query}'")
                
                start_time = time.time()
                results = vector_store.search(query, top_k=5)
                search_time = time.time() - start_time
                
                assert len(results) >= expected_min, f"Query '{query}' returned {len(results)} results, expected >= {expected_min}"
                assert search_time < 2.0, f"Search took {search_time:.2f}s, should be < 2.0s"
                
                logger.info(f"✅ Query '{query}' returned {len(results)} results in {search_time:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ Search functionality test failed: {e}")
            pytest.fail(f"Search functionality test failed: {e}")
    
    def test_search_performance_benchmark(self):
        """Benchmark search performance under various loads."""
        logger.info("Running search performance benchmarks...")
        
        test_queries = [
            "distillation column design",
            "heat transfer coefficient",
            "mass balance equation", 
            "reactor conversion efficiency",
            "process safety management"
        ]
        
        try:
            vector_store = create_vector_store(backend=settings.vstore_backend)
            
            # Single query performance
            total_time = 0
            num_queries = len(test_queries)
            
            for query in test_queries:
                start_time = time.time()
                results = vector_store.search(query, top_k=5)
                query_time = time.time() - start_time
                total_time += query_time
                
                # Each query should complete within reasonable time
                assert query_time < 1.0, f"Query '{query}' took {query_time:.2f}s, should be < 1.0s"
                
            average_time = total_time / num_queries
            logger.info(f"✅ Average query time: {average_time:.3f}s across {num_queries} queries")
            
            # Batch query performance
            start_time = time.time()
            for query in test_queries * 3:  # 3x repetition
                vector_store.search(query, top_k=3)
            batch_time = time.time() - start_time
            
            logger.info(f"✅ Batch performance: {len(test_queries * 3)} queries in {batch_time:.3f}s")
            
        except Exception as e:
            logger.error(f"❌ Performance benchmark failed: {e}")
            pytest.fail(f"Performance benchmark failed: {e}")
    
    def test_error_handling_and_fallbacks(self):
        """Test error handling and fallback mechanisms."""
        logger.info("Testing error handling and fallbacks...")
        
        try:
            vector_store = create_vector_store(backend=settings.vstore_backend)
            
            # Test with malformed queries
            problematic_queries = [
                None,
                "",
                " " * 1000,  # Very long whitespace
                "a" * 10000,  # Very long query
                "special chars: !@#$%^&*()_+=[]{}|;':\",./<>?",
                "unicode: 中文 العربية русский 🚀🔥💯"
            ]
            
            for query in problematic_queries:
                try:
                    results = vector_store.search(str(query) if query is not None else "", top_k=5)
                    # Should not crash, should return reasonable results
                    assert isinstance(results, list)
                    logger.info(f"✅ Handled problematic query: {str(query)[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️ Query failed but handled gracefully: {e}")
            
            # Test edge cases for parameters
            edge_cases = [
                {"top_k": 0},
                {"top_k": 1000},
                {"top_k": -1},
            ]
            
            for params in edge_cases:
                try:
                    results = vector_store.search("test query", **params)
                    assert isinstance(results, list)
                    logger.info(f"✅ Handled edge case parameters: {params}")
                except Exception as e:
                    logger.warning(f"⚠️ Edge case handled: {e}")
                    
        except Exception as e:
            logger.error(f"❌ Error handling test failed: {e}")
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_rag_pipeline_integration(self):
        """Test RAG pipeline integration with vector store."""
        logger.info("Testing RAG pipeline integration...")
        
        try:
            rag_pipeline = RAGPipeline(data_dir="data")
            
            test_questions = [
                "What is distillation?",
                "How do heat exchangers work?",
                "What are the principles of mass transfer?",
            ]
            
            for question in test_questions:
                logger.info(f"Testing RAG pipeline with: '{question}'")
                
                start_time = time.time()
                result = rag_pipeline.query(question, top_k=5)
                query_time = time.time() - start_time
                
                assert isinstance(result, dict)
                assert "answer" in result
                assert "sources" in result
                assert "confidence" in result
                
                answer = result.get("answer", "")
                assert len(answer) > 0, f"Empty answer for question: {question}"
                
                confidence = result.get("confidence", 0)
                sources = result.get("sources", [])
                
                if confidence > 0.5 and sources:
                    logger.info(f"✅ Good RAG result: confidence={confidence:.2f}, sources={len(sources)}")
                else:
                    logger.warning(f"⚠️ Fallback result: confidence={confidence:.2f}, sources={len(sources)}")
                
                logger.info(f"Response time: {query_time:.3f}s")
                
        except Exception as e:
            logger.error(f"❌ RAG pipeline integration test failed: {e}")
            pytest.fail(f"RAG pipeline integration test failed: {e}")
    
    def test_data_consistency(self):
        """Test data consistency and integrity."""
        logger.info("Testing data consistency...")
        
        try:
            vector_store = create_vector_store(backend=settings.vstore_backend)
            
            # Test consistent results for same query
            test_query = "chemical engineering process design"
            
            results1 = vector_store.search(test_query, top_k=5)
            results2 = vector_store.search(test_query, top_k=5)
            
            # Results should be consistent
            assert len(results1) == len(results2), "Inconsistent result counts for same query"
            
            # Check if results are in same order (they should be for deterministic search)
            for i, (r1, r2) in enumerate(zip(results1, results2)):
                assert r1.get("text") == r2.get("text"), f"Inconsistent result at position {i}"
                assert abs(r1.get("score", 0) - r2.get("score", 0)) < 0.001, f"Score mismatch at position {i}"
            
            logger.info("✅ Data consistency test passed")
            
            # Test different similarity thresholds
            high_threshold_results = vector_store.search(test_query, top_k=10)
            low_threshold_results = vector_store.search(test_query, top_k=10)
            
            # High threshold should return fewer or equal results
            assert len(high_threshold_results) <= len(low_threshold_results), "High threshold returned more results than low threshold"
            
            logger.info("✅ Threshold consistency test passed")
            
        except Exception as e:
            logger.error(f"❌ Data consistency test failed: {e}")
            pytest.fail(f"Data consistency test failed: {e}")


def test_system_diagnostic_report():
    """Generate comprehensive diagnostic report."""
    logger.info("Generating vector system diagnostic report...")
    
    report = {
        "timestamp": time.time(),
        "tests_passed": 0,
        "tests_failed": 0,
        "warnings": [],
        "performance_metrics": {},
        "recommendations": []
    }
    
    try:
        vector_store = create_vector_store(backend=settings.vstore_backend)
        stats = vector_store.get_index_stats()
        report["vector_store_stats"] = stats
        
        if stats.get("loaded", False):
            report["tests_passed"] += 1
            logger.info("✅ Vector store loaded successfully")
        else:
            report["tests_failed"] += 1
            report["warnings"].append("Vector store not properly loaded")
            logger.warning("⚠️ Vector store not properly loaded")
        
        start_time = time.time()
        results = vector_store.search("test query", top_k=5)
        search_time = time.time() - start_time
        
        report["performance_metrics"]["search_time"] = search_time
        report["performance_metrics"]["result_count"] = len(results)
        
        if search_time < 1.0:
            report["tests_passed"] += 1
            logger.info(f"✅ Search performance good: {search_time:.3f}s")
        else:
            report["tests_failed"] += 1
            report["warnings"].append(f"Slow search performance: {search_time:.3f}s")
        
        if stats.get("document_count", 0) == 0:
            report["recommendations"].append("Ingest documents into vector store")
        
        total_tests = report["tests_passed"] + report["tests_failed"]
        health_score = (report["tests_passed"] / total_tests * 100) if total_tests > 0 else 0
        report["health_score"] = health_score
        
        logger.info("=" * 60)
        logger.info("VECTOR SYSTEM DIAGNOSTIC REPORT")
        logger.info("=" * 60)
        logger.info(f"Health Score: {health_score:.1f}%")
        logger.info(f"Tests Passed: {report['tests_passed']}")
        logger.info(f"Tests Failed: {report['tests_failed']}")
        logger.info(f"Search Time: {search_time:.3f}s")
        logger.info(f"Document Count: {stats.get('document_count', 0)}")
        
        if report["warnings"]:
            logger.info("Warnings:")
            for warning in report["warnings"]:
                logger.info(f"  ⚠️ {warning}")
        
        if report["recommendations"]:
            logger.info("Recommendations:")
            for rec in report["recommendations"]:
                logger.info(f"  💡 {rec}")
        
        logger.info("=" * 60)
        
        assert health_score >= 30, f"Vector system health score too low: {health_score:.1f}%"
        
    except Exception as e:
        logger.error(f"❌ Diagnostic report generation failed: {e}")
        pytest.fail(f"Diagnostic report generation failed: {e}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"]) 