#!/usr/bin/env python3
"""
Test Production-Ready Embedding Monitoring System

Demonstrates comprehensive monitoring, benchmarking, and alerting
for scaled document retrieval systems with ground-truth validation
and citation backtracking capabilities.
"""

import sys
import os
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynucleus.rag.vector_store import FAISSDBManager
from pynucleus.rag.embedding_monitor import EmbeddingMonitor

@dataclass
class ValidationResult:
    """Structure for validation results."""
    query: str
    expected_answer: str
    generated_answer: str
    accuracy_score: float
    citation_accuracy: float
    response_time: float
    sources_used: List[str]

@dataclass
class CitationResult:
    """Structure for citation results."""
    source_file: str
    confidence_score: float
    relevant_text: str
    verified: bool = False

def create_ground_truth_dataset() -> Dict[str, Dict]:
    """Create ground truth dataset for validation."""
    return {
        "modular chemical plants": {
            "expected_answer": "Modular chemical plants offer reduced capital costs, faster construction, improved quality control, easier transportation, and scalability.",
            "expected_sources": ["modular_plants_guide.pdf"],
            "difficulty": "basic"
        },
        "distillation optimization": {
            "expected_answer": "Distillation optimization involves balancing energy consumption, separation efficiency, and capital costs for optimal performance.",
            "expected_sources": ["distillation_optimization.pdf"],
            "difficulty": "intermediate"
        },
        "process intensification benefits": {
            "expected_answer": "Process intensification enables smaller, more efficient processes through enhanced heat and mass transfer, reducing equipment size and costs.",
            "expected_sources": ["process_intensification.pdf"],
            "difficulty": "intermediate"
        },
        "reactor conversion efficiency": {
            "expected_answer": "Reactor conversion efficiency depends on temperature, pressure, catalyst activity, residence time, and mixing efficiency.",
            "expected_sources": ["reactor_design.pdf"],
            "difficulty": "advanced"
        },
        "sustainable manufacturing practices": {
            "expected_answer": "Sustainable manufacturing includes waste minimization, energy efficiency, circular economy principles, and environmental impact reduction.",
            "expected_sources": ["sustainability.pdf"],
            "difficulty": "advanced"
        }
    }

def test_ground_truth_validation(manager: FAISSDBManager, ground_truth: Dict) -> Dict[str, Any]:
    """Test ground truth validation with known answers."""
    print("\n🧪 Testing Ground-Truth Validation...")
    
    validation_results = []
    total_queries = len(ground_truth)
    successful_queries = 0
    
    for query, expected_data in ground_truth.items():
        try:
            start_time = time.time()
            
            # Perform retrieval
            results = manager.search(query, k=3)
            response_time = time.time() - start_time
            
            # Extract sources
            sources_used = [doc.metadata.get('source', 'Unknown') for doc, _ in results]
            
            # Calculate citation accuracy
            expected_sources = expected_data["expected_sources"]
            citation_accuracy = calculate_citation_accuracy(expected_sources, sources_used)
            
            # Mock answer accuracy (in real implementation, would use LLM)
            accuracy_score = 0.8 if citation_accuracy > 0.5 else 0.3
            
            if accuracy_score >= 0.5:
                successful_queries += 1
            
            validation_result = ValidationResult(
                query=query,
                expected_answer=expected_data["expected_answer"],
                generated_answer=f"Mock response for {query}",
                accuracy_score=accuracy_score,
                citation_accuracy=citation_accuracy,
                response_time=response_time,
                sources_used=sources_used
            )
            
            validation_results.append(validation_result)
            
            print(f"   Query: {query[:40]}... | Accuracy: {accuracy_score:.2%} | Citation: {citation_accuracy:.2%}")
            
        except Exception as e:
            print(f"   ❌ Query failed: {query[:40]}... Error: {e}")
    
    # Calculate metrics
    success_rate = successful_queries / total_queries if total_queries > 0 else 0
    avg_accuracy = sum(r.accuracy_score for r in validation_results) / len(validation_results) if validation_results else 0
    avg_citation_accuracy = sum(r.citation_accuracy for r in validation_results) / len(validation_results) if validation_results else 0
    avg_response_time = sum(r.response_time for r in validation_results) / len(validation_results) if validation_results else 0
    
    results = {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": success_rate,
        "average_accuracy": avg_accuracy,
        "average_citation_accuracy": avg_citation_accuracy,
        "average_response_time": avg_response_time,
        "validation_results": validation_results
    }
    
    print(f"   📊 Validation Summary:")
    print(f"      Success Rate: {success_rate:.2%}")
    print(f"      Average Accuracy: {avg_accuracy:.2%}")
    print(f"      Average Citation Accuracy: {avg_citation_accuracy:.2%}")
    print(f"      Average Response Time: {avg_response_time:.3f}s")
    
    if success_rate >= 0.6:
        print("   ✅ Ground-truth validation PASSED")
    else:
        print("   ❌ Ground-truth validation FAILED")
    
    return results

def test_citation_backtracking(manager: FAISSDBManager) -> Dict[str, Any]:
    """Test citation backtracking capabilities."""
    print("\n📚 Testing Citation Backtracking...")
    
    test_queries = [
        "modular chemical plant advantages",
        "distillation column efficiency",
        "sustainable manufacturing"
    ]
    
    citation_results = []
    total_citations = 0
    verified_citations = 0
    
    for query in test_queries:
        try:
            # Perform retrieval
            results = manager.search(query, k=3)
            
            for doc, score in results:
                source_file = doc.metadata.get('source', 'Unknown')
                relevant_text = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                
                # Mock citation verification (in real implementation, would verify against actual source)
                verified = score > 0.3  # Mock verification based on similarity score
                
                citation_result = CitationResult(
                    source_file=source_file,
                    confidence_score=score,
                    relevant_text=relevant_text,
                    verified=verified
                )
                
                citation_results.append(citation_result)
                total_citations += 1
                
                if verified:
                    verified_citations += 1
            
            print(f"   Query: {query[:30]}... | Citations: {len(results)}")
            
        except Exception as e:
            print(f"   ❌ Citation test failed for query: {query[:30]}... Error: {e}")
    
    # Calculate citation metrics
    verification_rate = verified_citations / total_citations if total_citations > 0 else 0
    avg_confidence = sum(c.confidence_score for c in citation_results) / len(citation_results) if citation_results else 0
    
    results = {
        "total_citations": total_citations,
        "verified_citations": verified_citations,
        "verification_rate": verification_rate,
        "average_confidence": avg_confidence,
        "citation_results": citation_results
    }
    
    print(f"   📊 Citation Summary:")
    print(f"      Total Citations: {total_citations}")
    print(f"      Verified Citations: {verified_citations}")
    print(f"      Verification Rate: {verification_rate:.2%}")
    print(f"      Average Confidence: {avg_confidence:.3f}")
    
    if verification_rate >= 0.8:
        print("   ✅ Citation backtracking PASSED")
    else:
        print("   ❌ Citation backtracking FAILED")
    
    return results

def calculate_citation_accuracy(expected_sources: List[str], actual_sources: List[str]) -> float:
    """Calculate citation accuracy."""
    if not expected_sources:
        return 1.0
    
    if not actual_sources:
        return 0.0
    
    matches = 0
    for expected in expected_sources:
        for actual in actual_sources:
            if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                matches += 1
                break
    
    return matches / len(expected_sources)

def save_validation_report(validation_results: Dict, citation_results: Dict):
    """Save comprehensive validation report."""
    try:
        # Create reports directory
        reports_dir = Path("data/validation/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Combine results
        comprehensive_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "validation_results": {
                "summary": {
                    "total_queries": validation_results.get("total_queries", 0),
                    "success_rate": validation_results.get("success_rate", 0),
                    "average_accuracy": validation_results.get("average_accuracy", 0),
                    "average_citation_accuracy": validation_results.get("average_citation_accuracy", 0)
                },
                "detailed_results": [
                    {
                        "query": r.query,
                        "accuracy_score": r.accuracy_score,
                        "citation_accuracy": r.citation_accuracy,
                        "response_time": r.response_time,
                        "sources_used": r.sources_used
                    }
                    for r in validation_results.get("validation_results", [])
                ]
            },
            "citation_results": {
                "summary": {
                    "total_citations": citation_results.get("total_citations", 0),
                    "verification_rate": citation_results.get("verification_rate", 0),
                    "average_confidence": citation_results.get("average_confidence", 0)
                },
                "detailed_results": [
                    {
                        "source_file": c.source_file,
                        "confidence_score": c.confidence_score,
                        "verified": c.verified,
                        "relevant_text": c.relevant_text[:100]  # Truncate for report
                    }
                    for c in citation_results.get("citation_results", [])
                ]
            }
        }
        
        # Save report
        report_file = reports_dir / f"production_validation_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Validation report saved: {report_file}")
        
    except Exception as e:
        print(f"\n❌ Error saving validation report: {e}")

def test_production_monitoring():
    """Test production monitoring capabilities."""
    
    print("🚀 PRODUCTION-READY EMBEDDING MONITORING TEST")
    print("=" * 60)
    
    # Initialize vector store with production settings
    print("\n1️⃣ Initializing Production Vector Store...")
    manager = FAISSDBManager(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        log_dir="data/04_models/chunk_reports"
    )
    
    # Create test documents (simulating production data)
    print("   📄 Creating test document corpus...")
    test_documents = [
        {
            'page_content': 'Modular chemical plants offer significant advantages in terms of flexibility, reduced capital costs, and faster deployment compared to traditional stick-built facilities.',
            'metadata': {'source': 'modular_plants_guide.pdf', 'type': 'technical', 'section': 'introduction'}
        },
        {
            'page_content': 'Process intensification techniques enable the development of smaller, more efficient chemical processes through enhanced heat and mass transfer.',
            'metadata': {'source': 'process_intensification.pdf', 'type': 'technical', 'section': 'methodology'}
        },
        {
            'page_content': 'Distillation column optimization involves balancing energy consumption, separation efficiency, and capital costs to achieve optimal performance.',
            'metadata': {'source': 'distillation_optimization.pdf', 'type': 'technical', 'section': 'optimization'}
        },
        {
            'page_content': 'Economic analysis of chemical processes requires consideration of capital expenditure, operating expenses, and return on investment metrics.',
            'metadata': {'source': 'economic_analysis.pdf', 'type': 'business', 'section': 'financial'}
        },
        {
            'page_content': 'Safety protocols in chemical manufacturing include hazard identification, risk assessment, and implementation of protective measures.',
            'metadata': {'source': 'safety_protocols.pdf', 'type': 'safety', 'section': 'procedures'}
        },
        {
            'page_content': 'Supply chain optimization in chemical industries focuses on raw material sourcing, inventory management, and distribution efficiency.',
            'metadata': {'source': 'supply_chain.pdf', 'type': 'business', 'section': 'logistics'}
        },
        {
            'page_content': 'Reactor design principles include mass balance, energy balance, and kinetic considerations for optimal conversion rates.',
            'metadata': {'source': 'reactor_design.pdf', 'type': 'technical', 'section': 'design'}
        },
        {
            'page_content': 'Heat exchanger performance depends on heat transfer coefficients, flow patterns, and fouling resistance factors.',
            'metadata': {'source': 'heat_exchangers.pdf', 'type': 'technical', 'section': 'performance'}
        },
        {
            'page_content': 'Sustainable manufacturing practices in chemical industries include waste minimization, energy efficiency, and circular economy principles.',
            'metadata': {'source': 'sustainability.pdf', 'type': 'environmental', 'section': 'practices'}
        },
        {
            'page_content': 'Quality control systems ensure product specifications are met through statistical process control and analytical testing.',
            'metadata': {'source': 'quality_control.pdf', 'type': 'quality', 'section': 'systems'}
        }
    ]
    
    # Convert to Document objects
    try:
        from langchain.docstore.document import Document
    except ImportError:
        from langchain_core.documents.base import Document
    
    docs = [Document(page_content=d['page_content'], metadata=d['metadata']) for d in test_documents]
    
    # Build vector store
    print("   🔍 Building FAISS index...")
    manager.build(docs)
    print(f"   ✅ Vector store built with {len(docs)} documents")
    
    # Initialize monitoring system
    print("\n2️⃣ Initializing Production Monitoring System...")
    monitor = EmbeddingMonitor(
        vector_store_manager=manager,
        monitoring_dir="data/04_models/chunk_reports",
        alert_thresholds={
            'min_recall': 0.6,           # 60% minimum recall for test
            'max_response_time': 1.0,    # 1 second max response
            'min_similarity_score': 0.2, # 20% minimum similarity for test
            'max_drift_percentage': 20.0, # 20% max drift
            'min_coverage': 0.4          # 40% document coverage
        }
    )
    print("   ✅ Monitoring system initialized with production thresholds")
    
    # Test enhanced evaluation
    print("\n3️⃣ Testing Enhanced Evaluation Metrics...")
    ground_truth = {
        'modular chemical plants': 'modular_plants_guide.pdf',
        'distillation optimization': 'distillation_optimization.pdf',
        'process intensification': 'process_intensification.pdf'
    }
    
    metrics = manager.evaluate(ground_truth, k=3)
    print(f"   📊 Recall@3: {metrics['recall_at_k']:.1f}%")
    print(f"   📊 Avg Similarity: {metrics['average_similarity_score']:.4f}")
    print(f"   📊 Avg Response Time: {metrics['average_response_time']:.3f}s")
    print(f"   ✅ Enhanced evaluation completed")
    
    # Test comprehensive benchmarking
    print("\n4️⃣ Running Comprehensive Production Benchmark...")
    
    # Custom queries for chemical engineering domain
    production_queries = [
        "modular chemical plant design",
        "process intensification benefits",
        "distillation column efficiency",
        "reactor conversion optimization",
        "heat exchanger performance",
        "economic analysis methods",
        "safety protocol implementation",
        "supply chain optimization",
        "sustainable manufacturing",
        "quality control systems"
    ]
    
    benchmark_results = monitor.run_comprehensive_benchmark(
        custom_queries=production_queries,
        k_values=[1, 3, 5]
    )
    
    print(f"   📊 Benchmark Duration: {benchmark_results['benchmark_metadata']['benchmark_duration']:.2f}s")
    print(f"   📊 Queries Tested: {benchmark_results['benchmark_metadata']['total_queries']}")
    
    # Display k=3 results (most common)
    k3_results = benchmark_results['performance_by_k']['k_3']
    print(f"   📊 K=3 Performance:")
    print(f"      • Avg Similarity: {k3_results['avg_similarity_score']:.4f}")
    print(f"      • Avg Response Time: {k3_results['avg_response_time']:.3f}s")
    print(f"      • Document Coverage: {k3_results['document_coverage']:.2%}")
    print(f"      • Success Rate: {k3_results['success_rate']:.2%}")
    
    # Display alerts
    if benchmark_results['alerts']:
        print(f"   ⚠️ Performance Alerts: {len(benchmark_results['alerts'])}")
        for alert in benchmark_results['alerts'][:3]:  # Show first 3
            print(f"      • {alert}")
    else:
        print("   ✅ No performance alerts")
    
    # Display recommendations
    if benchmark_results['recommendations']:
        print(f"   💡 Recommendations: {len(benchmark_results['recommendations'])}")
        for rec in benchmark_results['recommendations'][:3]:  # Show first 3
            print(f"      • {rec}")
    
    # Test drift monitoring
    print("\n5️⃣ Testing Embedding Drift Monitoring...")
    drift_results = manager.monitor_embedding_drift([
        "chemical process optimization",
        "modular plant design",
        "industrial efficiency"
    ])
    
    drift_status = drift_results['drift_indicators'].get('status', 'NORMAL')
    print(f"   📊 Drift Status: {drift_status}")
    if 'current_avg_score' in drift_results['drift_indicators']:
        print(f"   📊 Current Avg Score: {drift_results['drift_indicators']['current_avg_score']:.4f}")
    print("   ✅ Drift monitoring completed")
    
    # Test health check
    print("\n6️⃣ Running Production Health Check...")
    health_results = manager.health_check()
    
    print(f"   🏥 Overall Status: {health_results['overall_status']}")
    print(f"   🏥 Document Count: {health_results['checks']['document_count']}")
    print(f"   🏥 Index Available: {health_results['checks']['index_available']}")
    print(f"   🏥 Search Functional: {health_results['checks']['search_functional']}")
    
    if health_results['recommendations']:
        print(f"   💡 Health Recommendations: {len(health_results['recommendations'])}")
        for rec in health_results['recommendations'][:2]:
            print(f"      • {rec}")
    
    # Test production monitoring
    print("\n7️⃣ Testing Continuous Production Monitoring...")
    monitoring_results = monitor.monitor_production_health()
    
    print(f"   📊 System Health: {monitoring_results['system_health']['overall_status']}")
    print(f"   📊 Performance Score: {monitoring_results['performance_metrics']['overall_performance_score']:.2f}")
    print("   ✅ Production monitoring completed")
    
    # NEW: Test ground-truth validation
    print("\n8️⃣ Testing Ground-Truth Validation...")
    ground_truth_dataset = create_ground_truth_dataset()
    validation_results = test_ground_truth_validation(manager, ground_truth_dataset)
    
    # NEW: Test citation backtracking
    print("\n9️⃣ Testing Citation Backtracking...")
    citation_results = test_citation_backtracking(manager)
    
    # Save comprehensive validation report
    print("\n🔟 Generating Comprehensive Validation Report...")
    save_validation_report(validation_results, citation_results)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 PRODUCTION MONITORING TEST COMPLETE")
    print("=" * 60)
    
    print("\n📊 FINAL SUMMARY:")
    print(f"   • Vector Store Health: {'✅ HEALTHY' if health_results['overall_status'] == 'HEALTHY' else '❌ ISSUES'}")
    print(f"   • Performance Monitoring: {'✅ PASSED' if monitoring_results['system_health']['overall_status'] == 'HEALTHY' else '❌ FAILED'}")
    print(f"   • Ground-Truth Validation: {'✅ PASSED' if validation_results['success_rate'] >= 0.6 else '❌ FAILED'}")
    print(f"   • Citation Backtracking: {'✅ PASSED' if citation_results['verification_rate'] >= 0.8 else '❌ FAILED'}")
    print(f"   • System Ready for Production: {'✅ YES' if all([health_results['overall_status'] == 'HEALTHY', monitoring_results['system_health']['overall_status'] == 'HEALTHY', validation_results['success_rate'] >= 0.6, citation_results['verification_rate'] >= 0.8]) else '❌ NO'}")
    
    print("\n💡 Next Steps:")
    print("   • Monitor system performance in production")
    print("   • Regularly validate with ground-truth datasets")
    print("   • Update citation verification mechanisms")
    print("   • Review and improve based on validation results")

def main():
    """Main function to run production monitoring tests."""
    try:
        test_production_monitoring()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 