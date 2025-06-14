#!/usr/bin/env python3
"""
Test Production-Ready Embedding Monitoring System

Demonstrates comprehensive monitoring, benchmarking, and alerting
for scaled document retrieval systems.
"""

import sys
import os
from pathlib import Path
import json
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from pynucleus.rag.vector_store import FAISSDBManager
from pynucleus.rag.embedding_monitor import EmbeddingMonitor

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
    
    print(f"   📊 Overall Status: {monitoring_results['overall_status']}")
    print(f"   📊 Health Status: {monitoring_results['health_status']['overall_status']}")
    print(f"   📊 Drift Status: {monitoring_results['drift_status']['drift_indicators'].get('status', 'NORMAL')}")
    print("   ✅ Production monitoring completed")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PRODUCTION MONITORING TEST SUMMARY")
    print("=" * 60)
    
    print(f"✅ Vector Store: {len(docs)} documents indexed")
    print(f"✅ Evaluation: {metrics['recall_at_k']:.1f}% recall, {metrics['average_response_time']:.3f}s avg time")
    print(f"✅ Benchmark: {len(production_queries)} queries tested across {len(benchmark_results['performance_by_k'])} k-values")
    print(f"✅ Monitoring: {len(benchmark_results['alerts'])} alerts, {len(benchmark_results['recommendations'])} recommendations")
    print(f"✅ Health: {health_results['overall_status']} status")
    print(f"✅ Production Ready: {monitoring_results['overall_status']}")
    
    # Production readiness assessment
    production_ready = (
        health_results['overall_status'] == 'HEALTHY' and
        monitoring_results['overall_status'] in ['HEALTHY', 'ATTENTION_NEEDED'] and
        metrics['recall_at_k'] >= 50 and  # At least 50% recall
        metrics['average_response_time'] < 2.0  # Under 2 seconds
    )
    
    print(f"\n🚀 PRODUCTION READINESS: {'✅ READY' if production_ready else '⚠️ NEEDS ATTENTION'}")
    
    if not production_ready:
        print("   💡 Address performance issues before production deployment")
    else:
        print("   🎯 System meets production readiness criteria")
        print("   📈 Monitoring and alerting systems operational")
        print("   🔄 Ready for scaled document retrieval implementation")
    
    return {
        'production_ready': production_ready,
        'metrics': metrics,
        'benchmark_results': benchmark_results,
        'health_status': health_results,
        'monitoring_status': monitoring_results
    }

def main():
    """Run production monitoring test."""
    try:
        results = test_production_monitoring()
        
        # Save test results
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        results_file = f"data/04_models/chunk_reports/production_test_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        json_results = {
            'production_ready': results['production_ready'],
            'test_timestamp': timestamp,
            'summary': {
                'recall': results['metrics']['recall_at_k'],
                'avg_response_time': results['metrics']['average_response_time'],
                'health_status': results['health_status']['overall_status'],
                'monitoring_status': results['monitoring_status']['overall_status']
            }
        }
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Test results saved to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 