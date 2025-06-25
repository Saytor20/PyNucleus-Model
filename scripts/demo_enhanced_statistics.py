#!/usr/bin/env python3
"""
Enhanced Statistics Demo Script
===============================

This script demonstrates the improved statistics, metrics, and evaluation capabilities
of PyNucleus without changing core functions or design - only enhancing what we measure
and how we view the system.

Run with: python scripts/demo_enhanced_statistics.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def demo_enhanced_metrics():
    """Demonstrate enhanced metrics collection."""
    print("🚀 ENHANCED METRICS SYSTEM DEMO")
    print("=" * 50)
    
    try:
        from pynucleus.metrics import Metrics, get_summary, get_trends
        
        # Simulate some queries for demonstration
        print("📊 Simulating query metrics...")
        
        # Record some sample queries
        sample_queries = [
            ("What is distillation?", 1.2, 0.5, 0.7, 3, 150, 0.85, "chemical_engineering"),
            ("How do modular plants work?", 0.8, 0.3, 0.5, 2, 120, 0.90, "process_design"),
            ("Explain heat exchangers", 1.5, 0.6, 0.9, 4, 200, 0.75, "heat_transfer"),
            ("Safety in chemical plants", 1.1, 0.4, 0.7, 2, 180, 0.80, "safety"),
            ("Reactor conversion efficiency", 0.9, 0.3, 0.6, 3, 160, 0.88, "chemical_engineering")
        ]
        
        for i, (question, response_time, retrieval_time, generation_time, sources, length, confidence, domain) in enumerate(sample_queries):
            print(f"  Recording query {i+1}: '{question[:30]}...'")
            
            Metrics.record_query(
                question=question,
                response_time=response_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                sources_count=sources,
                answer_length=length,
                confidence_score=confidence,
                domain=domain,
                success=True
            )
            time.sleep(0.1)  # Small delay for realistic timestamps
        
        print(f"✅ Recorded {len(sample_queries)} sample queries\n")
        
        # Get performance summary
        print("📈 PERFORMANCE SUMMARY:")
        print("-" * 30)
        summary = get_summary()
        
        if summary.get("query_performance"):
            perf = summary["query_performance"]
            print(f"📊 Query Performance (Last Hour):")
            print(f"   • Total Queries: {perf.get('total_queries', 0)}")
            print(f"   • Success Rate: {perf.get('success_rate', 0):.1%}")
            print(f"   • Avg Response Time: {perf.get('avg_response_time', 0):.3f}s")
            print(f"   • Avg Confidence: {perf.get('avg_confidence_score', 0):.3f}")
            print(f"   • Performance Trend: {perf.get('performance_trend', 'unknown')}")
        
        if summary.get("quality_metrics"):
            quality = summary["quality_metrics"]
            print(f"\n🎯 Quality Metrics:")
            if quality.get("domain_performance"):
                print(f"   • Domain Performance:")
                for domain, stats in quality["domain_performance"].items():
                    print(f"     - {domain}: {stats.get('success_rate', 0):.1%} ({stats.get('count', 0)} queries)")
        
        if summary.get("system_health"):
            health = summary["system_health"]
            print(f"\n🏥 System Health:")
            print(f"   • Queries/minute: {health.get('queries_per_minute', 0)}")
            print(f"   • Response consistency: {health.get('response_time_consistency', 0):.3f}")
        
        print("\n" + "=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error in metrics demo: {e}")
        return False


def demo_enhanced_evaluation():
    """Demonstrate enhanced evaluation system."""
    print("\n🔍 ENHANCED EVALUATION SYSTEM DEMO")
    print("=" * 50)
    
    try:
        from pynucleus.eval.golden_eval import EnhancedEvaluator
        
        # Check if golden dataset exists
        golden_path = Path("data/validation/golden_dataset.csv")
        if not golden_path.exists():
            print("⚠️ Golden dataset not found - creating sample data...")
            return demo_mock_evaluation()
        
        print("📄 Running enhanced evaluation with golden dataset...")
        
        evaluator = EnhancedEvaluator()
        result = evaluator.run_comprehensive_eval(
            threshold=0.7,
            sample_size=5,  # Small sample for demo
            save_results=False
        )
        
        if result.get("analysis"):
            analysis = result["analysis"]
            
            print(f"\n📊 EVALUATION RESULTS:")
            print(f"-" * 30)
            
            overall = analysis.get("overall_metrics", {})
            print(f"Overall Performance:")
            print(f"   • Success Rate: {overall.get('success_rate', 0):.1%}")
            print(f"   • Avg Response Time: {overall.get('avg_response_time', 0):.3f}s")
            print(f"   • Avg Confidence: {overall.get('avg_confidence_score', 0):.3f}")
            print(f"   • Avg Answer Length: {overall.get('avg_answer_length', 0):.0f} chars")
            
            # Quality distribution
            quality_dist = analysis.get("quality_distribution", {})
            if quality_dist:
                print(f"\n📈 Quality Distribution:")
                confidence_scores = quality_dist.get("confidence_scores", {})
                for level, count in confidence_scores.items():
                    print(f"   • {level.title()}: {count} questions")
            
            # Recommendations
            recommendations = analysis.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                    print(f"   {i}. {rec}")
        
        print(f"\n✅ Enhanced evaluation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in evaluation demo: {e}")
        return False


def demo_mock_evaluation():
    """Demo with mock evaluation data."""
    print("📊 Using mock evaluation data for demonstration...")
    
    mock_analysis = {
        "overall_metrics": {
            "success_rate": 0.75,
            "avg_response_time": 1.2,
            "avg_confidence_score": 0.82,
            "total_questions": 10,
            "successful_questions": 8
        },
        "quality_distribution": {
            "confidence_scores": {
                "excellent": 4,
                "good": 3,
                "fair": 2,
                "poor": 1
            }
        },
        "domain_analysis": {
            "chemical_engineering": {"success_rate": 0.8, "questions": 5},
            "process_design": {"success_rate": 0.7, "questions": 3},
            "safety": {"success_rate": 0.5, "questions": 2}
        },
        "recommendations": [
            "⚡ Consider optimizing retrieval speed - average response time is above 1 second",
            "📚 Consider expanding document collection for safety domain",
            "🔍 Increase retrieval k parameter for better source coverage"
        ]
    }
    
    print(f"\n📊 MOCK EVALUATION RESULTS:")
    print(f"-" * 30)
    
    overall = mock_analysis["overall_metrics"]
    print(f"Overall Performance:")
    print(f"   • Success Rate: {overall['success_rate']:.1%}")
    print(f"   • Avg Response Time: {overall['avg_response_time']:.3f}s")
    print(f"   • Avg Confidence: {overall['avg_confidence_score']:.3f}")
    print(f"   • Questions: {overall['successful_questions']}/{overall['total_questions']}")
    
    print(f"\n🏷️ Domain Performance:")
    for domain, stats in mock_analysis["domain_analysis"].items():
        print(f"   • {domain}: {stats['success_rate']:.1%} ({stats['questions']} questions)")
    
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(mock_analysis["recommendations"], 1):
        print(f"   {i}. {rec}")
    
    return True


def demo_api_endpoints():
    """Demonstrate new API endpoints."""
    print("\n🌐 ENHANCED API ENDPOINTS DEMO")
    print("=" * 50)
    
    endpoints = {
        "/system_statistics": "Enhanced system statistics with performance metrics",
        "/enhanced_evaluation": "Comprehensive evaluation with quality analysis",
        "/metrics_export": "Export detailed metrics to JSON files"
    }
    
    print("📡 New API endpoints available:")
    for endpoint, description in endpoints.items():
        print(f"   • {endpoint}")
        print(f"     {description}")
    
    print(f"\n🔧 Usage Examples:")
    print(f"   • GET /system_statistics - Get enhanced system statistics")
    print(f"   • GET /enhanced_evaluation?threshold=0.8&sample_size=10")
    print(f"   • GET /metrics_export?export=file - Export to file")
    print(f"   • GET /enhanced_evaluation?format=json - JSON format")
    
    return True


def demo_web_interface_improvements():
    """Demonstrate web interface improvements."""
    print("\n🖥️ WEB INTERFACE IMPROVEMENTS DEMO")
    print("=" * 50)
    
    improvements = [
        "✅ Enhanced Statistics button for comprehensive metrics",
        "✅ Enhanced Evaluation button for quality analysis", 
        "✅ Export Metrics button for data export",
        "✅ Better formatting of performance data",
        "✅ Historical trends visualization",
        "✅ Quality insights display",
        "✅ Domain-specific performance breakdown"
    ]
    
    print("🎨 Web Interface Enhancements:")
    for improvement in improvements:
        print(f"   {improvement}")
    
    print(f"\n🚀 Access the enhanced interface:")
    print(f"   • Main interface: http://localhost:5001/")
    print(f"   • Developer dashboard: http://localhost:5001/dev")
    print(f"   • Try the new 'Enhanced Evaluation' and 'Export Metrics' buttons!")
    
    return True


def main():
    """Run the complete enhanced statistics demo."""
    print("🎯 PyNucleus Enhanced Statistics & Evaluation Demo")
    print("=" * 60)
    print("This demo showcases improvements to:")
    print("  1. 📊 What we measure (detailed metrics)")
    print("  2. 🔍 How we evaluate (comprehensive analysis)")
    print("  3. 👀 How we view the system (enhanced output)")
    print("=" * 60)
    
    results = []
    
    # Run all demos
    results.append(demo_enhanced_metrics())
    results.append(demo_enhanced_evaluation())
    results.append(demo_api_endpoints())
    results.append(demo_web_interface_improvements())
    
    # Summary
    successful_demos = sum(results)
    total_demos = len(results)
    
    print("\n" + "=" * 60)
    print("📈 DEMO SUMMARY")
    print("=" * 60)
    print(f"✅ Successful demos: {successful_demos}/{total_demos}")
    
    if successful_demos == total_demos:
        print("🎉 All enhanced statistics features demonstrated successfully!")
        print("\n🚀 Key Improvements Achieved:")
        print("   • 📊 Enhanced metrics collection with detailed tracking")
        print("   • 🔍 Comprehensive evaluation with quality analysis")
        print("   • 🌐 New API endpoints for better data access")
        print("   • 🖥️ Improved web interface with richer displays")
        print("   • 📈 Historical trending and performance insights")
        print("   • 💡 Actionable recommendations based on analysis")
    else:
        print("⚠️ Some demos encountered issues - check error messages above")
    
    print("\n🎯 Next Steps:")
    print("   • Start the web server: python run_web_app.py --direct")
    print("   • Visit the enhanced interfaces to see improvements")
    print("   • Try the new evaluation and metrics export features")
    print("   • Generate some queries to see real-time metrics")


if __name__ == "__main__":
    main() 