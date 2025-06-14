#!/usr/bin/env python3
"""
Production-Ready Embedding Monitoring System

Provides comprehensive monitoring, benchmarking, and update protocols
for scaled document retrieval and pipeline implementation.
"""

import sys
import os
from pathlib import Path
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

# Import from project
try:
    from pynucleus.rag.vector_store import FAISSDBManager
    from pynucleus.rag.config import REPORTS_DIR
except ImportError:
    REPORTS_DIR = "data/04_models/chunk_reports"

class EmbeddingMonitor:
    """Production-ready embedding monitoring and benchmarking system."""
    
    def __init__(self, 
                 vector_store_manager: 'FAISSDBManager' = None,
                 monitoring_dir: str = None,
                 alert_thresholds: Dict[str, float] = None):
        """Initialize embedding monitor.
        
        Args:
            vector_store_manager: FAISS vector store manager
            monitoring_dir: Directory for monitoring data
            alert_thresholds: Performance alert thresholds
        """
        self.vector_store = vector_store_manager
        self.monitoring_dir = Path(monitoring_dir or REPORTS_DIR)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Default alert thresholds for production systems
        self.alert_thresholds = alert_thresholds or {
            'min_recall': 0.7,           # 70% minimum recall
            'max_response_time': 2.0,    # 2 seconds max response
            'min_similarity_score': 0.3, # 30% minimum similarity
            'max_drift_percentage': 15.0, # 15% max drift
            'min_coverage': 0.5          # 50% document coverage
        }
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup production logging."""
        logger = logging.getLogger('embedding_monitor')
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.monitoring_dir / f"embedding_monitor_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def run_comprehensive_benchmark(self, 
                                  custom_queries: List[str] = None,
                                  k_values: List[int] = None) -> Dict[str, Any]:
        """Run comprehensive benchmark for production readiness assessment.
        
        Args:
            custom_queries: Custom test queries (domain-specific)
            k_values: Different k values to test (default: [1, 3, 5, 10])
            
        Returns:
            Comprehensive benchmark results
        """
        if not self.vector_store:
            raise ValueError("Vector store manager required for benchmarking")
            
        k_values = k_values or [1, 3, 5, 10]
        
        # Default production-ready test queries for chemical engineering domain
        if not custom_queries:
            custom_queries = [
                # Core domain queries
                "modular chemical plant design principles",
                "process intensification techniques",
                "distillation column optimization methods",
                "reactor conversion efficiency improvement",
                "heat exchanger performance analysis",
                "separation process economics",
                "chemical process safety protocols",
                "industrial automation systems",
                "sustainable manufacturing practices",
                "supply chain optimization strategies",
                
                # Technical queries
                "mass transfer coefficient calculation",
                "thermodynamic equilibrium modeling",
                "catalyst deactivation mechanisms",
                "process control system design",
                "equipment sizing methodology",
                
                # Business/Economic queries
                "capital cost estimation methods",
                "operating expense optimization",
                "return on investment analysis",
                "risk assessment frameworks",
                "regulatory compliance requirements"
            ]
        
        benchmark_start = time.time()
        self.logger.info(f"Starting comprehensive benchmark with {len(custom_queries)} queries")
        
        results = {
            'benchmark_metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_queries': len(custom_queries),
                'k_values_tested': k_values,
                'document_count': len(self.vector_store.documents) if self.vector_store.documents else 0,
                'benchmark_duration': 0
            },
            'performance_by_k': {},
            'query_analysis': {},
            'system_performance': {},
            'alerts': [],
            'recommendations': []
        }
        
        # Test different k values
        for k in k_values:
            self.logger.info(f"Testing retrieval with k={k}")
            k_results = self._benchmark_k_value(custom_queries, k)
            results['performance_by_k'][f'k_{k}'] = k_results
            
            # Check for alerts
            self._check_performance_alerts(k_results, k, results['alerts'])
        
        # Analyze query patterns
        results['query_analysis'] = self._analyze_query_patterns(custom_queries)
        
        # System performance analysis
        results['system_performance'] = self._analyze_system_performance()
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Calculate total benchmark time
        results['benchmark_metadata']['benchmark_duration'] = time.time() - benchmark_start
        
        # Save results
        self._save_benchmark_results(results)
        
        # Log summary
        self._log_benchmark_summary(results)
        
        return results
    
    def _benchmark_k_value(self, queries: List[str], k: int) -> Dict[str, Any]:
        """Benchmark performance for specific k value."""
        scores = []
        response_times = []
        coverage_docs = set()
        failed_queries = 0
        
        for query in queries:
            try:
                start_time = time.time()
                results = self.vector_store.search(query, k=k)
                response_time = time.time() - start_time
                response_times.append(response_time)
                
                if results:
                    query_scores = [score for _, score in results]
                    scores.extend(query_scores)
                    
                    # Track document coverage
                    for doc, _ in results:
                        doc_id = doc.metadata.get('source', f'doc_{hash(doc.page_content[:100])}')
                        coverage_docs.add(doc_id)
                else:
                    failed_queries += 1
                    
            except Exception as e:
                self.logger.error(f"Query failed: {query[:50]}... Error: {e}")
                failed_queries += 1
        
        # Calculate metrics
        total_docs = len(self.vector_store.documents) if self.vector_store.documents else 1
        
        return {
            'avg_similarity_score': np.mean(scores) if scores and NUMPY_AVAILABLE else (sum(scores) / len(scores) if scores else 0),
            'min_similarity_score': min(scores) if scores else 0,
            'max_similarity_score': max(scores) if scores else 0,
            'similarity_std': np.std(scores) if scores and NUMPY_AVAILABLE else 0,
            'avg_response_time': np.mean(response_times) if response_times and NUMPY_AVAILABLE else (sum(response_times) / len(response_times) if response_times else 0),
            'max_response_time': max(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'document_coverage': len(coverage_docs) / total_docs,
            'failed_queries': failed_queries,
            'success_rate': (len(queries) - failed_queries) / len(queries) if queries else 0,
            'total_results': len(scores)
        }
    
    def _check_performance_alerts(self, k_results: Dict[str, Any], k: int, alerts: List[str]):
        """Check performance against thresholds and generate alerts."""
        
        # Response time alert
        if k_results['avg_response_time'] > self.alert_thresholds['max_response_time']:
            alerts.append(f"HIGH_RESPONSE_TIME: Average response time {k_results['avg_response_time']:.3f}s exceeds threshold {self.alert_thresholds['max_response_time']}s for k={k}")
        
        # Similarity score alert
        if k_results['avg_similarity_score'] < self.alert_thresholds['min_similarity_score']:
            alerts.append(f"LOW_SIMILARITY: Average similarity {k_results['avg_similarity_score']:.3f} below threshold {self.alert_thresholds['min_similarity_score']} for k={k}")
        
        # Coverage alert
        if k_results['document_coverage'] < self.alert_thresholds['min_coverage']:
            alerts.append(f"LOW_COVERAGE: Document coverage {k_results['document_coverage']:.2%} below threshold {self.alert_thresholds['min_coverage']:.2%} for k={k}")
        
        # Success rate alert
        if k_results['success_rate'] < 0.95:  # 95% success rate threshold
            alerts.append(f"LOW_SUCCESS_RATE: Query success rate {k_results['success_rate']:.2%} below 95% for k={k}")
    
    def _analyze_query_patterns(self, queries: List[str]) -> Dict[str, Any]:
        """Analyze query patterns for insights."""
        
        # Query length analysis
        query_lengths = [len(query.split()) for query in queries]
        
        # Query type classification (simple heuristic)
        technical_queries = [q for q in queries if any(term in q.lower() for term in ['calculation', 'coefficient', 'modeling', 'analysis', 'design'])]
        business_queries = [q for q in queries if any(term in q.lower() for term in ['cost', 'investment', 'economic', 'roi', 'expense'])]
        process_queries = [q for q in queries if any(term in q.lower() for term in ['process', 'reactor', 'distillation', 'separation', 'heat'])]
        
        return {
            'total_queries': len(queries),
            'avg_query_length': np.mean(query_lengths) if NUMPY_AVAILABLE else sum(query_lengths) / len(query_lengths),
            'min_query_length': min(query_lengths),
            'max_query_length': max(query_lengths),
            'query_types': {
                'technical': len(technical_queries),
                'business': len(business_queries),
                'process': len(process_queries),
                'other': len(queries) - len(technical_queries) - len(business_queries) - len(process_queries)
            }
        }
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze overall system performance."""
        
        # Check if historical data exists
        metrics_file = self.monitoring_dir / "performance_metrics.jsonl"
        historical_data = []
        
        if metrics_file.exists():
            try:
                with open(metrics_file, 'r') as f:
                    for line in f:
                        try:
                            historical_data.append(json.loads(line.strip()))
                        except:
                            continue
            except Exception as e:
                self.logger.warning(f"Could not read historical data: {e}")
        
        system_perf = {
            'historical_measurements': len(historical_data),
            'monitoring_active': metrics_file.exists(),
            'last_measurement': None,
            'trend_analysis': None
        }
        
        if historical_data:
            latest = historical_data[-1]
            system_perf['last_measurement'] = {
                'timestamp': latest.get('timestamp'),
                'recall': latest.get('recall_at_k'),
                'avg_score': latest.get('average_similarity_score'),
                'response_time': latest.get('average_response_time')
            }
            
            # Simple trend analysis (last 5 vs previous 5)
            if len(historical_data) >= 10:
                recent_scores = [d.get('average_similarity_score', 0) for d in historical_data[-5:]]
                older_scores = [d.get('average_similarity_score', 0) for d in historical_data[-10:-5]]
                
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                trend = 'IMPROVING' if recent_avg > older_avg else 'DECLINING' if recent_avg < older_avg else 'STABLE'
                system_perf['trend_analysis'] = {
                    'trend': trend,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'change_percentage': ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
                }
        
        return system_perf
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on benchmark results."""
        recommendations = []
        
        # Check if any k=3 performance (most common)
        k3_results = results['performance_by_k'].get('k_3', {})
        
        if k3_results:
            # Response time recommendations
            if k3_results['avg_response_time'] > 1.0:
                recommendations.append("Consider optimizing FAISS index or upgrading hardware for faster retrieval")
            
            # Similarity score recommendations
            if k3_results['avg_similarity_score'] < 0.4:
                recommendations.append("Consider retraining embeddings or using a more powerful embedding model")
            
            # Coverage recommendations
            if k3_results['document_coverage'] < 0.6:
                recommendations.append("Add more diverse documents or improve document chunking strategy")
            
            # Success rate recommendations
            if k3_results['success_rate'] < 0.9:
                recommendations.append("Investigate query processing pipeline for robustness issues")
        
        # Alert-based recommendations
        if results['alerts']:
            recommendations.append("Address performance alerts before production deployment")
        
        # System recommendations
        system_perf = results['system_performance']
        if not system_perf['monitoring_active']:
            recommendations.append("Enable continuous monitoring for production systems")
        
        if system_perf['trend_analysis'] and system_perf['trend_analysis']['trend'] == 'DECLINING':
            recommendations.append("Performance declining - investigate embedding drift or data quality issues")
        
        # Document count recommendations
        doc_count = results['benchmark_metadata']['document_count']
        if doc_count < 100:
            recommendations.append("Consider adding more documents for better retrieval performance")
        elif doc_count > 10000:
            recommendations.append("Monitor index size and consider distributed retrieval for very large document sets")
        
        return recommendations
    
    def _save_benchmark_results(self, results: Dict[str, Any]):
        """Save benchmark results for historical analysis."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        results_file = self.monitoring_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary for trend analysis
        summary = {
            'timestamp': results['benchmark_metadata']['timestamp'],
            'document_count': results['benchmark_metadata']['document_count'],
            'total_queries': results['benchmark_metadata']['total_queries'],
            'benchmark_duration': results['benchmark_metadata']['benchmark_duration'],
            'k3_avg_similarity': results['performance_by_k'].get('k_3', {}).get('avg_similarity_score', 0),
            'k3_avg_response_time': results['performance_by_k'].get('k_3', {}).get('avg_response_time', 0),
            'k3_coverage': results['performance_by_k'].get('k_3', {}).get('document_coverage', 0),
            'alert_count': len(results['alerts']),
            'recommendation_count': len(results['recommendations'])
        }
        
        summary_file = self.monitoring_dir / "benchmark_summary.jsonl"
        with open(summary_file, 'a') as f:
            f.write(json.dumps(summary) + '\n')
        
        self.logger.info(f"Benchmark results saved to {results_file}")
    
    def _log_benchmark_summary(self, results: Dict[str, Any]):
        """Log benchmark summary."""
        self.logger.info("=== BENCHMARK SUMMARY ===")
        self.logger.info(f"Documents: {results['benchmark_metadata']['document_count']}")
        self.logger.info(f"Queries: {results['benchmark_metadata']['total_queries']}")
        self.logger.info(f"Duration: {results['benchmark_metadata']['benchmark_duration']:.2f}s")
        
        k3_results = results['performance_by_k'].get('k_3', {})
        if k3_results:
            self.logger.info(f"K=3 Performance:")
            self.logger.info(f"  Avg Similarity: {k3_results['avg_similarity_score']:.4f}")
            self.logger.info(f"  Avg Response Time: {k3_results['avg_response_time']:.3f}s")
            self.logger.info(f"  Document Coverage: {k3_results['document_coverage']:.2%}")
            self.logger.info(f"  Success Rate: {k3_results['success_rate']:.2%}")
        
        if results['alerts']:
            self.logger.warning(f"ALERTS: {len(results['alerts'])}")
            for alert in results['alerts']:
                self.logger.warning(f"  {alert}")
        
        if results['recommendations']:
            self.logger.info(f"RECOMMENDATIONS: {len(results['recommendations'])}")
            for rec in results['recommendations']:
                self.logger.info(f"  {rec}")

    def monitor_production_health(self) -> Dict[str, Any]:
        """Continuous monitoring for production systems."""
        if not self.vector_store:
            raise ValueError("Vector store manager required for monitoring")
        
        # Run health check
        health_status = self.vector_store.health_check()
        
        # Run drift monitoring
        drift_status = self.vector_store.monitor_embedding_drift()
        
        # Quick performance check
        quick_benchmark = self.vector_store.benchmark_embedding_quality(
            sample_queries=[
                "chemical process optimization",
                "modular plant design",
                "industrial efficiency"
            ],
            k=3
        )
        
        monitoring_result = {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'drift_status': drift_status,
            'quick_benchmark': quick_benchmark,
            'overall_status': 'HEALTHY'
        }
        
        # Determine overall status
        if health_status['overall_status'] == 'CRITICAL':
            monitoring_result['overall_status'] = 'CRITICAL'
        elif health_status['overall_status'] == 'DEGRADED':
            monitoring_result['overall_status'] = 'DEGRADED'
        elif drift_status['drift_indicators'].get('drift_status') == 'SIGNIFICANT':
            monitoring_result['overall_status'] = 'ATTENTION_NEEDED'
        
        # Save monitoring result
        monitoring_file = self.monitoring_dir / "production_monitoring.jsonl"
        with open(monitoring_file, 'a') as f:
            f.write(json.dumps(monitoring_result) + '\n')
        
        # Log critical issues
        if monitoring_result['overall_status'] != 'HEALTHY':
            self.logger.warning(f"Production health issue detected: {monitoring_result['overall_status']}")
        
        return monitoring_result

def main():
    """Example usage of embedding monitor."""
    print("🔍 Embedding Monitor - Production Ready")
    print("Monitor initialized. Use with FAISSDBManager for full functionality.")

if __name__ == "__main__":
    main() 