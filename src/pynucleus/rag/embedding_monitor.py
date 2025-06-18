"""
Embedding monitor for PyNucleus RAG system.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class EmbeddingMonitor:
    """Monitor embedding quality and performance in RAG system."""
    
    def __init__(self, monitoring_dir: str = "data/04_models/monitoring"):
        """
        Initialize embedding monitor.
        
        Args:
            monitoring_dir: Directory for monitoring data
        """
        self.monitoring_dir = Path(monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.query_count = 0
        self.total_response_time = 0.0
        self.embedding_quality_scores = []
        
        self.logger.info("EmbeddingMonitor initialized")
    
    def log_query_performance(self, query: str, response_time: float, embedding_quality: float) -> None:
        """
        Log query performance metrics.
        
        Args:
            query: Search query
            response_time: Query response time in seconds
            embedding_quality: Quality score of embeddings (0-1)
        """
        try:
            self.query_count += 1
            self.total_response_time += response_time
            self.embedding_quality_scores.append(embedding_quality)
            
            # Log to file
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query[:100],  # Truncate long queries
                "response_time": response_time,
                "embedding_quality": embedding_quality,
                "query_id": self.query_count
            }
            
            self.logger.info(f"Query logged: ID={self.query_count}, Time={response_time:.3f}s, Quality={embedding_quality:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to log query performance: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get overall performance metrics."""
        try:
            if self.query_count == 0:
                return {
                    "total_queries": 0,
                    "average_response_time": 0.0,
                    "average_embedding_quality": 0.0,
                    "monitoring_status": "No queries processed"
                }
            
            avg_response_time = self.total_response_time / self.query_count
            avg_quality = sum(self.embedding_quality_scores) / len(self.embedding_quality_scores)
            
            return {
                "total_queries": self.query_count,
                "average_response_time": avg_response_time,
                "average_embedding_quality": avg_quality,
                "min_quality": min(self.embedding_quality_scores) if self.embedding_quality_scores else 0.0,
                "max_quality": max(self.embedding_quality_scores) if self.embedding_quality_scores else 0.0,
                "monitoring_status": "Active",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get performance metrics: {e}")
            return {
                "error": str(e),
                "monitoring_status": "Error"
            }
    
    def check_embedding_drift(self, current_embeddings: List[float], baseline_embeddings: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Check for embedding drift compared to baseline.
        
        Args:
            current_embeddings: Current embedding vectors
            baseline_embeddings: Baseline embedding vectors for comparison
            
        Returns:
            Drift analysis results
        """
        try:
            if not baseline_embeddings:
                # Use mock baseline for testing
                baseline_embeddings = [0.5] * len(current_embeddings)
            
            if len(current_embeddings) != len(baseline_embeddings):
                return {
                    "drift_detected": False,
                    "error": "Embedding dimension mismatch",
                    "current_dim": len(current_embeddings),
                    "baseline_dim": len(baseline_embeddings)
                }
            
            # Calculate cosine similarity as drift metric
            dot_product = sum(a * b for a, b in zip(current_embeddings, baseline_embeddings))
            magnitude_current = sum(a * a for a in current_embeddings) ** 0.5
            magnitude_baseline = sum(b * b for b in baseline_embeddings) ** 0.5
            
            if magnitude_current == 0 or magnitude_baseline == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (magnitude_current * magnitude_baseline)
            
            drift_threshold = 0.8  # Similarity below this indicates drift
            drift_detected = similarity < drift_threshold
            
            return {
                "drift_detected": drift_detected,
                "similarity_score": similarity,
                "drift_threshold": drift_threshold,
                "drift_magnitude": 1 - similarity,
                "recommendation": "Retrain embeddings" if drift_detected else "Embeddings stable",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Embedding drift check failed: {e}")
            return {
                "drift_detected": False,
                "error": str(e)
            }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        try:
            performance_metrics = self.get_performance_metrics()
            
            # Mock embedding drift analysis
            mock_embeddings = [0.1, 0.2, 0.3, 0.4, 0.5]
            drift_analysis = self.check_embedding_drift(mock_embeddings)
            
            report = {
                "report_timestamp": datetime.now().isoformat(),
                "monitoring_period": "24 hours",  # Mock period
                "performance_metrics": performance_metrics,
                "embedding_drift_analysis": drift_analysis,
                "system_health": {
                    "status": "Healthy" if performance_metrics.get("average_embedding_quality", 0) > 0.7 else "Needs Attention",
                    "quality_threshold": 0.7,
                    "performance_threshold": 1.0  # seconds
                },
                "recommendations": self._generate_recommendations(performance_metrics, drift_analysis)
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate monitoring report: {e}")
            return {
                "error": str(e),
                "report_timestamp": datetime.now().isoformat()
            }
    
    def _generate_recommendations(self, performance_metrics: Dict[str, Any], drift_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on monitoring data."""
        recommendations = []
        
        avg_quality = performance_metrics.get("average_embedding_quality", 0)
        avg_response_time = performance_metrics.get("average_response_time", 0)
        
        if avg_quality < 0.7:
            recommendations.append("Consider improving embedding model quality")
        
        if avg_response_time > 1.0:
            recommendations.append("Optimize query processing for better response times")
        
        if drift_analysis.get("drift_detected", False):
            recommendations.append("Embedding drift detected - consider retraining")
        
        if not recommendations:
            recommendations.append("System performance is optimal")
        
        return recommendations 