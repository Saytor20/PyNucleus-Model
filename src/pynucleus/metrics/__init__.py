"""
PyNucleus Enhanced Metrics System
=================================

This module provides comprehensive metrics collection and analysis for PyNucleus.
Focuses on improved measurement, tracking, and reporting without changing core functions.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque


@dataclass
class QueryMetrics:
    """Detailed metrics for individual queries."""
    timestamp: str
    question: str
    response_time: float
    retrieval_time: float
    generation_time: float
    sources_count: int
    answer_length: int
    confidence_score: float
    domain: str = ""
    success: bool = True
    error_type: str = ""


@dataclass
class SystemSnapshot:
    """Complete system state snapshot."""
    timestamp: str
    memory_usage_mb: float
    cpu_usage_percent: float
    active_connections: int
    vector_db_size_mb: float
    total_documents: int
    queries_last_hour: int
    avg_response_time_1h: float
    success_rate_1h: float


class EnhancedMetrics:
    """Enhanced metrics collection and analysis system."""
    
    def __init__(self, max_history_size: int = 10000):
        self.max_history_size = max_history_size
        self._lock = threading.Lock()
        
        # Historical data storage
        self.query_history: deque = deque(maxlen=max_history_size)
        self.system_snapshots: deque = deque(maxlen=1440)  # 24 hours of minute snapshots
        
        # Real-time counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(float)
        self.gauges = defaultdict(float)
        
        # Performance windows
        self.last_hour_queries: deque = deque(maxlen=3600)  # Track recent queries
        self.response_times: deque = deque(maxlen=1000)     # Recent response times
        
        # Quality metrics
        self.quality_scores = defaultdict(list)
        self.error_patterns = defaultdict(int)
        
        # Start background monitoring
        self._start_monitoring()
    
    def record_query(self, question: str, response_time: float, 
                    retrieval_time: float = 0, generation_time: float = 0,
                    sources_count: int = 0, answer_length: int = 0,
                    confidence_score: float = 0.0, domain: str = "",
                    success: bool = True, error_type: str = ""):
        """Record detailed query metrics."""
        with self._lock:
            timestamp = datetime.now().isoformat()
            
            metric = QueryMetrics(
                timestamp=timestamp,
                question=question[:100],  # Truncate for privacy
                response_time=response_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                sources_count=sources_count,
                answer_length=answer_length,
                confidence_score=confidence_score,
                domain=domain,
                success=success,
                error_type=error_type
            )
            
            self.query_history.append(metric)
            self.last_hour_queries.append(time.time())
            self.response_times.append(response_time)
            
            # Update counters
            self.counters['total_queries'] += 1
            if success:
                self.counters['successful_queries'] += 1
            else:
                self.counters['failed_queries'] += 1
                self.error_patterns[error_type] += 1
            
            # Update quality tracking
            if domain:
                self.quality_scores[domain].append(confidence_score)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            now = time.time()
            
            # Filter recent queries (last hour)
            recent_queries = [q for q in self.query_history 
                            if (now - time.mktime(time.strptime(q.timestamp[:19], '%Y-%m-%dT%H:%M:%S'))) < 3600]
            
            if not recent_queries:
                return self._empty_summary()
            
            # Calculate metrics
            total_queries = len(recent_queries)
            successful = sum(1 for q in recent_queries if q.success)
            avg_response_time = sum(q.response_time for q in recent_queries) / total_queries
            avg_retrieval_time = sum(q.retrieval_time for q in recent_queries) / total_queries
            avg_generation_time = sum(q.generation_time for q in recent_queries) / total_queries
            avg_sources = sum(q.sources_count for q in recent_queries) / total_queries
            avg_confidence = sum(q.confidence_score for q in recent_queries) / total_queries
            
            # Performance trends
            if len(self.response_times) >= 10:
                recent_10 = list(self.response_times)[-10:]
                trend = "improving" if recent_10[-1] < sum(recent_10[:-1]) / 9 else "stable"
            else:
                trend = "stable"
            
            # Domain performance
            domain_stats = {}
            for query in recent_queries:
                if query.domain:
                    if query.domain not in domain_stats:
                        domain_stats[query.domain] = {"count": 0, "success": 0, "avg_confidence": 0}
                    domain_stats[query.domain]["count"] += 1
                    if query.success:
                        domain_stats[query.domain]["success"] += 1
                    domain_stats[query.domain]["avg_confidence"] += query.confidence_score
            
            # Finalize domain stats
            for domain in domain_stats:
                stats = domain_stats[domain]
                stats["success_rate"] = stats["success"] / stats["count"] if stats["count"] > 0 else 0
                stats["avg_confidence"] = stats["avg_confidence"] / stats["count"] if stats["count"] > 0 else 0
            
            return {
                "period": "last_hour",
                "query_performance": {
                    "total_queries": total_queries,
                    "successful_queries": successful,
                    "success_rate": successful / total_queries if total_queries > 0 else 0,
                    "avg_response_time": round(avg_response_time, 3),
                    "avg_retrieval_time": round(avg_retrieval_time, 3),
                    "avg_generation_time": round(avg_generation_time, 3),
                    "avg_sources_used": round(avg_sources, 1),
                    "avg_confidence_score": round(avg_confidence, 3),
                    "performance_trend": trend
                },
                "quality_metrics": {
                    "domain_performance": domain_stats,
                    "avg_answer_length": sum(q.answer_length for q in recent_queries) / total_queries if total_queries > 0 else 0,
                    "error_distribution": dict(self.error_patterns) if self.error_patterns else {}
                },
                "system_health": {
                    "queries_per_minute": round(total_queries / 60, 2),
                    "peak_response_time": max(q.response_time for q in recent_queries) if recent_queries else 0,
                    "fastest_response_time": min(q.response_time for q in recent_queries) if recent_queries else 0,
                    "response_time_consistency": self._calculate_consistency(recent_queries)
                }
            }
    
    def get_historical_trends(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical performance trends."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter historical data
            historical_queries = [
                q for q in self.query_history 
                if datetime.fromisoformat(q.timestamp[:19]) > cutoff_time
            ]
            
            if not historical_queries:
                return {"message": "No historical data available", "hours": hours}
            
            # Group by hour for trending
            hourly_stats = defaultdict(lambda: {"queries": 0, "successful": 0, "total_time": 0})
            
            for query in historical_queries:
                hour_key = query.timestamp[:13]  # YYYY-MM-DDTHH
                hourly_stats[hour_key]["queries"] += 1
                if query.success:
                    hourly_stats[hour_key]["successful"] += 1
                hourly_stats[hour_key]["total_time"] += query.response_time
            
            # Calculate hourly averages
            trending_data = []
            for hour, stats in sorted(hourly_stats.items()):
                trending_data.append({
                    "hour": hour,
                    "queries": stats["queries"],
                    "success_rate": stats["successful"] / stats["queries"] if stats["queries"] > 0 else 0,
                    "avg_response_time": stats["total_time"] / stats["queries"] if stats["queries"] > 0 else 0
                })
            
            return {
                "period_hours": hours,
                "total_data_points": len(historical_queries),
                "hourly_trends": trending_data,
                "overall_stats": {
                    "total_queries": len(historical_queries),
                    "overall_success_rate": sum(1 for q in historical_queries if q.success) / len(historical_queries),
                    "overall_avg_response_time": sum(q.response_time for q in historical_queries) / len(historical_queries)
                }
            }
    
    def _calculate_consistency(self, queries: List[QueryMetrics]) -> float:
        """Calculate response time consistency (lower is better)."""
        if len(queries) < 2:
            return 0.0
        
        times = [q.response_time for q in queries]
        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        coefficient_of_variation = (variance ** 0.5) / mean_time if mean_time > 0 else 0
        
        return round(coefficient_of_variation, 3)
    
    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty metrics summary."""
        return {
            "period": "last_hour",
            "query_performance": {
                "total_queries": 0,
                "message": "No queries in the last hour"
            },
            "quality_metrics": {},
            "system_health": {}
        }
    
    def _start_monitoring(self):
        """Start background system monitoring."""
        # This could be expanded to collect system snapshots periodically
        pass
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        with self._lock:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "query_history": [asdict(q) for q in self.query_history],
                "performance_summary": self.get_performance_summary(),
                "counters": dict(self.counters),
                "quality_scores": {k: v[-100:] for k, v in self.quality_scores.items()}  # Last 100 scores per domain
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)


# Global metrics instance
Metrics = EnhancedMetrics()


# Convenience functions for backward compatibility
def inc(counter_name: str, value: int = 1):
    """Increment a counter."""
    Metrics.counters[counter_name] += value


def start(timer_name: str):
    """Start a timer."""
    Metrics.timers[f"{timer_name}_start"] = time.time()


def stop(timer_name: str):
    """Stop a timer and return elapsed time."""
    start_time = Metrics.timers.get(f"{timer_name}_start", 0)
    if start_time:
        elapsed = time.time() - start_time
        Metrics.timers[timer_name] = elapsed
        return elapsed
    return 0


def gauge(gauge_name: str, value: float):
    """Set a gauge value."""
    Metrics.gauges[gauge_name] = value


def get_summary():
    """Get performance summary."""
    return Metrics.get_performance_summary()


def get_trends(hours: int = 24):
    """Get historical trends."""
    return Metrics.get_historical_trends(hours) 