"""
PyNucleus Deployment Monitoring

Production-ready monitoring and alerting for PyNucleus deployments.
Tracks performance, health, and scaling metrics across multiple instances.
"""

import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import psutil
import requests
from requests.exceptions import RequestException

from ..utils.logger import logger
from .scaling_manager import InstanceMetrics, CacheManager

@dataclass
class InstanceHealth:
    """Health status of a single instance"""
    instance_id: str
    host: str
    port: int
    healthy: bool
    response_time: float
    last_check: datetime
    error_message: str = ""
    consecutive_failures: int = 0

@dataclass
class SystemAlert:
    """System alert definition"""
    alert_id: str
    severity: str  # critical, warning, info
    message: str
    metric_name: str
    threshold: float
    current_value: float
    timestamp: datetime
    resolved: bool = False

class DeploymentMonitor:
    """Comprehensive deployment monitoring system"""
    
    def __init__(self, instances: List[Dict[str, Any]], cache_manager: Optional[CacheManager] = None):
        self.instances = instances
        self.cache_manager = cache_manager
        self._instance_health: Dict[str, InstanceHealth] = {}
        self._alerts: List[SystemAlert] = []
        self._metrics_history: List[Dict[str, Any]] = []
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Alert thresholds
        self._thresholds = {
            'response_time_warning': 2.0,
            'response_time_critical': 5.0,
            'cpu_warning': 75.0,
            'cpu_critical': 90.0,
            'memory_warning': 80.0,
            'memory_critical': 95.0,
            'cache_hit_rate_low': 50.0,
            'error_rate_warning': 5.0,
            'error_rate_critical': 10.0
        }
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[SystemAlert], None]] = []
        
        logger.info(f"Deployment monitor initialized for {len(instances)} instances")
    
    def add_alert_callback(self, callback: Callable[[SystemAlert], None]):
        """Add callback function to be called when alerts are triggered"""
        self._alert_callbacks.append(callback)
    
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info(f"Monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Check instance health
                self._check_all_instances()
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    def _check_all_instances(self):
        """Check health of all instances"""
        for instance in self.instances:
            instance_id = f"{instance['host']}:{instance['port']}"
            health = self._check_instance_health(instance)
            
            with self._lock:
                self._instance_health[instance_id] = health
    
    def _check_instance_health(self, instance: Dict[str, Any]) -> InstanceHealth:
        """Check health of a single instance"""
        host = instance['host']
        port = instance['port']
        instance_id = f"{host}:{port}"
        
        start_time = time.time()
        
        try:
            # Health check request
            response = requests.get(
                f"http://{host}:{port}/health",
                timeout=10
            )
            
            response_time = (time.time() - start_time) * 1000  # ms
            
            if response.status_code == 200:
                health_data = response.json()
                healthy = health_data.get('status') == 'healthy'
            else:
                healthy = False
                
            return InstanceHealth(
                instance_id=instance_id,
                host=host,
                port=port,
                healthy=healthy,
                response_time=response_time,
                last_check=datetime.now(),
                consecutive_failures=0 if healthy else self._instance_health.get(instance_id, InstanceHealth("", "", 0, False, 0, datetime.now())).consecutive_failures + 1
            )
            
        except RequestException as e:
            response_time = (time.time() - start_time) * 1000
            prev_health = self._instance_health.get(instance_id)
            consecutive_failures = (prev_health.consecutive_failures + 1) if prev_health else 1
            
            return InstanceHealth(
                instance_id=instance_id,
                host=host,
                port=port,
                healthy=False,
                response_time=response_time,
                last_check=datetime.now(),
                error_message=str(e),
                consecutive_failures=consecutive_failures
            )
    
    def _collect_system_metrics(self):
        """Collect overall system metrics"""
        try:
            # System-wide metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Instance-specific metrics
            healthy_instances = sum(1 for h in self._instance_health.values() if h.healthy)
            total_instances = len(self._instance_health)
            avg_response_time = sum(h.response_time for h in self._instance_health.values()) / max(1, total_instances)
            
            # Cache metrics
            cache_stats = self.cache_manager.get_cache_stats() if self.cache_manager else {}
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": disk.percent,
                    "disk_free_gb": disk.free / (1024**3)
                },
                "instances": {
                    "total": total_instances,
                    "healthy": healthy_instances,
                    "unhealthy": total_instances - healthy_instances,
                    "health_rate": (healthy_instances / max(1, total_instances)) * 100,
                    "avg_response_time": avg_response_time
                },
                "cache": cache_stats
            }
            
            with self._lock:
                self._metrics_history.append(metrics)
                
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions and trigger alerts"""
        if not self._metrics_history:
            return
        
        latest_metrics = self._metrics_history[-1]
        current_time = datetime.now()
        
        # System alerts
        system = latest_metrics.get("system", {})
        self._check_threshold_alert(
            "cpu_usage", system.get("cpu_percent", 0),
            self._thresholds["cpu_warning"], self._thresholds["cpu_critical"]
        )
        
        self._check_threshold_alert(
            "memory_usage", system.get("memory_percent", 0),
            self._thresholds["memory_warning"], self._thresholds["memory_critical"]
        )
        
        # Instance health alerts
        instances = latest_metrics.get("instances", {})
        health_rate = instances.get("health_rate", 100)
        if health_rate < 100:
            self._create_alert(
                f"instance_health_{current_time.timestamp()}",
                "critical" if health_rate < 50 else "warning",
                f"Instance health degraded: {health_rate:.1f}% healthy",
                "instance_health_rate",
                100,
                health_rate
            )
        
        # Response time alerts
        avg_response_time = instances.get("avg_response_time", 0) / 1000  # Convert to seconds
        self._check_threshold_alert(
            "response_time", avg_response_time,
            self._thresholds["response_time_warning"], self._thresholds["response_time_critical"]
        )
        
        # Cache performance alerts
        cache = latest_metrics.get("cache", {})
        hit_rate = cache.get("hit_rate_percent", 100)
        if hit_rate < self._thresholds["cache_hit_rate_low"]:
            self._create_alert(
                f"cache_hit_rate_{current_time.timestamp()}",
                "warning",
                f"Low cache hit rate: {hit_rate:.1f}%",
                "cache_hit_rate",
                self._thresholds["cache_hit_rate_low"],
                hit_rate
            )
    
    def _check_threshold_alert(self, metric_name: str, current_value: float, warning_threshold: float, critical_threshold: float):
        """Check if metric exceeds thresholds and create alerts"""
        if current_value >= critical_threshold:
            self._create_alert(
                f"{metric_name}_critical_{datetime.now().timestamp()}",
                "critical",
                f"Critical {metric_name}: {current_value:.1f}%",
                metric_name,
                critical_threshold,
                current_value
            )
        elif current_value >= warning_threshold:
            self._create_alert(
                f"{metric_name}_warning_{datetime.now().timestamp()}",
                "warning", 
                f"High {metric_name}: {current_value:.1f}%",
                metric_name,
                warning_threshold,
                current_value
            )
    
    def _create_alert(self, alert_id: str, severity: str, message: str, metric_name: str, threshold: float, current_value: float):
        """Create and process a new alert"""
        # Check if similar alert already exists and is recent
        recent_alerts = [a for a in self._alerts if a.alert_id.startswith(f"{metric_name}_") and 
                        (datetime.now() - a.timestamp).seconds < 300 and not a.resolved]
        
        if recent_alerts:
            return  # Don't spam similar alerts
        
        alert = SystemAlert(
            alert_id=alert_id,
            severity=severity,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now()
        )
        
        with self._lock:
            self._alerts.append(alert)
        
        # Trigger alert callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
        
        logger.warning(f"ALERT [{severity.upper()}]: {message}")
    
    def _cleanup_old_data(self):
        """Clean up old metrics and alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        with self._lock:
            # Keep only last 24 hours of metrics (assuming 30s intervals = ~2880 entries)
            if len(self._metrics_history) > 2880:
                self._metrics_history = self._metrics_history[-2880:]
            
            # Keep only recent alerts
            self._alerts = [a for a in self._alerts if (datetime.now() - a.timestamp) < timedelta(hours=24)]
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data"""
        with self._lock:
            current_time = datetime.now()
            
            # Recent metrics (last hour)
            recent_metrics = [
                m for m in self._metrics_history 
                if (current_time - datetime.fromisoformat(m["timestamp"])) < timedelta(hours=1)
            ]
            
            # Active alerts
            active_alerts = [a for a in self._alerts if not a.resolved]
            
            # Instance status summary
            instance_summary = {}
            for instance_id, health in self._instance_health.items():
                instance_summary[instance_id] = {
                    "healthy": health.healthy,
                    "response_time": health.response_time,
                    "last_check": health.last_check.isoformat(),
                    "consecutive_failures": health.consecutive_failures,
                    "error_message": health.error_message
                }
            
            # Performance trends
            if recent_metrics:
                cpu_trend = [m["system"]["cpu_percent"] for m in recent_metrics]
                memory_trend = [m["system"]["memory_percent"] for m in recent_metrics]
                response_time_trend = [m["instances"]["avg_response_time"] for m in recent_metrics]
                
                trends = {
                    "cpu": {
                        "current": cpu_trend[-1] if cpu_trend else 0,
                        "average": sum(cpu_trend) / len(cpu_trend) if cpu_trend else 0,
                        "trend": "stable"  # Could implement trend calculation
                    },
                    "memory": {
                        "current": memory_trend[-1] if memory_trend else 0,
                        "average": sum(memory_trend) / len(memory_trend) if memory_trend else 0,
                        "trend": "stable"
                    },
                    "response_time": {
                        "current": response_time_trend[-1] if response_time_trend else 0,
                        "average": sum(response_time_trend) / len(response_time_trend) if response_time_trend else 0,
                        "trend": "stable"
                    }
                }
            else:
                trends = {}
            
            return {
                "status": "active" if self._monitoring_active else "inactive",
                "timestamp": current_time.isoformat(),
                "instances": instance_summary,
                "alerts": {
                    "active_count": len(active_alerts),
                    "critical_count": len([a for a in active_alerts if a.severity == "critical"]),
                    "warning_count": len([a for a in active_alerts if a.severity == "warning"]),
                    "recent_alerts": [asdict(a) for a in active_alerts[-10:]]  # Last 10 alerts
                },
                "performance": trends,
                "cache_stats": self.cache_manager.get_cache_stats() if self.cache_manager else {},
                "metrics_available": len(self._metrics_history),
                "monitoring_duration": len(self._metrics_history) * 30  # Assuming 30s intervals
            }
    
    def export_metrics(self, filepath: Path, hours: int = 24):
        """Export metrics to JSON file"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            filtered_metrics = [
                m for m in self._metrics_history 
                if datetime.fromisoformat(m["timestamp"]) > cutoff_time
            ]
            
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "export_period_hours": hours,
                "metrics_count": len(filtered_metrics),
                "metrics": filtered_metrics,
                "alerts": [asdict(a) for a in self._alerts if (datetime.now() - a.timestamp) < timedelta(hours=hours)],
                "instance_health": {k: asdict(v) for k, v in self._instance_health.items()},
                "thresholds": self._thresholds
            }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark an alert as resolved"""
        with self._lock:
            for alert in self._alerts:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
        return False 