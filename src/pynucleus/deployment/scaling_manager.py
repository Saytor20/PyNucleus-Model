"""
PyNucleus Scaling Manager

Intelligent auto-scaling service that monitors system performance and scales
API instances based on load, response times, and resource utilization.

Features:
- Real-time monitoring of API instances
- Auto-scaling based on configurable thresholds
- Redis-based distributed caching
- Docker container orchestration
- Health checks and failover
- Performance metrics collection
"""

import os
import sys
import time
import json
import redis
import docker
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scaling_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class InstanceMetrics:
    """Metrics for a single API instance"""
    instance_id: str
    cpu_usage: float
    memory_usage: float
    response_time_avg: float
    requests_per_second: float
    error_rate: float
    timestamp: datetime
    health_status: str

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior"""
    min_instances: int = 2
    max_instances: int = 8
    target_cpu_usage: float = 70.0
    target_response_time: float = 2.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 40.0
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 600  # seconds
    health_check_interval: int = 30  # seconds
    metrics_window: int = 300  # seconds

class CacheManager:
    """Redis-based distributed caching manager"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.logger = logging.getLogger(f"{__name__}.CacheManager")
        
    def get_cached_response(self, query_hash: str) -> Optional[Dict]:
        """Get cached RAG response"""
        try:
            cached = self.redis_client.get(f"rag_response:{query_hash}")
            if cached:
                return json.loads(cached)
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
        return None
    
    def cache_response(self, query_hash: str, response: Dict, ttl: int = 3600):
        """Cache RAG response with TTL"""
        try:
            self.redis_client.setex(
                f"rag_response:{query_hash}",
                ttl,
                json.dumps(response)
            )
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
    
    def get_instance_metrics(self, instance_id: str) -> Optional[InstanceMetrics]:
        """Get cached instance metrics"""
        try:
            metrics_data = self.redis_client.get(f"metrics:{instance_id}")
            if metrics_data:
                data = json.loads(metrics_data)
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                return InstanceMetrics(**data)
        except Exception as e:
            self.logger.error(f"Failed to get metrics for {instance_id}: {e}")
        return None
    
    def store_instance_metrics(self, metrics: InstanceMetrics):
        """Store instance metrics in Redis"""
        try:
            data = asdict(metrics)
            data['timestamp'] = metrics.timestamp.isoformat()
            self.redis_client.setex(
                f"metrics:{metrics.instance_id}",
                300,  # 5 minutes TTL
                json.dumps(data)
            )
        except Exception as e:
            self.logger.error(f"Failed to store metrics for {metrics.instance_id}: {e}")
    
    def get_all_instance_metrics(self) -> List[InstanceMetrics]:
        """Get metrics for all instances"""
        metrics = []
        try:
            for key in self.redis_client.scan_iter(match="metrics:*"):
                instance_id = key.split(":", 1)[1]
                instance_metrics = self.get_instance_metrics(instance_id)
                if instance_metrics:
                    metrics.append(instance_metrics)
        except Exception as e:
            self.logger.error(f"Failed to get all metrics: {e}")
        return metrics

class DockerManager:
    """Docker container management for scaling"""
    
    def __init__(self):
        self.client = docker.from_env()
        self.logger = logging.getLogger(f"{__name__}.DockerManager")
        
    def get_api_containers(self) -> List:
        """Get all PyNucleus API containers"""
        try:
            return self.client.containers.list(
                filters={"label": "com.docker.compose.service=api"}
            )
        except Exception as e:
            self.logger.error(f"Failed to get API containers: {e}")
            return []
    
    def scale_service(self, service_name: str, replica_count: int):
        """Scale a Docker Compose service"""
        try:
            import subprocess
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.scale.yml",
                "up", "--scale", f"{service_name}={replica_count}", "-d"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Scaled {service_name} to {replica_count} replicas")
                return True
            else:
                self.logger.error(f"Failed to scale {service_name}: {result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"Scale operation failed: {e}")
            return False
    
    def get_container_stats(self, container_id: str) -> Dict:
        """Get container resource statistics"""
        try:
            container = self.client.containers.get(container_id)
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                       stats['precpu_stats']['cpu_usage']['total_usage']
            system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                          stats['precpu_stats']['system_cpu_usage']
            cpu_percent = (cpu_delta / system_delta) * 100.0
            
            # Calculate memory percentage
            memory_usage = stats['memory_stats']['usage']
            memory_limit = stats['memory_stats']['limit']
            memory_percent = (memory_usage / memory_limit) * 100.0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'network_rx_mb': stats['networks']['eth0']['rx_bytes'] / (1024 * 1024),
                'network_tx_mb': stats['networks']['eth0']['tx_bytes'] / (1024 * 1024)
            }
        except Exception as e:
            self.logger.error(f"Failed to get stats for container {container_id}: {e}")
            return {}

class ScalingManager:
    """Main scaling manager orchestrating auto-scaling decisions"""
    
    def __init__(self, config: ScalingConfig = None):
        self.config = config or ScalingConfig()
        self.cache_manager = CacheManager(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
        self.docker_manager = DockerManager()
        self.logger = logging.getLogger(f"{__name__}.ScalingManager")
        
        self.last_scale_up = datetime.min
        self.last_scale_down = datetime.min
        self.current_instances = self.config.min_instances
        
    async def collect_metrics(self):
        """Collect metrics from all API instances"""
        containers = self.docker_manager.get_api_containers()
        
        for container in containers:
            try:
                # Get container stats
                stats = self.docker_manager.get_container_stats(container.id)
                
                # Get health status
                health_status = "healthy" if container.status == "running" else "unhealthy"
                
                # Create metrics object
                metrics = InstanceMetrics(
                    instance_id=container.name,
                    cpu_usage=stats.get('cpu_percent', 0),
                    memory_usage=stats.get('memory_percent', 0),
                    response_time_avg=await self._get_response_time(container.name),
                    requests_per_second=await self._get_requests_per_second(container.name),
                    error_rate=await self._get_error_rate(container.name),
                    timestamp=datetime.now(),
                    health_status=health_status
                )
                
                # Store in cache
                self.cache_manager.store_instance_metrics(metrics)
                
            except Exception as e:
                self.logger.error(f"Failed to collect metrics for {container.name}: {e}")
    
    async def _get_response_time(self, instance_id: str) -> float:
        """Get average response time for instance"""
        # This would integrate with your monitoring system
        # For now, return a mock value
        import random
        return random.uniform(0.5, 3.0)
    
    async def _get_requests_per_second(self, instance_id: str) -> float:
        """Get requests per second for instance"""
        # This would integrate with your monitoring system
        import random
        return random.uniform(1, 50)
    
    async def _get_error_rate(self, instance_id: str) -> float:
        """Get error rate for instance"""
        # This would integrate with your monitoring system
        import random
        return random.uniform(0, 5)
    
    def should_scale_up(self, metrics: List[InstanceMetrics]) -> bool:
        """Determine if we should scale up"""
        if not metrics:
            return False
        
        # Check cooldown
        if datetime.now() - self.last_scale_up < timedelta(seconds=self.config.scale_up_cooldown):
            return False
        
        # Check if we're at max capacity
        if self.current_instances >= self.config.max_instances:
            return False
        
        # Check metrics thresholds
        avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
        avg_response_time = sum(m.response_time_avg for m in metrics) / len(metrics)
        
        return (avg_cpu > self.config.scale_up_threshold or 
                avg_response_time > self.config.target_response_time * 1.5)
    
    def should_scale_down(self, metrics: List[InstanceMetrics]) -> bool:
        """Determine if we should scale down"""
        if not metrics:
            return False
        
        # Check cooldown
        if datetime.now() - self.last_scale_down < timedelta(seconds=self.config.scale_down_cooldown):
            return False
        
        # Check if we're at min capacity
        if self.current_instances <= self.config.min_instances:
            return False
        
        # Check metrics thresholds
        avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
        avg_response_time = sum(m.response_time_avg for m in metrics) / len(metrics)
        
        return (avg_cpu < self.config.scale_down_threshold and 
                avg_response_time < self.config.target_response_time)
    
    async def make_scaling_decision(self):
        """Make auto-scaling decisions based on current metrics"""
        metrics = self.cache_manager.get_all_instance_metrics()
        
        if not metrics:
            self.logger.warning("No metrics available for scaling decision")
            return
        
        # Filter recent metrics only
        recent_metrics = [
            m for m in metrics 
            if datetime.now() - m.timestamp < timedelta(seconds=self.config.metrics_window)
        ]
        
        if not recent_metrics:
            self.logger.warning("No recent metrics available")
            return
        
        # Log current state
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.response_time_avg for m in recent_metrics) / len(recent_metrics)
        
        self.logger.info(
            f"Current state: {len(recent_metrics)} instances, "
            f"avg CPU: {avg_cpu:.1f}%, avg response time: {avg_response_time:.2f}s"
        )
        
        # Make scaling decisions
        if self.should_scale_up(recent_metrics):
            new_count = min(self.current_instances + 1, self.config.max_instances)
            self.logger.info(f"Scaling up from {self.current_instances} to {new_count} instances")
            
            if self.docker_manager.scale_service("api", new_count):
                self.current_instances = new_count
                self.last_scale_up = datetime.now()
                
        elif self.should_scale_down(recent_metrics):
            new_count = max(self.current_instances - 1, self.config.min_instances)
            self.logger.info(f"Scaling down from {self.current_instances} to {new_count} instances")
            
            if self.docker_manager.scale_service("api", new_count):
                self.current_instances = new_count
                self.last_scale_down = datetime.now()
    
    async def health_check_loop(self):
        """Continuous health checking and scaling loop"""
        while True:
            try:
                await self.collect_metrics()
                await self.make_scaling_decision()
                await asyncio.sleep(self.config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(30)  # Wait before retrying
    
    def start(self):
        """Start the scaling manager"""
        self.logger.info("Starting PyNucleus Scaling Manager")
        self.logger.info(f"Configuration: {self.config}")
        
        # Start the main loop
        asyncio.run(self.health_check_loop())

def main():
    """Main entry point"""
    # Load configuration
    config = ScalingConfig()
    
    # Override with environment variables if provided
    config.min_instances = int(os.getenv('SCALING_MIN_INSTANCES', config.min_instances))
    config.max_instances = int(os.getenv('SCALING_MAX_INSTANCES', config.max_instances))
    config.target_cpu_usage = float(os.getenv('SCALING_TARGET_CPU', config.target_cpu_usage))
    config.target_response_time = float(os.getenv('SCALING_TARGET_RESPONSE_TIME', config.target_response_time))
    
    # Create and start manager
    manager = ScalingManager(config)
    manager.start()

if __name__ == "__main__":
    main()