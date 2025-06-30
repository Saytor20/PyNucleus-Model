"""
Load Balancer Configuration for PyNucleus Horizontal Scaling

Provides configuration for nginx, HAProxy, and cloud load balancers
to distribute traffic across multiple PyNucleus instances.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class InstanceConfig:
    """Configuration for a single PyNucleus instance"""
    host: str
    port: int
    weight: int = 1
    backup: bool = False
    max_fails: int = 3
    fail_timeout: str = "30s"

@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    endpoint: str = "/health"
    interval: str = "30s"
    timeout: str = "10s"
    retries: int = 3
    expected_status: int = 200

class LoadBalancerConfig:
    """Load balancer configuration generator"""
    
    def __init__(self, instances: List[InstanceConfig], health_check: Optional[HealthCheckConfig] = None):
        self.instances = instances
        self.health_check = health_check or HealthCheckConfig()
    
    def generate_nginx_config(self, upstream_name: str = "pynucleus_backend") -> str:
        """Generate nginx configuration for load balancing"""
        
        # Upstream configuration
        upstream_block = f"upstream {upstream_name} {{\n"
        upstream_block += "    least_conn;  # Use least connections algorithm\n"
        
        for instance in self.instances:
            backup_flag = " backup" if instance.backup else ""
            upstream_block += f"    server {instance.host}:{instance.port} weight={instance.weight} max_fails={instance.max_fails} fail_timeout={instance.fail_timeout}{backup_flag};\n"
        
        upstream_block += "}\n\n"
        
        # Server configuration
        server_config = """
server {
    listen 80;
    server_name localhost;
    
    # Enable gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;
    
    # Security headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    
    # Health check endpoint
    location /health {
        access_log off;
        proxy_pass http://""" + upstream_name + """;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Health check specific timeouts
        proxy_connect_timeout """ + self.health_check.timeout + """;
        proxy_send_timeout """ + self.health_check.timeout + """;
        proxy_read_timeout """ + self.health_check.timeout + """;
    }
    
    # Main application
    location / {
        proxy_pass http://""" + upstream_name + """;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;  # Allow longer for AI processing
        
        # Buffer settings for large responses
        proxy_buffering on;
        proxy_buffer_size 128k;
        proxy_buffers 4 256k;
        proxy_busy_buffers_size 256k;
        
        # Enable keep-alive
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
    
    # WebSocket support (if needed for real-time features)
    location /ws {
        proxy_pass http://""" + upstream_name + """;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # Static files (if served by nginx)
    location /static/ {
        alias /app/static/;
        expires 1d;
        add_header Cache-Control "public, no-transform";
    }
    
    # API documentation
    location /docs {
        proxy_pass http://""" + upstream_name + """;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
        """
        
        return upstream_block + server_config
    
    def generate_haproxy_config(self) -> str:
        """Generate HAProxy configuration for load balancing"""
        
        config = """
global
    daemon
    maxconn 4096
    log stdout local0
    
defaults
    mode http
    timeout connect 5s
    timeout client 300s
    timeout server 300s
    option httplog
    option dontlognull
    option redispatch
    retries 3
    
frontend pynucleus_frontend
    bind *:80
    default_backend pynucleus_servers
    
    # Security headers
    http-response set-header X-Content-Type-Options nosniff
    http-response set-header X-Frame-Options DENY
    http-response set-header X-XSS-Protection "1; mode=block"
    
    # Health check bypass
    acl health_check path_beg /health
    use_backend pynucleus_health if health_check

backend pynucleus_servers
    balance leastconn
    option httpchk GET """ + self.health_check.endpoint + """
    
"""
        
        # Add server instances
        for i, instance in enumerate(self.instances):
            backup_flag = " backup" if instance.backup else ""
            config += f"    server pynucleus{i+1} {instance.host}:{instance.port} weight {instance.weight} check inter {self.health_check.interval} rise {self.health_check.retries} fall {self.health_check.retries}{backup_flag}\n"
        
        config += """
backend pynucleus_health
    option httpchk GET """ + self.health_check.endpoint + """
"""
        
        # Add health check servers
        for i, instance in enumerate(self.instances):
            config += f"    server health{i+1} {instance.host}:{instance.port} check inter {self.health_check.interval}\n"
        
        return config
    
    def generate_docker_compose_scaling(self, base_service_name: str = "pynucleus-api") -> Dict[str, Any]:
        """Generate Docker Compose configuration for horizontal scaling"""
        
        services = {}
        
        # Load balancer service
        services["nginx-lb"] = {
            "image": "nginx:alpine",
            "ports": ["80:80", "443:443"],
            "volumes": [
                "./nginx.conf:/etc/nginx/nginx.conf:ro",
                "./ssl:/etc/nginx/ssl:ro"
            ],
            "depends_on": [f"{base_service_name}-{i+1}" for i in range(len(self.instances))],
            "restart": "unless-stopped",
            "networks": ["pynucleus-network"]
        }
        
        # Redis service for caching
        services["redis"] = {
            "image": "redis:7-alpine",
            "ports": ["6379:6379"],
            "command": "redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru",
            "volumes": ["redis-data:/data"],
            "restart": "unless-stopped",
            "networks": ["pynucleus-network"],
            "healthcheck": {
                "test": ["CMD", "redis-cli", "ping"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
        }
        
        # Application instances
        for i, instance in enumerate(self.instances):
            service_name = f"{base_service_name}-{i+1}"
            services[service_name] = {
                "build": {
                    "context": "..",
                    "dockerfile": "docker/Dockerfile.api"
                },
                "environment": {
                    "PYTHONPATH": "/app/src",
                    "PYNUCLEUS_API_KEY": "production",
                    "REDIS_HOST": "redis",
                    "REDIS_PORT": "6379",
                    "WORKER_ID": str(i+1),
                    "GUNICORN_WORKERS": "2",
                    "GUNICORN_TIMEOUT": "300"
                },
                "depends_on": {
                    "redis": {"condition": "service_healthy"}
                },
                "volumes": [
                    "../data:/app/data:ro",
                    "../logs:/app/logs"
                ],
                "networks": ["pynucleus-network"],
                "restart": "unless-stopped",
                "healthcheck": {
                    "test": ["CMD", "curl", "-f", f"http://localhost:{instance.port}/health"],
                    "interval": "30s",
                    "timeout": "10s",
                    "retries": 3
                },
                "deploy": {
                    "resources": {
                        "limits": {
                            "memory": "2G",
                            "cpus": "1.0"
                        },
                        "reservations": {
                            "memory": "512M",
                            "cpus": "0.25"
                        }
                    }
                }
            }
        
        # Complete Docker Compose structure
        compose_config = {
            "version": "3.8",
            "services": services,
            "networks": {
                "pynucleus-network": {
                    "driver": "bridge"
                }
            },
            "volumes": {
                "redis-data": {}
            }
        }
        
        return compose_config
    
    def generate_kubernetes_manifest(self, namespace: str = "pynucleus") -> Dict[str, Any]:
        """Generate Kubernetes deployment manifest"""
        
        # Deployment
        deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "pynucleus-api",
                "namespace": namespace,
                "labels": {
                    "app": "pynucleus-api",
                    "version": "v1"
                }
            },
            "spec": {
                "replicas": len(self.instances),
                "selector": {
                    "matchLabels": {
                        "app": "pynucleus-api"
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "pynucleus-api"
                        }
                    },
                    "spec": {
                        "containers": [{
                            "name": "pynucleus-api",
                            "image": "pynucleus:latest",
                            "ports": [{"containerPort": 5000}],
                            "env": [
                                {"name": "REDIS_HOST", "value": "redis-service"},
                                {"name": "REDIS_PORT", "value": "6379"},
                                {"name": "PYTHONPATH", "value": "/app/src"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "512Mi",
                                    "cpu": "250m"
                                },
                                "limits": {
                                    "memory": "2Gi", 
                                    "cpu": "1000m"
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 5000
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/health", 
                                    "port": 5000
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 10
                            }
                        }]
                    }
                }
            }
        }
        
        # Service
        service = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": "pynucleus-service",
                "namespace": namespace
            },
            "spec": {
                "selector": {
                    "app": "pynucleus-api"
                },
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 5000
                }],
                "type": "LoadBalancer"
            }
        }
        
        # HorizontalPodAutoscaler
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "pynucleus-hpa",
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "pynucleus-api"
                },
                "minReplicas": 1,
                "maxReplicas": 10,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 70
                            }
                        }
                    },
                    {
                        "type": "Resource", 
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ]
            }
        }
        
        return {
            "deployment": deployment,
            "service": service,
            "hpa": hpa
        }
    
    def save_configs(self, output_dir: Path):
        """Save all load balancer configurations to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Nginx config
        nginx_config = self.generate_nginx_config()
        (output_dir / "nginx.conf").write_text(nginx_config)
        
        # HAProxy config
        haproxy_config = self.generate_haproxy_config()
        (output_dir / "haproxy.cfg").write_text(haproxy_config)
        
        # Docker Compose config
        docker_compose = self.generate_docker_compose_scaling()
        (output_dir / "docker-compose.scale.yml").write_text(
            json.dumps(docker_compose, indent=2)
        )
        
        # Kubernetes manifests
        k8s_manifests = self.generate_kubernetes_manifest()
        for name, manifest in k8s_manifests.items():
            (output_dir / f"k8s-{name}.yaml").write_text(
                json.dumps(manifest, indent=2)
            ) 