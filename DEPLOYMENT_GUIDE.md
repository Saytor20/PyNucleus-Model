# PyNucleus Scalable Deployment Guide

## Overview

This guide covers deploying PyNucleus in a production-ready, horizontally scalable configuration with:

- **Load Balancing**: Nginx load balancer distributing traffic across multiple API instances
- **Distributed Caching**: Redis for shared response caching across instances
- **Auto-Scaling**: Intelligent scaling manager monitoring and adjusting instance count
- **Health Monitoring**: Comprehensive health checks and metrics collection
- **Horizontal Scaling**: Support for 2-20 API instances based on load

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │────│   API Instance 1  │────│                 │
│    (Nginx)      │    │                   │    │                 │
│                 │    ├──────────────────┤    │   Redis Cache   │
│  Rate Limiting  │────│   API Instance 2  │────│                 │
│  Health Checks  │    │                   │    │  Shared State   │
│                 │    ├──────────────────┤    │                 │
│                 │────│   API Instance N  │────│                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Scaling        │              │
         └──────────────│  Manager        │──────────────┘
                        │                 │
                        │  Auto-scaling   │
                        │  Health Monitor │
                        │  Docker Control │
                        └─────────────────┘
```

## Quick Start

### Prerequisites

1. **Docker & Docker Compose**
   ```bash
   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **System Requirements**
   - Minimum: 8GB RAM, 4 CPU cores, 20GB disk
   - Recommended: 16GB RAM, 8 CPU cores, 50GB disk
   - Production: 32GB RAM, 16 CPU cores, 100GB SSD

### Basic Deployment

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd PyNucleus-Model
   chmod +x scripts/launch_scaled_deployment.sh
   ```

2. **Start Deployment**
   ```bash
   # Development with 3 instances
   ./scripts/launch_scaled_deployment.sh up --instances 3 --env development
   
   # Production with 5 instances
   ./scripts/launch_scaled_deployment.sh up --instances 5 --env production
   ```

3. **Verify Deployment**
   ```bash
   # Check status
   ./scripts/launch_scaled_deployment.sh status
   
   # Test API
   curl http://localhost/health
   
   # Run stress test
   python scripts/simple_stress_test.py --concurrent 20 --duration 120
   ```

## Configuration

### Environment Variables

#### Core Application
```bash
# Application
FLASK_ENV=production
FLASK_DEBUG=false
SECRET_KEY=your-secret-key-here
PYNUCLEUS_API_KEY=your-api-key

# Redis Cache
REDIS_URL=redis://redis:6379/0

# Instance Identification
PYNUCLEUS_INSTANCE_ID=api-1

# Gunicorn Workers
GUNICORN_WORKERS=4
GUNICORN_THREADS=2
```

#### Auto-Scaling
```bash
# Scaling Configuration
SCALING_MIN_INSTANCES=2
SCALING_MAX_INSTANCES=10
SCALING_TARGET_CPU=70.0
SCALING_TARGET_RESPONSE_TIME=2.0
SCALING_SCALE_UP_THRESHOLD=80.0
SCALING_SCALE_DOWN_THRESHOLD=40.0
SCALING_SCALE_UP_COOLDOWN=300
SCALING_SCALE_DOWN_COOLDOWN=600
```

### Docker Compose Configuration

The system uses `docker/docker-compose.scale.yml` with these services:

#### Redis Cache
```yaml
redis:
  image: redis:7-alpine
  command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
  ports:
    - "6379:6379"
```

#### Load Balancer
```yaml
load-balancer:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf:ro
```

#### API Instances
```yaml
api-1:
  build:
    context: ..
    dockerfile: docker/Dockerfile.api
  environment:
    - REDIS_URL=redis://redis:6379/0
    - PYNUCLEUS_INSTANCE_ID=api-1
```

## Scaling Operations

### Manual Scaling

```bash
# Scale to 8 instances
./scripts/launch_scaled_deployment.sh scale --instances 8

# Check scaling status
./scripts/launch_scaled_deployment.sh status
```

### Auto-Scaling

The scaling manager automatically:
- Monitors CPU usage, memory usage, and response times
- Scales up when thresholds exceed 80% CPU or 3s response time
- Scales down when below 40% CPU and 2s response time
- Respects cooldown periods to prevent thrashing

### Scaling Configuration

Edit scaling parameters in the scaling manager:

```python
config = ScalingConfig(
    min_instances=2,
    max_instances=8,
    target_cpu_usage=70.0,
    target_response_time=2.0,
    scale_up_threshold=80.0,
    scale_down_threshold=40.0,
    scale_up_cooldown=300,  # 5 minutes
    scale_down_cooldown=600  # 10 minutes
)
```

## Monitoring & Observability

### Health Checks

- **Application Health**: `GET /health`
- **Load Balancer Health**: Nginx upstream health checks
- **Cache Health**: Redis ping checks

### Performance Monitoring

The system includes comprehensive monitoring capabilities:

```bash
# Real-time monitoring dashboard
python src/pynucleus/diagnostics/dashboard.py

# Export performance metrics
python -c "
from src.pynucleus.deployment.monitoring import DeploymentMonitor
monitor = DeploymentMonitor(instances=[])
monitor.export_metrics('metrics_export.json')
"
```

### Metrics Collection

Available metrics endpoints:
- **Prometheus Metrics**: `GET /metrics`
- **Health Status**: `GET /health`
- **Cache Statistics**: Redis INFO command

Key metrics monitored:
- Response times (average, P95, P99)
- Error rates
- Cache hit/miss ratios
- CPU and memory utilization
- Request throughput (RPS)

## Testing & Validation

### Integration Testing

Run comprehensive integration tests:

```bash
# Full integration test suite
python scripts/integration_test.py

# Expected output: All 7 tests should pass
# ✅ Redis Connectivity: PASS
# ✅ Cache Integration: PASS
# ✅ Scaling Manager: PASS
# ✅ Flask App Factory: PASS
# ✅ Docker Configuration: PASS
# ✅ Flask API Endpoints: PASS
# ✅ Stress Test Infrastructure: PASS
```

### Stress Testing

```bash
# Simple stress test
python scripts/simple_stress_test.py --concurrent 20 --duration 60

# Comprehensive scaling validation
python scripts/stress_test_suite.py --validation

# Custom stress test
python scripts/stress_test_suite.py --users 50 --requests 100 --ramp-up 30
```

### Performance Benchmarks

Expected performance targets:
- **Response Time**: < 2s average, < 5s P95
- **Throughput**: > 100 RPS per instance
- **Error Rate**: < 1% under normal load
- **Cache Hit Rate**: > 70% for common queries
- **Availability**: > 99.9% uptime
- **Instance Health**: Docker container health checks

### Metrics

#### Prometheus Metrics
```bash
# Cache metrics
curl http://localhost/metrics
```

#### Manual Monitoring
```bash
# Real-time monitoring dashboard
./scripts/launch_scaled_deployment.sh monitor

# Service logs
./scripts/launch_scaled_deployment.sh logs api-1
./scripts/launch_scaled_deployment.sh logs scaling-manager
```

### Performance Testing

#### Stress Testing
```bash
# Basic stress test
python scripts/simple_stress_test.py \
  --url http://localhost \
  --concurrent 50 \
  --duration 300

# Advanced stress test
python scripts/stress_test.py \
  --url http://localhost \
  --concurrent 100 \
  --duration 600 \
  --output test_results/
```

#### Load Testing Results Interpretation

- **Success Rate**: Should be >99% under normal load
- **Response Time**: P95 should be <3s, P99 <5s
- **Cache Hit Rate**: Should be >30% for repeated queries
- **Instance Distribution**: Should be relatively balanced across instances

## Cache Configuration

### Redis Setup

The Redis cache is configured with:
- **Memory Policy**: `allkeys-lru` (evict least recently used)
- **Max Memory**: 512MB (adjustable)
- **Persistence**: AOF enabled for durability
- **TTL Strategy**: Dynamic based on confidence and complexity

### Cache TTL Strategy

```python
# High confidence responses: 2 hours
if confidence > 0.8:
    cache_ttl = 7200

# Complex questions: 30 minutes  
elif is_complex_question:
    cache_ttl = 1800

# Default: 1 hour
else:
    cache_ttl = 3600
```

### Cache Performance

Monitor cache effectiveness via:
```bash
# Cache statistics in health check
curl http://localhost/health | jq '.components.cache_stats'

# Prometheus metrics
curl http://localhost/metrics | grep cache_hit_rate
```

## Security Considerations

### Network Security
- Use HTTPS in production (SSL termination at load balancer)
- Restrict Redis access to application network only
- Configure firewall rules for exposed ports

### API Security
```bash
# Set strong API key
export PYNUCLEUS_API_KEY=$(openssl rand -base64 32)

# Use secure secret key
export SECRET_KEY=$(openssl rand -base64 32)
```

### Resource Limits
Each service has resource limits defined:
```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Production Deployment

### Environment Setup

1. **Production Environment Variables**
   ```bash
   # Create production environment file
   cat > .env.production << EOF
   FLASK_ENV=production
   FLASK_DEBUG=false
   SECRET_KEY=$(openssl rand -base64 32)
   PYNUCLEUS_API_KEY=$(openssl rand -base64 32)
   REDIS_URL=redis://redis:6379/0
   GUNICORN_WORKERS=4
   GUNICORN_THREADS=2
   EOF
   ```

2. **Start Production Deployment**
   ```bash
   ./scripts/launch_scaled_deployment.sh up \
     --instances 5 \
     --env production \
     --build
   ```

### SSL/HTTPS Setup

1. **Obtain SSL Certificates**
   ```bash
   # Using Let's Encrypt
   certbot certonly --standalone -d your-domain.com
   ```

2. **Configure Nginx SSL**
   Update `docker/nginx.conf`:
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /etc/ssl/certs/cert.pem;
       ssl_certificate_key /etc/ssl/private/key.pem;
       # ... rest of configuration
   }
   ```

### Database Persistence

For production, ensure data persistence:
```yaml
volumes:
  redis_data:
    driver: local
  app_data:
    driver: local
```

## Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check service status
./scripts/launch_scaled_deployment.sh status

# Check logs
./scripts/launch_scaled_deployment.sh logs redis
./scripts/launch_scaled_deployment.sh logs api-1

# Restart problematic service
docker-compose -f docker/docker-compose.scale.yml restart api-1
```

#### 2. High Response Times
```bash
# Check system resources
docker stats

# Check cache hit rate
curl http://localhost/health | jq '.components.cache_stats.hit_rate'

# Scale up if needed
./scripts/launch_scaled_deployment.sh scale --instances 8
```

#### 3. Cache Issues
```bash
# Test Redis connection
docker-compose -f docker/docker-compose.scale.yml exec redis redis-cli ping

# Check Redis memory usage
docker-compose -f docker/docker-compose.scale.yml exec redis redis-cli info memory

# Clear cache if needed
docker-compose -f docker/docker-compose.scale.yml exec redis redis-cli flushall
```

#### 4. Load Balancer Issues
```bash
# Check nginx configuration
docker-compose -f docker/docker-compose.scale.yml exec load-balancer nginx -t

# Check upstream status
curl -H "Host: localhost" http://localhost/health
```

### Performance Optimization

1. **Tune Gunicorn Workers**
   ```bash
   # Formula: (2 × CPU cores) + 1
   export GUNICORN_WORKERS=9  # For 4 cores
   ```

2. **Optimize Redis Memory**
   ```bash
   # Increase Redis memory limit
   docker-compose -f docker/docker-compose.scale.yml exec redis redis-cli config set maxmemory 1gb
   ```

3. **Adjust Auto-Scaling Thresholds**
   ```bash
   # More aggressive scaling
   export SCALING_SCALE_UP_THRESHOLD=60.0
   export SCALING_SCALE_DOWN_THRESHOLD=30.0
   ```

## Maintenance

### Regular Tasks

1. **Update Dependencies**
   ```bash
   # Rebuild with latest dependencies
   ./scripts/launch_scaled_deployment.sh down
   ./scripts/launch_scaled_deployment.sh up --build
   ```

2. **Log Rotation**
   ```bash
   # Archive old logs
   cd logs/
   tar -czf logs_$(date +%Y%m%d).tar.gz *.log
   rm *.log
   ```

3. **Performance Monitoring**
   ```bash
   # Weekly stress test
   python scripts/simple_stress_test.py --concurrent 50 --duration 600
   ```

### Backup and Recovery

1. **Backup Redis Data**
   ```bash
   docker-compose -f docker/docker-compose.scale.yml exec redis redis-cli save
   docker cp $(docker-compose -f docker/docker-compose.scale.yml ps -q redis):/data/dump.rdb backup/redis_$(date +%Y%m%d).rdb
   ```

2. **Backup Application Data**
   ```bash
   tar -czf backup/app_data_$(date +%Y%m%d).tar.gz data/ logs/ configs/
   ```

## Support

For issues and questions:
1. Check logs: `./scripts/launch_scaled_deployment.sh logs`
2. Verify health: `curl http://localhost/health`
3. Run diagnostics: `python scripts/simple_stress_test.py`
4. Review this deployment guide
5. Check system resources: `docker stats`

## Appendix

### Service Ports
- Load Balancer: 80, 443
- Redis: 6379
- Individual API instances: Internal only

### File Structure
```
PyNucleus-Model/
├── docker/
│   ├── docker-compose.scale.yml
│   ├── nginx.conf
│   ├── Dockerfile.api
│   └── Dockerfile.scaling
├── scripts/
│   ├── launch_scaled_deployment.sh
│   └── simple_stress_test.py
├── src/pynucleus/deployment/
│   ├── scaling_manager.py
│   └── cache_integration.py
└── logs/
    └── (service logs)
``` 