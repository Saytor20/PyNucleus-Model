#!/bin/bash
"""
PyNucleus Scaled Deployment Launcher

Launches the complete scaled PyNucleus infrastructure including:
- Redis cache
- Load balancer (nginx)
- Multiple API instances
- Scaling manager
- Monitoring
"""

set -e

# Configuration
DEFAULT_INSTANCES=3
DEFAULT_ENVIRONMENT="production"
COMPOSE_FILE="docker/docker-compose.scale.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Commands:
    up          Start the scaled deployment
    down        Stop the deployment
    scale       Scale API instances
    status      Show deployment status
    logs        Show service logs
    restart     Restart all services
    monitor     Show real-time monitoring

Options:
    --instances N       Number of API instances (default: $DEFAULT_INSTANCES)
    --env ENV          Environment (development|production) (default: $DEFAULT_ENVIRONMENT)
    --no-cache         Disable Redis cache
    --no-scaling       Disable auto-scaling manager
    --build            Force rebuild containers

Examples:
    $0 up --instances 5
    $0 scale --instances 8
    $0 logs api-1
    $0 monitor
EOF
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check if Redis port is available (if running locally)
    if ! $NO_CACHE && netstat -ln 2>/dev/null | grep -q ":6379 "; then
        log_warning "Port 6379 is already in use - may conflict with Redis"
    fi
    
    log_success "Prerequisites check passed"
}

setup_environment() {
    log_info "Setting up environment..."
    
    # Create required directories
    mkdir -p logs
    mkdir -p data/03_processed/chromadb
    mkdir -p test_output
    
    # Set environment variables
    export PYNUCLEUS_ENV=$ENVIRONMENT
    export SCALING_MIN_INSTANCES=2
    export SCALING_MAX_INSTANCES=10
    export SCALING_TARGET_CPU=70
    export SCALING_TARGET_RESPONSE_TIME=2.0
    
    if [ "$ENVIRONMENT" = "production" ]; then
        export FLASK_ENV=production
        export FLASK_DEBUG=false
        export GUNICORN_WORKERS=4
        export GUNICORN_THREADS=2
    else
        export FLASK_ENV=development
        export FLASK_DEBUG=true
        export GUNICORN_WORKERS=2
        export GUNICORN_THREADS=1
    fi
    
    log_success "Environment configured for $ENVIRONMENT"
}

wait_for_redis() {
    log_info "Waiting for Redis to be ready..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if docker-compose -f $COMPOSE_FILE exec -T redis redis-cli ping 2>/dev/null | grep -q "PONG"; then
            log_success "Redis is ready!"
            return 0
        fi
        log_info "Attempt $attempt/$max_attempts: Redis not ready yet..."
        sleep 2
        attempt=$((attempt + 1))
    done
    
    log_error "Redis failed to start after $max_attempts attempts"
    return 1
}

start_deployment() {
    log_info "Starting PyNucleus scaled deployment..."
    
    BUILD_FLAG=""
    if $BUILD; then
        BUILD_FLAG="--build"
    fi
    
    # Start core services first
    log_info "Starting Redis cache..."
    docker-compose -f $COMPOSE_FILE up -d redis $BUILD_FLAG
    
    # Wait for Redis to be ready
    if ! wait_for_redis; then
        exit 1
    fi
    
    # Start model service
    log_info "Starting model service..."
    docker-compose -f $COMPOSE_FILE up -d model $BUILD_FLAG
    
    # Start API instances
    log_info "Starting $INSTANCES API instances..."
    for i in $(seq 1 $INSTANCES); do
        log_info "Starting api-$i..."
        docker-compose -f $COMPOSE_FILE up -d api-$i $BUILD_FLAG
    done
    
    # Start load balancer
    log_info "Starting load balancer..."
    docker-compose -f $COMPOSE_FILE up -d load-balancer $BUILD_FLAG
    
    # Start scaling manager if enabled
    if ! $NO_SCALING; then
        log_info "Starting scaling manager..."
        docker-compose -f $COMPOSE_FILE up -d scaling-manager $BUILD_FLAG
    fi
    
    log_success "Deployment started successfully!"
    
    # Show status
    show_status
    
    # Show access URLs
    echo ""
    log_info "Access URLs:"
    echo "  API (Load Balanced): http://localhost"
    echo "  Health Check:        http://localhost/health"
    echo "  Metrics:            http://localhost/metrics"
    if ! $NO_CACHE; then
        echo "  Redis:              localhost:6379"
    fi
}

stop_deployment() {
    log_info "Stopping PyNucleus deployment..."
    docker-compose -f $COMPOSE_FILE down
    log_success "Deployment stopped"
}

scale_instances() {
    log_info "Scaling API instances to $INSTANCES..."
    
    # Scale using docker-compose
    docker-compose -f $COMPOSE_FILE up -d --scale api=$INSTANCES
    
    # Update nginx configuration if needed
    # (In production, this would be handled by the scaling manager)
    
    log_success "Scaled to $INSTANCES API instances"
    show_status
}

show_status() {
    log_info "Deployment Status:"
    echo ""
    
    # Check service status
    services=("redis" "load-balancer" "model")
    
    # Add API instances
    for i in $(seq 1 10); do  # Check up to 10 instances
        if docker-compose -f $COMPOSE_FILE ps api-$i 2>/dev/null | grep -q "Up"; then
            services+=("api-$i")
        fi
    done
    
    if ! $NO_SCALING; then
        services+=("scaling-manager")
    fi
    
    for service in "${services[@]}"; do
        status=$(docker-compose -f $COMPOSE_FILE ps $service 2>/dev/null | tail -n +3 | awk '{print $4}' | head -1)
        if [ "$status" = "Up" ]; then
            echo -e "  ${GREEN}✓${NC} $service: $status"
        else
            echo -e "  ${RED}✗${NC} $service: $status"
        fi
    done
    
    # Show resource usage
    echo ""
    log_info "Resource Usage:"
    echo "  CPU: $(docker stats --no-stream --format "table {{.CPUPerc}}" | tail -n +2 | head -1)"
    echo "  Memory: $(docker stats --no-stream --format "table {{.MemUsage}}" | tail -n +2 | head -1)"
}

show_logs() {
    service=${1:-""}
    if [ -z "$service" ]; then
        docker-compose -f $COMPOSE_FILE logs -f
    else
        docker-compose -f $COMPOSE_FILE logs -f $service
    fi
}

restart_deployment() {
    log_info "Restarting deployment..."
    stop_deployment
    sleep 2
    start_deployment
}

monitor_deployment() {
    log_info "Starting real-time monitoring (Press Ctrl+C to exit)..."
    
    while true; do
        clear
        echo "PyNucleus Deployment Monitor - $(date)"
        echo "=================================="
        
        show_status
        
        echo ""
        log_info "Recent Activity:"
        docker-compose -f $COMPOSE_FILE logs --tail=5 2>/dev/null | head -20
        
        sleep 5
    done
}

# Parse command line arguments
INSTANCES=$DEFAULT_INSTANCES
ENVIRONMENT=$DEFAULT_ENVIRONMENT
NO_CACHE=false
NO_SCALING=false
BUILD=false
COMMAND=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --instances)
            INSTANCES="$2"
            shift 2
            ;;
        --env)
            ENVIRONMENT="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --no-scaling)
            NO_SCALING=true
            shift
            ;;
        --build)
            BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        up|down|scale|status|logs|restart|monitor)
            COMMAND="$1"
            shift
            break
            ;;
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate instances count
if ! [[ "$INSTANCES" =~ ^[0-9]+$ ]] || [ "$INSTANCES" -lt 1 ] || [ "$INSTANCES" -gt 20 ]; then
    log_error "Invalid instances count: $INSTANCES (must be 1-20)"
    exit 1
fi

# Validate environment
if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
    log_error "Invalid environment: $ENVIRONMENT (must be development or production)"
    exit 1
fi

# Execute command
case $COMMAND in
    up)
        check_prerequisites
        setup_environment
        start_deployment
        ;;
    down)
        stop_deployment
        ;;
    scale)
        scale_instances
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs $1
        ;;
    restart)
        restart_deployment
        ;;
    monitor)
        monitor_deployment
        ;;
    "")
        log_error "No command specified"
        show_usage
        exit 1
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac 