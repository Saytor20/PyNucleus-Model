#!/bin/bash
# PyNucleus Docker Build Script
# Builds and validates all Docker services

set -e  # Exit on any error

echo "🚀 Building PyNucleus Docker Infrastructure..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Validate requirements files
echo "🔍 Validating requirements files..."
if pip-compile --dry-run requirements.txt > /dev/null 2>&1; then
    print_status "Main requirements.txt is valid"
else
    print_error "Invalid requirements.txt - please fix pip specifiers"
    exit 1
fi

if pip-compile --dry-run docker/requirements_simple.txt > /dev/null 2>&1; then
    print_status "Docker requirements file is valid"
else
    print_error "Invalid docker/requirements_simple.txt"
    exit 1
fi

# Build individual services
echo "🏗️  Building Docker services..."

# Build main application
echo "Building main application..."
cd .. # Go to project root
if docker build -f docker/Dockerfile -t pynucleus:latest .; then
    print_status "Main application built successfully"
else
    print_error "Failed to build main application"
    exit 1
fi

# Build API service
echo "Building API service..."
if docker build -f docker/Dockerfile.api -t pynucleus-api:latest .; then
    print_status "API service built successfully"
else
    print_error "Failed to build API service"
    exit 1
fi

# Build model service
echo "Building model service..."
cd docker
if docker build -f Dockerfile.model -t pynucleus-model:latest .; then
    print_status "Model service built successfully"
else
    print_error "Failed to build model service"
    exit 1
fi

# Validate Docker Compose
echo "🔧 Validating Docker Compose..."
if docker-compose config > /dev/null 2>&1; then
    print_status "Docker Compose configuration is valid"
else
    print_error "Invalid Docker Compose configuration"
    exit 1
fi

# Test container startup
echo "🧪 Testing container startup..."
if docker run --rm pynucleus:latest python -c "import sys; print('Python version:', sys.version)"; then
    print_status "Container startup test passed"
else
    print_warning "Container startup test failed (non-critical)"
fi

# Display image information
echo "📦 Built Docker Images:"
docker images | grep pynucleus

print_status "Docker build complete! 🎉"
echo ""
echo "🚀 To start services:"
echo "   docker-compose up -d"
echo ""
echo "🔍 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 To stop services:"
echo "   docker-compose down" 