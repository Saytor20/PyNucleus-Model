#!/bin/bash

# PyNucleus Dashboard Launcher Script
# This script provides easy deployment options for the real-time analytics dashboard

set -e

# Configuration
DASHBOARD_PORT=${PYNUCLEUS_DASHBOARD_PORT:-8501}
DASHBOARD_HOST=${PYNUCLEUS_DASHBOARD_HOST:-localhost}
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🔬 PyNucleus Real-Time Analytics Dashboard Launcher"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Dashboard URL: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
echo ""

# Check dependencies
echo "Checking dependencies..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly
fi

# Check if PyNucleus modules are accessible
cd "$PROJECT_ROOT"
if ! python -c "import sys; sys.path.insert(0, 'src'); from pynucleus.diagnostics.runner import DiagnosticRunner" 2>/dev/null; then
    echo "⚠️  Warning: PyNucleus modules may not be fully accessible"
    echo "   Dashboard will attempt to run with graceful error handling"
fi

# Check if logs directory exists
if [ ! -d "$PROJECT_ROOT/logs" ]; then
    echo "⚠️  Warning: Logs directory not found. Creating..."
    mkdir -p "$PROJECT_ROOT/logs"
    touch "$PROJECT_ROOT/logs/app.log"
fi

echo "✅ Dependencies checked"
echo ""

# Launch options
echo "Select launch mode:"
echo "1) Local development (default port 8501)"
echo "2) Custom port"
echo "3) Background service"
echo "4) Production deployment"
echo ""
read -p "Enter choice (1-4) [1]: " choice
choice=${choice:-1}

case $choice in
    1)
        echo "🚀 Launching dashboard in development mode..."
        streamlit run src/pynucleus/diagnostics/dashboard.py \
            --server.port $DASHBOARD_PORT \
            --server.address $DASHBOARD_HOST
        ;;
    2)
        read -p "Enter port number: " custom_port
        echo "🚀 Launching dashboard on port $custom_port..."
        streamlit run src/pynucleus/diagnostics/dashboard.py \
            --server.port $custom_port \
            --server.address $DASHBOARD_HOST
        ;;
    3)
        echo "🚀 Launching dashboard as background service..."
        nohup streamlit run src/pynucleus/diagnostics/dashboard.py \
            --server.port $DASHBOARD_PORT \
            --server.address $DASHBOARD_HOST \
            --server.headless true > logs/dashboard.log 2>&1 &
        echo "Dashboard started in background. PID: $!"
        echo "Access at: http://$DASHBOARD_HOST:$DASHBOARD_PORT"
        echo "Logs: $PROJECT_ROOT/logs/dashboard.log"
        ;;
    4)
        echo "🚀 Production deployment mode..."
        echo "Using production-optimized settings..."
        streamlit run src/pynucleus/diagnostics/dashboard.py \
            --server.port $DASHBOARD_PORT \
            --server.address 0.0.0.0 \
            --server.headless true \
            --server.enableCORS false \
            --server.enableXsrfProtection true
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac 