#!/bin/bash
# PyNucleus Web Interface Startup Script

echo "🧪 PyNucleus Web Interface Setup"
echo "================================"

# Check if we're in the right directory
if [ ! -f "web_interface.py" ]; then
    echo "❌ Error: Please run this script from the PyNucleus-Model directory"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python version OK: $python_version"

# Install Gradio if not present
if ! python3 -c "import gradio" 2>/dev/null; then
    echo "📦 Installing Gradio..."
    pip3 install gradio>=4.0.0
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install Gradio. Please install manually:"
        echo "   pip install gradio>=4.0.0"
        exit 1
    fi
fi

echo "✅ Gradio installed"

# Check system health
echo "🏥 Checking system health..."
python3 -m src.pynucleus.cli version > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "⚠️  Warning: PyNucleus CLI not fully functional"
    echo "   Web interface will still work but some features may be limited"
else
    echo "✅ PyNucleus CLI operational"
fi

# Start the web interface
echo ""
echo "🚀 Starting PyNucleus Web Interface..."
echo "📍 Access the interface at:"
echo "   🌐 Local:  http://localhost:7860"
echo "   🌐 Network: http://0.0.0.0:7860"
echo ""
echo "💡 Press Ctrl+C to stop the server"
echo ""

# Launch with error handling
python3 web_interface.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Web interface failed to start"
    echo "💡 Try running manually: python3 web_interface.py"
    echo "📋 Check requirements: pip install -r requirements_web.txt"
fi