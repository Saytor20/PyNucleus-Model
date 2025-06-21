#!/usr/bin/env python3
"""
PyNucleus Web Application Launcher

This script provides a convenient way to start the PyNucleus web application
with proper initialization and error handling.
"""

import os
import sys
from pathlib import Path

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

def main():
    """Main entry point for the web application."""
    
    print("🚀 Starting PyNucleus Web Application...")
    print("=" * 50)
    
    try:
        # Import after path setup
        from pynucleus.api.app import app
        from pynucleus.settings import settings
        from pynucleus.utils.logger import logger
        
        print(f"✅ PyNucleus modules loaded successfully")
        print(f"📋 Configuration:")
        print(f"   • ChromaDB Path: {settings.CHROMA_PATH}")
        print(f"   • Model: {settings.MODEL_ID}")
        print(f"   • Embedding Model: {settings.EMB_MODEL}")
        print(f"   • Max Tokens: {settings.MAX_TOKENS}")
        print(f"   • Retrieve Top-K: {settings.RETRIEVE_TOP_K}")
        
        # Check if vector database exists
        chroma_path = Path(settings.CHROMA_PATH)
        if chroma_path.exists():
            print(f"✅ Vector Database Found: {settings.CHROMA_PATH}")
        else:
            print(f"⚠️ Vector Database Not Found: {settings.CHROMA_PATH}")
            print(f"💡 Run document ingestion first using:")
            print(f"   python run_pipeline.py ingest")
        
        print(f"\n🌐 Starting web server...")
        print(f"📍 Application will be available at:")
        print(f"   • Local: http://localhost:5001")
        print(f"   • Network: http://0.0.0.0:5001")
        print(f"\n🔧 Features Available:")
        print(f"   • 💬 Chat Interface - Ask questions about chemical engineering")
        print(f"   • 📤 File Upload - Upload documents to expand knowledge base")
        print(f"   • 📊 System Status - Monitor database and model status")
        print(f"   • 🔧 Developer Console - Advanced diagnostics and testing")
        print(f"   • 🎨 Dark/Light Theme - Toggle appearance")
        print(f"   • 📱 Responsive Design - Works on desktop and mobile")
        print(f"\n💡 Tips:")
        print(f"   • Use Ctrl+Enter to send messages quickly")
        print(f"   • Upload TXT, PDF, MD, DOC, or DOCX files")
        print(f"   • Try the quick action buttons for common questions")
        print(f"   • System auto-refreshes status every 30 seconds")
        print(f"   • Developer console:   http://localhost:5001/dev")
        
        print(f"\n" + "=" * 50)
        print(f"🎯 Ready! Visit http://localhost:5001 to start using PyNucleus")
        print(f"🛑 Press Ctrl+C to stop the server")
        print(f"=" * 50)
        
        logger.info("PyNucleus web application starting")
        
        # Start the Flask application
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5001,
            use_reloader=False  # Avoid double startup messages
        )
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   • Ensure you're in the PyNucleus-Model directory")
        print(f"   • Install the package: pip install -e .")
        print(f"   • Install dependencies: pip install -r requirements.txt")
        print(f"   • Check Python environment setup")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print(f"\n💡 Troubleshooting:")
        print(f"   • Check your Python environment")
        print(f"   • Verify all required directories exist")
        print(f"   • Ensure port 5001 is available")
        print(f"   • For advanced diagnostics, see logs/ directory")
        sys.exit(1)

if __name__ == "__main__":
    main() 