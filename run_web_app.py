#!/usr/bin/env python3
"""
Stable PyNucleus Web Application Runner
Provides enhanced stability, error handling, and automatic recovery
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Set environment variables for stability
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

# Add src to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/stable_runner.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global cleanup handler
def cleanup_resources():
    """Clean up resources on app shutdown."""
    try:
        logger.info("Cleaning up application resources...")
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
            
        logger.info("Resource cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

# Register cleanup handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def check_dependencies():
    """Check if required dependencies are available."""
    try:
        import flask
        import torch
        logger.info("✅ Core dependencies available")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        return False

def setup_directories():
    """Ensure required directories exist."""
    dirs = ['logs', 'data/03_processed', 'source_documents']
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("✅ Required directories setup complete")

def memory_check():
    """Check available memory before starting."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logger.warning(f"⚠️ High memory usage: {memory.percent:.1f}%")
            return False
        logger.info(f"✅ Memory usage acceptable: {memory.percent:.1f}%")
        return True
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return True

def run_app_directly():
    """Run the Flask app directly in the same process."""
    try:
        from pynucleus.api.app import app
        from pynucleus.settings import settings
        from pynucleus.utils.logging_config import get_logger
        
        app_logger = get_logger(__name__)
        
        print("🚀 Starting PyNucleus Web Application...")
        print("=" * 50)
        print(f"✅ PyNucleus modules loaded successfully")
        print(f"📋 Configuration:")
        print(f"   • ChromaDB Path: {settings.CHROMA_PATH}")
        print(f"   • Model: {settings.MODEL_ID}")
        print(f"   • Embedding Model: {settings.EMB_MODEL}")
        print(f"   • Max Tokens: {settings.MAX_TOKENS}")
        print(f"   • Retrieve Top-K: {settings.RETRIEVE_TOP_K}")
        
        chroma_path = Path(settings.CHROMA_PATH)
        if chroma_path.exists():
            print(f"✅ Vector Database Found: {settings.CHROMA_PATH}")
        else:
            print(f"⚠️ Vector Database Not Found: {settings.CHROMA_PATH}")
        
        print(f"\n🌐 Starting web server...")
        print(f"🔧 Developer Console: http://localhost:5001")
        print(f"🔧 Health monitoring: http://localhost:5001/health")
        print(f"=" * 50)
        print(f"🎯 Ready! Visit http://localhost:5001 for PyNucleus Developer Console")
        print(f"🛑 Press Ctrl+C to stop the server")
        print(f"=" * 50)
        
        app_logger.info("PyNucleus web application starting")
        
        # Start the Flask application
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5001,
            use_reloader=False  # Avoid double startup messages
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        return False

class StableWebAppRunner:
    def __init__(self):
        self.restart_count = 0
        self.max_restarts = 10
        self.restart_delay = 30  # seconds
        self.start_time = time.time()
        self.running = True
        
    def should_restart(self):
        """Determine if the application should restart."""
        if self.restart_count >= self.max_restarts:
            logger.error(f"❌ Maximum restart attempts ({self.max_restarts}) exceeded")
            return False
            
        uptime = time.time() - self.start_time
        if uptime < 60:  # If crashed within 1 minute
            logger.warning(f"⚠️ Quick restart detected (uptime: {uptime:.1f}s)")
            time.sleep(self.restart_delay)
            
        return True
        
    def run(self):
        """Main run loop with restart capability."""
        logger.info("🎯 PyNucleus Stable Web App Runner Starting...")
        logger.info(f"📊 Max restarts: {self.max_restarts}")
        logger.info(f"⏱️ Restart delay: {self.restart_delay}s")
        
        # Pre-flight checks
        if not check_dependencies():
            logger.error("❌ Dependency check failed")
            return 1
            
        setup_directories()
        
        if not memory_check():
            logger.error("❌ Memory check failed")
            return 1
            
        # Main run loop
        while self.running:
            self.start_time = time.time()
            
            try:
                logger.info(f"✅ Starting application (attempt {self.restart_count + 1})")
                
                # Run the app directly
                success = run_app_directly()
                
                if not success and self.running:
                    self.restart_count += 1
                    logger.warning(f"⚠️ Application failed, restart attempt {self.restart_count}")
                    
                    cleanup_resources()
                    
                    if not self.should_restart():
                        break
                else:
                    # App completed successfully (normal shutdown)
                    break
                    
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                self.restart_count += 1
                logger.error(f"❌ Unexpected error: {e}")
                
                cleanup_resources()
                
                if not self.should_restart():
                    break
                
        logger.info("🛑 PyNucleus Stable Web App Runner Stopped")
        return 0

def main():
    """Main entry point - try direct run first, fallback to stable runner."""
    try:
        # For development/testing, try running directly first
        if len(sys.argv) > 1 and sys.argv[1] == '--direct':
            return run_app_directly()
        
        # Default: use stable runner with restart capability
        runner = StableWebAppRunner()
        exit_code = runner.run()
        sys.exit(exit_code)
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 