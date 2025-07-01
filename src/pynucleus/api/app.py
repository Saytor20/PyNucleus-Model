"""
PyNucleus Flask API Application

Production-ready Flask app with:
- Application factory pattern  
- Confidence calibration integration
- RAG pipeline integration
- Redis distributed caching
- Horizontal scaling support
- Comprehensive error handling
- Health monitoring
"""

import os
import sys
import atexit
import signal
import logging
from pathlib import Path
from flask import Flask, request, jsonify, g, render_template
from flask.logging import default_handler
import json
from typing import Optional, Dict, Any
from datetime import datetime

# Setup path for imports
app_root = Path(__file__).parent.parent.parent.parent
if str(app_root) not in sys.path:
    sys.path.insert(0, str(app_root))

logger = logging.getLogger(__name__)

# Global RAG engine instance (lazy initialization)
_rag_engine = None

def get_rag_engine():
    """Get or initialize RAG engine singleton"""
    global _rag_engine
    if _rag_engine is None:
        try:
            from pynucleus.rag.engine import ask
            _rag_engine = ask
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            _rag_engine = None
    return _rag_engine

def create_app(config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Application factory for creating Flask app instances.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Flask application
    """
    # Set template folder path
    template_dir = Path(__file__).parent.parent / "templates"
    app = Flask(__name__, template_folder=str(template_dir))
    
    # Basic configuration
    app.config.update({
        'DEBUG': os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        'TESTING': False,
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
        'JSON_SORT_KEYS': False,
        'REDIS_URL': os.getenv('REDIS_URL', 'redis://localhost:6379/0'),
        'PYNUCLEUS_INSTANCE_ID': os.getenv('PYNUCLEUS_INSTANCE_ID', 'api-default')
    })
    
    # Apply additional config if provided
    if config:
        app.config.update(config)
    
    # Setup logging
    configure_logging(app)
    
    # Initialize Redis cache
    initialize_cache(app)
    
    # Register routes
    register_routes(app)
    
    # Setup cleanup handlers
    setup_cleanup_handlers(app)
    
    logger.info(f"PyNucleus Flask API starting up (Instance: {app.config['PYNUCLEUS_INSTANCE_ID']})...")
    
    return app

def configure_logging(app: Flask) -> None:
    """Configure application logging"""
    if not app.debug:
        # Production logging setup
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s %(levelname)s %(name)s: %(message)s'
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)

def initialize_cache(app: Flask) -> None:
    """Initialize Redis distributed cache"""
    try:
        from pynucleus.deployment.cache_integration import get_cache
        cache = get_cache()
        app.cache = cache
        logger.info("Redis distributed cache initialized")
    except Exception as e:
        logger.warning(f"Cache initialization failed: {e}")
        app.cache = None

def register_routes(app: Flask) -> None:
    """Register all application routes"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint with cache status"""
        cache_status = False
        cache_stats = {}
        
        if hasattr(app, 'cache') and app.cache:
            cache_status = app.cache.enabled
            if cache_status:
                cache_stats = app.cache.get_stats()
        
        logger.info("Health check requested")
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "instance_id": app.config.get('PYNUCLEUS_INSTANCE_ID'),
            "components": {
                "rag_engine": _rag_engine is not None,
                "confidence_calibration": True,
                "cache": cache_status,
                "cache_stats": cache_stats
            }
        })
    
    @app.route('/metrics', methods=['GET'])
    def metrics():
        """Prometheus-style metrics endpoint"""
        try:
            from pynucleus.deployment.cache_integration import get_cache_metrics
            metrics_data = get_cache_metrics().get_metrics()
            
            # Format as Prometheus metrics
            response = []
            response.append(f"# HELP pynucleus_cache_hits_total Total cache hits")
            response.append(f"# TYPE pynucleus_cache_hits_total counter")
            response.append(f"pynucleus_cache_hits_total {{instance=\"{app.config['PYNUCLEUS_INSTANCE_ID']}\"}} {metrics_data['hit_count']}")
            
            response.append(f"# HELP pynucleus_cache_misses_total Total cache misses")
            response.append(f"# TYPE pynucleus_cache_misses_total counter")
            response.append(f"pynucleus_cache_misses_total {{instance=\"{app.config['PYNUCLEUS_INSTANCE_ID']}\"}} {metrics_data['miss_count']}")
            
            response.append(f"# HELP pynucleus_cache_hit_rate Cache hit rate")
            response.append(f"# TYPE pynucleus_cache_hit_rate gauge")
            response.append(f"pynucleus_cache_hit_rate {{instance=\"{app.config['PYNUCLEUS_INSTANCE_ID']}\"}} {metrics_data['hit_rate']}")
            
            return "\n".join(response), 200, {'Content-Type': 'text/plain'}
        except Exception as e:
            logger.error(f"Metrics endpoint error: {e}")
            return "# Error generating metrics", 500, {'Content-Type': 'text/plain'}
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        """
        Main RAG query endpoint with confidence calibration and caching.
        
        Request JSON:
        {
            "question": "What is distillation?",
            "user_feedback": 0.8,  // Optional: for calibration training
            "use_cache": true       // Optional: enable/disable caching
        }
        
        Response JSON:
        {
            "answer": "...",
            "sources": [...],
            "confidence": 0.85,  // Calibrated confidence
            "confidence_calibration": {
                "original": 0.7,
                "calibrated": 0.85,
                "platt_calibrated": 0.82,
                "isotonic_calibrated": 0.88
            },
            "metadata": {...}
        }
        """
        try:
            # Validate request
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            question = data.get('question', '').strip()
            user_feedback = data.get('user_feedback')  # Optional for training
            use_cache = data.get('use_cache', True)  # Cache enabled by default
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            logger.info(f"Processing question: {question[:50]}... (cache: {use_cache})")
            
            # Check cache first if enabled
            cached_response = None
            if use_cache and hasattr(app, 'cache') and app.cache:
                cached_response = app.cache.get(question)
                if cached_response:
                    from pynucleus.deployment.cache_integration import get_cache_metrics
                    get_cache_metrics().record_hit()
                    logger.info("Returning cached response")
                    return jsonify(cached_response)
                else:
                    from pynucleus.deployment.cache_integration import get_cache_metrics
                    get_cache_metrics().record_miss()
            
            # Get RAG engine
            rag_ask = get_rag_engine()
            if rag_ask is None:
                return jsonify({
                    "error": "RAG engine not available",
                    "answer": "System temporarily unavailable",
                    "confidence": 0.0
                }), 503
            
            # Process question through RAG pipeline (with confidence calibration)
            result = rag_ask(question)
            
            # Add user feedback for calibration training if provided
            if user_feedback is not None:
                try:
                    from pynucleus.eval.confidence_calibration import get_calibrator
                    calibrator = get_calibrator()
                    calibrator.collect_interaction_data(question, result, user_feedback)
                    logger.info(f"Collected calibration training data with feedback: {user_feedback}")
                except Exception as e:
                    logger.warning(f"Failed to collect calibration training data: {e}")
            
            # Ensure required fields
            response = {
                "answer": result.get("answer", "No answer available"),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.0),
                "metadata": {
                    "response_time": result.get("response_time", 0.0),
                    "retrieval_count": result.get("retrieval_count", 0),
                    "has_citations": result.get("has_citations", False),
                    "tokens_used": result.get("tokens_used", 0),
                    "is_complex_question": result.get("is_complex_question", False),
                    "instance_id": app.config['PYNUCLEUS_INSTANCE_ID'],
                    "cache_hit": False
                }
            }
            
            # Include confidence calibration details if available
            if "confidence_calibration" in result:
                response["confidence_calibration"] = result["confidence_calibration"]
            
            # Cache the response if enabled and successful
            if use_cache and hasattr(app, 'cache') and app.cache and not result.get("error"):
                try:
                    # Determine cache TTL based on confidence and complexity
                    cache_ttl = 3600  # Default 1 hour
                    if result.get("confidence", 0) > 0.8:
                        cache_ttl = 7200  # 2 hours for high confidence
                    elif result.get("is_complex_question", False):
                        cache_ttl = 1800  # 30 minutes for complex questions
                    
                    app.cache.set(question, response, ttl=cache_ttl)
                    logger.debug(f"Cached response with TTL: {cache_ttl}s")
                except Exception as e:
                    logger.warning(f"Failed to cache response: {e}")
            
            logger.info(f"Question processed successfully in {result.get('response_time', 0):.2f}s")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Ask endpoint error: {e}")
            return jsonify({
                "error": "Internal server error",
                "answer": "I apologize, but I encountered an error processing your question.",
                "confidence": 0.0,
                "metadata": {
                    "instance_id": app.config['PYNUCLEUS_INSTANCE_ID'],
                    "error_type": type(e).__name__
                }
            }), 500
    
    @app.route('/calibration/report', methods=['GET'])
    def calibration_report():
        """Get confidence calibration system report"""
        try:
            from pynucleus.eval.confidence_calibration import get_calibrator
            calibrator = get_calibrator()
            report = calibrator.get_calibration_report()
            return jsonify(report)
        except Exception as e:
            logger.error(f"Calibration report error: {e}")
            return jsonify({
                "error": "Failed to generate calibration report",
                "status": "error"
            }), 500
    
    @app.route('/calibration/feedback', methods=['POST'])
    def submit_feedback():
        """
        Submit user feedback for calibration training.
        
        Request JSON:
        {
            "question": "What is distillation?",
            "answer": "...",
            "feedback_score": 0.8  // 0.0 (poor) to 1.0 (excellent)
        }
        """
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
                
            data = request.get_json()
            question = data.get('question', '').strip()
            answer = data.get('answer', '').strip()
            feedback_score = data.get('feedback_score')
            
            if not all([question, answer]) or feedback_score is None:
                return jsonify({
                    "error": "question, answer, and feedback_score are required"
                }), 400
                
            if not 0.0 <= feedback_score <= 1.0:
                return jsonify({
                    "error": "feedback_score must be between 0.0 and 1.0"
                }), 400
            
            # Create mock result for calibration training
            mock_result = {
                "answer": answer,
                "confidence": 0.5,  # Will be updated by calibration
                "response_time": 1.0,
                "sources": [],
                "has_citations": False,
                "retrieval_score": 0.5,
                "context_length": len(answer)
            }
            
            from pynucleus.eval.confidence_calibration import get_calibrator
            calibrator = get_calibrator()
            sample = calibrator.collect_interaction_data(question, mock_result, feedback_score)
            
            return jsonify({
                "message": "Feedback submitted successfully",
                "sample_id": str(sample.timestamp),
                "training_samples_count": len(calibrator.training_samples)
            })
            
        except Exception as e:
            logger.error(f"Feedback submission error: {e}")
            return jsonify({
                "error": "Failed to submit feedback"
            }), 500
    
    @app.route('/', methods=['GET'])
    def home():
        """Simple home page"""
        return jsonify({
            "message": "PyNucleus API",
            "version": "1.0.0",
            "endpoints": {
                "health": "/health",
                "ask": "/ask",
                "calibration_report": "/calibration/report", 
                "submit_feedback": "/calibration/feedback"
            },
            "confidence_calibration": "enabled"
        })

    @app.route('/dashboard', methods=['GET'])
    def dashboard():
        """Main dashboard page with Q&A and system diagnostics."""
        return render_template('dashboard.html')
    
    @app.route('/api/ask', methods=['POST'])
    def api_ask():
        """Enhanced Q&A endpoint with confidence rating."""
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            question = data.get('question', '').strip()
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            # Get RAG engine
            rag_engine = get_rag_engine()
            if not rag_engine:
                return jsonify({"error": "RAG engine not available"}), 503
            
            # Get answer from RAG system
            result = rag_engine(question)
            
            response = {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "confidence": result.get("confidence", 0.5),
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"API ask error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/confidence-rating', methods=['POST'])
    def submit_confidence_rating():
        """Submit confidence rating for Q&A responses."""
        try:
            if not request.is_json:
                return jsonify({"error": "Request must be JSON"}), 400
            
            data = request.get_json()
            question = data.get('question', '').strip()
            answer = data.get('answer', '').strip()
            rating = data.get('rating', 0)
            
            if not question or not answer:
                return jsonify({"error": "Question and answer are required"}), 400
            
            if not isinstance(rating, int) or rating < 1 or rating > 10:
                return jsonify({"error": "Rating must be integer between 1-10"}), 400
            
            # Save confidence rating (you can extend this to save to database)
            confidence_data = {
                "question": question,
                "answer": answer,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            
            # For now, just log it (you can save to file/database later)
            logger.info(f"Confidence rating submitted: {confidence_data}")
            
            return jsonify({
                "message": "Confidence rating submitted successfully",
                "rating": rating
            })
            
        except Exception as e:
            logger.error(f"Confidence rating error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/diagnostics', methods=['POST'])
    def run_diagnostics():
        """Run comprehensive system diagnostics and return terminal output."""
        try:
            import subprocess
            import time
            
            # Run comprehensive system diagnostic
            cmd = ['python3', 'scripts/comprehensive_system_diagnostic.py']
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            execution_time = time.time() - start_time
            
            response = {
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Diagnostics timed out after 5 minutes"}), 408
        except Exception as e:
            logger.error(f"Diagnostics error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/validation', methods=['POST'])
    def run_validation():
        """Run system validation and return terminal output."""
        try:
            import subprocess
            import time
            
            # Run system validator
            cmd = ['python3', 'scripts/system_validator.py']
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            execution_time = time.time() - start_time
            
            response = {
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Validation timed out after 5 minutes"}), 408
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return jsonify({"error": str(e)}), 500
    
    @app.route('/api/statistics', methods=['POST'])
    def run_statistics():
        """Run system statistics and return terminal output."""
        try:
            import subprocess
            import time
            
            # Run system statistics
            cmd = ['python3', 'scripts/system_statistics.py']
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            execution_time = time.time() - start_time
            
            response = {
                "return_code": result.returncode,
                "output": result.stdout,
                "error": result.stderr,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
            
            return jsonify(response)
            
        except subprocess.TimeoutExpired:
            return jsonify({"error": "Statistics timed out after 2 minutes"}), 408
        except Exception as e:
            logger.error(f"Statistics error: {e}")
            return jsonify({"error": str(e)}), 500

def setup_cleanup_handlers(app: Flask) -> None:
    """Setup graceful shutdown handlers"""
    
    def cleanup_resources():
        """Clean up application resources"""
        logger.info("Cleaning up application resources...")
        # Add any cleanup logic here
        logger.info("Resource cleanup completed")
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        cleanup_resources()
        sys.exit(0)
    
    # Register cleanup handlers
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# Create default app instance for development
app = create_app()

if __name__ == '__main__':
    # Development server
    app.run(
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 5001)), 
        debug=True
    ) 