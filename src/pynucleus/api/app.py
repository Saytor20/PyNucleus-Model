"""
PyNucleus Flask API Application

Production-ready Flask app with:
- Application factory pattern  
- Confidence calibration integration
- RAG pipeline integration
- Comprehensive error handling
- Health monitoring
"""

import os
import sys
import atexit
import signal
import logging
from pathlib import Path
from flask import Flask, request, jsonify, g
from flask.logging import default_handler
import json
from typing import Optional, Dict, Any

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
    app = Flask(__name__)
    
    # Basic configuration
    app.config.update({
        'DEBUG': os.getenv('FLASK_DEBUG', 'false').lower() == 'true',
        'TESTING': False,
        'SECRET_KEY': os.getenv('SECRET_KEY', 'dev-key-change-in-production'),
        'JSON_SORT_KEYS': False
    })
    
    # Apply additional config if provided
    if config:
        app.config.update(config)
    
    # Setup logging
    configure_logging(app)
    
    # Register routes
    register_routes(app)
    
    # Setup cleanup handlers
    setup_cleanup_handlers(app)
    
    logger.info("PyNucleus Flask API starting up...")
    
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

def register_routes(app: Flask) -> None:
    """Register all application routes"""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        logger.info("Health check requested")
        return jsonify({
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "rag_engine": _rag_engine is not None,
                "confidence_calibration": True
            }
        })
    
    @app.route('/ask', methods=['POST'])
    def ask_question():
        """
        Main RAG query endpoint with confidence calibration.
        
        Request JSON:
        {
            "question": "What is distillation?",
            "user_feedback": 0.8  // Optional: for calibration training
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
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            logger.info(f"Processing question: {question[:50]}... (stream: False)")
            
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
                    "is_complex_question": result.get("is_complex_question", False)
                }
            }
            
            # Include confidence calibration details if available
            if "confidence_calibration" in result:
                response["confidence_calibration"] = result["confidence_calibration"]
            
            logger.info(f"Question processed successfully in {result.get('response_time', 0):.2f}s")
            return jsonify(response)
            
        except Exception as e:
            logger.error(f"Ask endpoint error: {e}")
            return jsonify({
                "error": "Internal server error",
                "answer": "An error occurred while processing your question",
                "confidence": 0.0
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