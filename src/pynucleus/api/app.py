"""
PyNucleus Flask API application.

Provides REST endpoints for the PyNucleus chemical process simulation and RAG system.
Includes health checks and question answering capabilities.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory

# Add project root to Python path to access run_pipeline.py
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus import __version__
from pynucleus.pipeline.pipeline_rag import RAGPipeline
from pynucleus.utils.logging_config import configure_logging, get_logger

app = Flask(__name__)

# Configure logging
logger = configure_logging(level="INFO")
api_logger = get_logger(__name__)


def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        expected_key = os.environ.get('PYNUCLEUS_API_KEY')
        
        if not expected_key:
            api_logger.error("PYNUCLEUS_API_KEY environment variable not set")
            return jsonify({"error": "API key authentication not configured"}), 500
        
        if not api_key:
            api_logger.warning("API request missing API key")
            return jsonify({"error": "API key required"}), 401
        
        if api_key != expected_key:
            api_logger.warning("API request with invalid API key")
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function


@app.route('/', methods=['GET'])
def index():
    """Serve the browser UI."""
    static_dir = Path(__file__).parent / 'static'
    return send_from_directory(static_dir, 'index.html')


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    api_logger.info("Health check requested")
    return jsonify({
        "status": "healthy",
        "version": __version__,
        "service": "PyNucleus API"
    })


@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question to the RAG system."""
    try:
        # Validate request
        if not request.is_json:
            api_logger.warning("Ask request with non-JSON content type")
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data or 'question' not in data:
            api_logger.warning("Ask request missing question field")
            return jsonify({"error": "Missing 'question' field in request body"}), 400
        
        question = data['question']
        if not question or not question.strip():
            api_logger.warning("Ask request with empty question")
            return jsonify({"error": "Question cannot be empty"}), 400
        
        # Optional parameters
        model_id = data.get('model_id', 'Qwen/Qwen2.5-1.5B-Instruct')
        top_k = data.get('top_k', 5)
        
        api_logger.info(f"Processing question: {question[:50]}...")
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(data_dir="data")
        
        # Query the RAG system
        result = rag_pipeline.query(question, top_k=top_k)
        
        # Always use LLM for enhanced response
        try:
            from pynucleus.llm.llm_runner import LLMRunner
            llm_runner = LLMRunner(model_id=model_id)
            
            api_logger.info(f"Generating enhanced response with {model_id}")
            
            # Create enhanced prompt with context using Guidance integration
            from pynucleus.llm.prompting import build_prompt
            context = result.get('answer', '')
            enhanced_prompt = build_prompt(context, question)
            
            llm_response = llm_runner.ask(
                question=enhanced_prompt,
                max_length=500,
                temperature=0.7
            )
            
            response = {
                "answer": llm_response,
                "rag_answer": result.get('answer', ''),
                "confidence": float(result.get('confidence', 0)),
                "sources": result.get('sources', []),
                "model_id": model_id
            }
            
        except Exception as llm_error:
            api_logger.error(f"LLM processing failed: {llm_error}")
            response = {
                "answer": result.get('answer', 'No answer available'),
                "confidence": float(result.get('confidence', 0)),
                "sources": result.get('sources', []),
                "model_id": "RAG-only (LLM failed)"
            }
        
        api_logger.info("Question processed successfully")
        return jsonify(response)
        
    except Exception as e:
        api_logger.error(f"Question processing failed: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    api_logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Only run in development mode when called directly
    app.run(debug=True, port=5001) 