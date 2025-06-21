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
from flask import Flask, request, jsonify, send_from_directory, Response, stream_template
from werkzeug.utils import secure_filename
import time

# Add project root to Python path to access run_pipeline.py
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src to Python path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from pynucleus import __version__
from pynucleus.rag.engine import ask as rag_ask, retrieve
from pynucleus.rag.collector import ingest
from pynucleus.settings import settings
from pynucleus.utils.logging_config import configure_logging, get_logger

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logger = configure_logging(level="INFO")
api_logger = get_logger(__name__)

# Allowed file extensions for document upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'doc', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        expected_key = os.environ.get('PYNUCLEUS_API_KEY')
        
        if not expected_key:
            # Allow operation without API key in development
            if app.debug:
                return f(*args, **kwargs)
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


@app.route('/status', methods=['GET'])
def status():
    """Get system status including database and model information."""
    try:
        # Check ChromaDB status
        chroma_status = "unknown"
        doc_count = 0
        try:
            test_docs = retrieve("test", k=1)
            if test_docs:
                chroma_status = "ready"
                # Get rough document count by testing retrieval
                all_docs = retrieve("", k=1000)  # Get up to 1000 docs
                doc_count = len(all_docs) if all_docs else 0
            else:
                chroma_status = "empty"
        except Exception as e:
            chroma_status = f"error: {str(e)}"
        
        # Check database size
        db_size = 0
        chroma_path = Path(settings.CHROMA_PATH)
        if chroma_path.exists():
            db_files = list(chroma_path.rglob("*"))
            db_size = sum(f.stat().st_size for f in db_files if f.is_file())
        
        return jsonify({
            "status": "operational",
            "version": __version__,
            "database": {
                "status": chroma_status,
                "path": settings.CHROMA_PATH,
                "size_mb": round(db_size / 1024 / 1024, 2),
                "estimated_docs": doc_count
            },
            "model": {
                "id": settings.MODEL_ID,
                "embedding_model": settings.EMB_MODEL,
                "max_tokens": settings.MAX_TOKENS,
                "top_k": settings.RETRIEVE_TOP_K
            },
            "system": {
                "cuda_enabled": settings.USE_CUDA,
                "log_level": settings.LOG_LEVEL
            }
        })
    except Exception as e:
        api_logger.error(f"Status check failed: {e}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500


@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question to the RAG system with enhanced response."""
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
        
        api_logger.info(f"Processing question: {question[:50]}...")
        
        # Record start time for performance metrics
        start_time = time.time()
        
        # Query the RAG system
        result = rag_ask(question)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Enhance response with metadata
        response = {
            "answer": result.get('answer', 'No answer available'),
            "sources": result.get('sources', []),
            "metadata": {
                "processing_time": round(processing_time, 2),
                "question_length": len(question),
                "answer_length": len(result.get('answer', '')),
                "source_count": len(result.get('sources', [])),
                "model": settings.MODEL_ID,
                "timestamp": int(time.time())
            }
        }
        
        api_logger.info(f"Question processed successfully in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        api_logger.error(f"Question processing failed: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "timestamp": int(time.time())
        }), 500


@app.route('/search', methods=['POST'])
def search():
    """Search documents without generating an answer."""
    try:
        if not request.is_json:
            return jsonify({"error": "Content-Type must be application/json"}), 400
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "Missing 'query' field in request body"}), 400
        
        query = data['query']
        top_k = data.get('top_k', settings.RETRIEVE_TOP_K)
        
        if not query or not query.strip():
            return jsonify({"error": "Query cannot be empty"}), 400
        
        api_logger.info(f"Searching documents for: {query[:50]}...")
        
        start_time = time.time()
        documents = retrieve(query, k=top_k)
        processing_time = time.time() - start_time
        
        return jsonify({
            "documents": documents,
            "metadata": {
                "processing_time": round(processing_time, 2),
                "query_length": len(query),
                "result_count": len(documents),
                "timestamp": int(time.time())
            }
        })
        
    except Exception as e:
        api_logger.error(f"Search failed: {str(e)}")
        return jsonify({
            "error": f"Search error: {str(e)}",
            "timestamp": int(time.time())
        }), 500


@app.route('/upload', methods=['POST'])
def upload_document():
    """Upload and process documents."""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                "error": f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        upload_dir = Path("data/01_raw/uploaded_documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / filename
        file.save(str(file_path))
        
        api_logger.info(f"File uploaded: {filename}")
        
        # Process the uploaded document
        try:
            ingest(source_dir=str(upload_dir))
            api_logger.info(f"Document processed successfully: {filename}")
            
            return jsonify({
                "message": "Document uploaded and processed successfully",
                "filename": filename,
                "status": "processed",
                "timestamp": int(time.time())
            })
            
        except Exception as process_error:
            api_logger.error(f"Document processing failed: {process_error}")
            return jsonify({
                "message": "Document uploaded but processing failed",
                "filename": filename,
                "status": "uploaded",
                "error": str(process_error),
                "timestamp": int(time.time())
            }), 207  # Multi-status: partial success
            
    except Exception as e:
        api_logger.error(f"Upload failed: {str(e)}")
        return jsonify({
            "error": f"Upload error: {str(e)}",
            "timestamp": int(time.time())
        }), 500


@app.route('/ingest', methods=['POST'])
def ingest_documents():
    """Ingest documents from the data directory."""
    try:
        # Set DWSIM validation to False to avoid circular imports
        os.environ['DWSIM_VALIDATION'] = 'false'
        
        api_logger.info("Starting document ingestion")
        result = ingest()
        
        return jsonify({
            "status": "success",
            "message": "Documents ingested successfully",
            "details": result
        })
    except Exception as e:
        api_logger.error(f"Document ingestion failed: {e}")
        return jsonify({
            "error": f"Ingestion failed: {str(e)}"
        }), 500


@app.route('/system_diagnostic', methods=['GET'])
def system_diagnostic():
    """Run system validator and return JSON results."""
    try:
        import subprocess
        import json
        
        api_logger.info("Running system diagnostic")
        
        # Run the system validator script with JSON output
        proc = subprocess.run(
            ['python', 'scripts/system_validator.py', '--quiet'],
            capture_output=True, 
            text=True,
            cwd=project_root
        )
        
        # Try to parse any JSON output from the validator
        try:
            # Look for validation results file
            import glob
            results_files = glob.glob('data/validation/results/system_validation_*.json')
            if results_files:
                # Get the most recent results file
                latest_file = max(results_files)
                with open(latest_file, 'r') as f:
                    data = json.load(f)
            else:
                # Fallback to basic status
                data = {
                    "status": "completed",
                    "return_code": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr
                }
        except Exception:
            data = {
                "status": "error", 
                "return_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr
            }
        
        api_logger.info(f"System diagnostic completed with return code {proc.returncode}")
        return jsonify(data)
        
    except Exception as e:
        api_logger.error(f"System diagnostic failed: {e}")
        return jsonify({
            "error": f"Diagnostic failed: {str(e)}"
        }), 500


@app.route('/dev', methods=['GET'])
def dev_console():
    """Serve the developer dashboard."""
    static_dir = Path(__file__).parent / 'static'
    return send_from_directory(static_dir, 'developer_dashboard.html')


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large errors."""
    return jsonify({"error": "File too large. Maximum size is 16MB"}), 413


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    api_logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Only run in development mode when called directly
    app.run(debug=True, port=5001, host='0.0.0.0') 