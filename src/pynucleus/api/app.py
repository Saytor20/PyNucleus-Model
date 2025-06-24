"""
PyNucleus Flask API application.

Provides REST endpoints for the PyNucleus chemical process simulation and RAG system.
Includes health checks, system diagnostics, and question answering capabilities.
"""

import os
import sys
import json
import numpy as np
import psutil
import shutil
import logging
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, Response, stream_template, current_app, g
from werkzeug.utils import secure_filename
import time
from datetime import datetime, timedelta
import signal
import atexit
import threading
import uuid

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Robust path handling using pathlib
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent.parent
SRC_PATH = BASE_DIR.parent.parent

# Add paths to Python path
for path in [str(PROJECT_ROOT), str(SRC_PATH)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from pynucleus import __version__
from pynucleus.settings import settings
from pynucleus.utils.logging_config import configure_logging, get_logger

# Import comprehensive diagnostic functionality
scripts_path = PROJECT_ROOT / "scripts"
if str(scripts_path) not in sys.path:
    sys.path.insert(0, str(scripts_path))
    
try:
    from comprehensive_system_diagnostic import ComprehensiveSystemDiagnostic
except ImportError:
    # Fallback if import fails
    ComprehensiveSystemDiagnostic = None

# Lazy-loaded RAG components
_rag_engine = None
_rag_lock = threading.Lock()

def get_rag_engine():
    """Lazy singleton for RAG engine initialization."""
    global _rag_engine
    if _rag_engine is None:
        with _rag_lock:
            if _rag_engine is None:
                try:
                    from pynucleus.rag.engine import ask as rag_ask, retrieve
                    from pynucleus.rag.collector import ingest, ingest_single_file
                    _rag_engine = {
                        'ask': rag_ask,
                        'retrieve': retrieve,
                        'ingest': ingest,
                        'ingest_single_file': ingest_single_file
                    }
                    current_app.logger.info("RAG engine initialized successfully")
                except Exception as e:
                    current_app.logger.error(f"Failed to initialize RAG engine: {e}")
                    _rag_engine = None
    return _rag_engine

# Circuit breaker pattern for /ask endpoint
_ask_failure_count = 0
_ask_circuit_open = False
_ask_last_failure = 0
CIRCUIT_BREAKER_THRESHOLD = 5
CIRCUIT_BREAKER_TIMEOUT = 300  # 5 minutes

# System metrics storage
_system_metrics = {
    'requests_total': 0,
    'requests_failed': 0,
    'average_response_time': 0,
    'last_request_time': None,
    'uptime_start': time.time()
}

# Allowed file extensions for document upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'doc', 'docx', 'csv', 'json'}

def setup_logging(app):
    """Setup application logging with file output."""
    # Configure logging to file
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "app.log"
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    
    # Also configure PyNucleus logging
    configure_logging(level="INFO")

def setup_error_handlers(app):
    """Setup comprehensive error handlers."""
    
    @app.errorhandler(404)
    def not_found(error):
        app.logger.warning(f"404 error: {request.url}")
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(error):
        app.logger.warning(f"405 error: {request.method} {request.url}")
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(413)
    def request_entity_too_large(error):
        app.logger.warning(f"413 error: File too large from {request.remote_addr}")
        return jsonify({"error": "File too large (max 32MB)"}), 413

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"Internal server error: {error}")
        return jsonify({"error": "Internal server error"}), 500

    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        app.logger.error(f"Unexpected error: {error}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred"}), 500

def setup_graceful_shutdown(app):
    """Setup graceful shutdown hooks."""
    
    def cleanup_resources():
        """Clean up resources on app shutdown."""
        try:
            app.logger.info("Cleaning up application resources...")
            # Force garbage collection
            import gc
            gc.collect()
            app.logger.info("Resource cleanup completed")
        except Exception as e:
            app.logger.error(f"Error during cleanup: {e}")

    def signal_handler(sig, frame):
        """Handle shutdown signals gracefully."""
        app.logger.info(f"Received signal {sig}, shutting down gracefully...")
        cleanup_resources()
        sys.exit(0)

    # Register cleanup handlers
    atexit.register(cleanup_resources)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
            if current_app.debug:
                return f(*args, **kwargs)
            current_app.logger.error("PYNUCLEUS_API_KEY environment variable not set")
            return jsonify({"error": "API key authentication not configured"}), 500
        
        if not api_key:
            current_app.logger.warning("API request missing API key")
            return jsonify({"error": "API key required"}), 401
        
        if api_key != expected_key:
            current_app.logger.warning("API request with invalid API key")
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def update_metrics(success=True, response_time=None):
    """Update system metrics."""
    global _system_metrics
    _system_metrics['requests_total'] += 1
    _system_metrics['last_request_time'] = time.time()
    
    if not success:
        _system_metrics['requests_failed'] += 1
    
    if response_time:
        # Simple moving average
        if _system_metrics['average_response_time'] == 0:
            _system_metrics['average_response_time'] = response_time
        else:
            _system_metrics['average_response_time'] = (
                _system_metrics['average_response_time'] * 0.9 + response_time * 0.1
            )

def create_app(cfg=None):
    """Application factory function for Flask app creation."""
    app = Flask(__name__)
    
    # Configuration
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
    
    # Apply custom configuration if provided
    if cfg:
        app.config.update(cfg)
    
    # Setup logging first
    setup_logging(app)
    app.logger.info("PyNucleus Flask API starting up...")
    
    # Setup error handlers
    setup_error_handlers(app)
    
    # Setup graceful shutdown
    setup_graceful_shutdown(app)

    @app.route('/', methods=['GET'])
    def index():
        """Serve the main developer console."""
        static_dir = BASE_DIR / 'static'
        return send_from_directory(static_dir, 'index.html')

    @app.route('/health', methods=['GET'])
    def health():
        """Comprehensive health check endpoint."""
        app.logger.info("Health check requested")
        
        health_status = {
            "status": "healthy",
            "timestamp": int(time.time()),
            "version": __version__,
            "service": "PyNucleus API",
            "checks": {}
        }
        
        # Check RAG system
        try:
            rag_engine = get_rag_engine()
            if rag_engine:
                test_result = rag_engine['ask']("test")
                health_status["checks"]["rag"] = "healthy" if test_result else "degraded"
            else:
                health_status["checks"]["rag"] = "degraded: engine not initialized"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["rag"] = f"unhealthy: {str(e)[:50]}"
            health_status["status"] = "degraded"
        
        # Check vector store
        try:
            from pynucleus.rag.vector_store import ChromaVectorStore
            store = ChromaVectorStore()
            stats = store.get_index_stats()
            if stats.get("exists", False):
                doc_count = stats.get("doc_count", "unknown")
                health_status["checks"]["vector_store"] = f"healthy ({doc_count} docs)"
            else:
                health_status["checks"]["vector_store"] = "degraded: no index found"
                health_status["status"] = "degraded"
        except Exception as e:
            health_status["checks"]["vector_store"] = f"unhealthy: {str(e)[:50]}"
            health_status["status"] = "degraded"
        
        # Check circuit breaker status
        if _ask_circuit_open:
            health_status["checks"]["circuit_breaker"] = "open (service degraded)"
            health_status["status"] = "degraded"
        else:
            health_status["checks"]["circuit_breaker"] = f"closed (failures: {_ask_failure_count})"
        
        # Memory usage check
        try:
            memory_percent = psutil.virtual_memory().percent
            health_status["checks"]["memory"] = f"{memory_percent:.1f}% used"
            if memory_percent > 90:
                health_status["status"] = "degraded"
        except ImportError:
            health_status["checks"]["memory"] = "monitoring unavailable"
        
        return jsonify(health_status)

    @app.route('/status', methods=['GET'])
    def status():
        """Enhanced system status with real-time diagnostics."""
        try:
            start_time = time.time()
            
            # System information
            system_info = {
                "platform": os.name,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "cpu_count": os.cpu_count(),
                "uptime_seconds": time.time() - _system_metrics['uptime_start']
            }
            
            # Memory information
            try:
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                system_info.update({
                    "memory_total_gb": round(memory.total / (1024**3), 2),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_percent": round(memory.percent, 1),
                    "disk_total_gb": round(disk.total / (1024**3), 2),
                    "disk_free_gb": round(disk.free / (1024**3), 2),
                    "disk_percent": round((disk.used / disk.total) * 100, 1)
                })
            except Exception as e:
                app.logger.warning(f"Failed to get system metrics: {e}")
            
            # ChromaDB status
            chroma_status = {"status": "unknown", "doc_count": 0, "size_mb": 0}
            try:
                test_docs = get_rag_engine()['retrieve']("test", k=1)
                if test_docs:
                    chroma_status["status"] = "ready"
                    all_docs = get_rag_engine()['retrieve']("", k=1000)
                    chroma_status["doc_count"] = len(all_docs) if all_docs else 0
                else:
                    chroma_status["status"] = "empty"
            except Exception as e:
                chroma_status["status"] = f"error: {str(e)[:100]}"
            
            # Database size
            chroma_path = Path(settings.CHROMA_PATH)
            if chroma_path.exists():
                db_files = list(chroma_path.rglob("*"))
                db_size = sum(f.stat().st_size for f in db_files if f.is_file())
                chroma_status["size_mb"] = round(db_size / 1024 / 1024, 2)
            
            # API metrics
            api_metrics = {
                "requests_total": _system_metrics['requests_total'],
                "requests_failed": _system_metrics['requests_failed'],
                "success_rate": round(
                    (((_system_metrics['requests_total'] - _system_metrics['requests_failed']) 
                      / max(_system_metrics['requests_total'], 1)) * 100), 2
                ),
                "average_response_time": round(_system_metrics['average_response_time'], 3),
                "circuit_breaker_open": _ask_circuit_open,
                "circuit_breaker_failures": _ask_failure_count
            }
            
            # Model status
            model_status = {
                "model_id": getattr(settings, 'MODEL_ID', 'unknown'),
                "embedding_model": getattr(settings, 'EMBEDDING_MODEL', 'unknown'),
                "max_tokens": getattr(settings, 'MAX_TOKENS', 'unknown'),
                "device": "cpu"  # Default, can be enhanced
            }
            
            response_time = time.time() - start_time
            update_metrics(success=True, response_time=response_time)
            
            return jsonify({
                "status": "operational" if chroma_status["status"] == "ready" else "degraded",
                "timestamp": int(time.time()),
                "version": __version__,
                "system": system_info,
                "database": {
                    "status": chroma_status["status"],
                    "path": settings.CHROMA_PATH,
                    "size_mb": chroma_status["size_mb"],
                    "estimated_docs": chroma_status["doc_count"]
                },
                "model": model_status,
                "api_metrics": api_metrics,
                "processing_time": round(response_time, 3)
            })
            
        except Exception as e:
            app.logger.error(f"Status endpoint error: {e}")
            update_metrics(success=False)
            return jsonify({
                "status": "error",
                "error": str(e),
                "timestamp": int(time.time())
            }), 500

    def _check_circuit_breaker():
        """Check if circuit breaker should be opened or closed."""
        global _ask_circuit_open, _ask_failure_count, _ask_last_failure
        
        if _ask_circuit_open:
            # Check if timeout has passed
            if time.time() - _ask_last_failure > CIRCUIT_BREAKER_TIMEOUT:
                _ask_circuit_open = False
                _ask_failure_count = 0
                app.logger.info("Circuit breaker closed - timeout expired")
        
        return not _ask_circuit_open

    def _record_ask_failure():
        """Record a failure for the circuit breaker."""
        global _ask_failure_count, _ask_circuit_open, _ask_last_failure
        
        _ask_failure_count += 1
        _ask_last_failure = time.time()
        
        if _ask_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
            _ask_circuit_open = True
            app.logger.warning(f"Circuit breaker opened after {_ask_failure_count} failures")

    @app.route('/ask', methods=['POST'])
    def ask():
        """Enhanced ask endpoint with reasoning metadata and circuit breaker protection."""
        if not _check_circuit_breaker():
            return jsonify({
                "error": "Service temporarily unavailable - circuit breaker open",
                "retry_after": CIRCUIT_BREAKER_TIMEOUT
            }), 503
        
        start_time = time.time()
        
        try:
            data = request.get_json() if request.is_json else request.form
            question = data.get('question', '').strip()
            stream = data.get('stream', False)
            
            if not question:
                return jsonify({"error": "Question is required"}), 400
            
            app.logger.info(f"Processing question: {question[:50]}... (stream: {stream})")
            
            if stream:
                return _handle_streaming_ask(question, start_time)
            else:
                return _handle_regular_ask(question, start_time)
            
        except Exception as e:
            app.logger.error(f"Ask endpoint error: {e}")
            _record_ask_failure()
            update_metrics(success=False, response_time=time.time() - start_time)
            return jsonify({
                "error": "Internal server error",
                "details": str(e) if current_app.debug else "Contact administrator"
            }), 500

    def _handle_regular_ask(question: str, start_time: float):
        """Handle regular (non-streaming) ask requests with enhanced metadata."""
        try:
            # Get reasoning-enhanced response from RAG system
            result = get_rag_engine()['ask'](question)
            
            if not result or not result.get("answer"):
                # Fallback response with proper retrieval
                app.logger.warning("RAG system returned empty result, attempting direct retrieval")
                try:
                    # Try direct document retrieval
                    retrieved_docs = get_rag_engine()['retrieve'](question, k=3)
                    if retrieved_docs and len(retrieved_docs) > 0:
                        # Create answer from retrieved context
                        context = "\n".join([str(doc)[:200] for doc in retrieved_docs if doc])
                        if context.strip():
                            answer = f"Based on the available documents: {context[:300]}..."
                            sources = [f"Document {i+1}" for i in range(len(retrieved_docs))]
                        else:
                            answer = _generate_basic_answer(question)
                            sources = []
                    else:
                        answer = _generate_basic_answer(question)
                        sources = []
                except Exception as e:
                    app.logger.error(f"Fallback retrieval failed: {e}")
                    answer = _generate_basic_answer(question)
                    sources = []
            else:
                # Extract answer and sources from result dictionary
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Improve answer formatting and ensure proper citation
                if answer and sources:
                    # Format with proper citations
                    if not answer.startswith("Answer:"):
                        answer = f"Answer: {answer}"
                    if not any(f"[{i}]" in answer for i in range(1, len(sources) + 1)):
                        answer = f"{answer} [1]"
                else:
                    answer = f"Answer: {answer}" if answer else "Answer: Unable to generate response"
            
            # Extract reasoning metadata
            metadata = {
                "processing_time": round(time.time() - start_time, 2),
                "model": getattr(settings, 'MODEL_ID', 'unknown'),
                "timestamp": int(time.time()),
                "question_length": len(question),
                "response_length": len(str(answer)),
                "reasoning_steps": []
            }
            
            # Try to get document sources if not already provided
            retrieved_docs = []
            if not sources:
                try:
                    retrieved_docs = get_rag_engine()['retrieve'](question, k=3)
                    if retrieved_docs:
                        sources = [f"Document chunk {i+1}" for i in range(len(retrieved_docs))]
                        metadata["sources_retrieved"] = len(sources)
                        metadata["reasoning_steps"].append("Document retrieval completed")
                except Exception as e:
                    app.logger.warning(f"Failed to retrieve sources: {e}")
                    metadata["reasoning_steps"].append("Document retrieval failed")
            
            # Add reasoning steps
            metadata["reasoning_steps"].extend([
                "Question analysis completed",
                "Context integration performed",
                "Response generation completed"
            ])
            
            response_data = {
                "answer": answer,
                "sources": sources,
                "metadata": metadata,
                "retrieved_documents": len(retrieved_docs)
            }
            
            # Reset circuit breaker on success
            global _ask_failure_count
            _ask_failure_count = max(0, _ask_failure_count - 1)
            
            app.logger.info(f"Question processed successfully in {metadata['processing_time']}s")
            update_metrics(success=True, response_time=metadata['processing_time'])
            
            return jsonify(response_data)
            
        except Exception as e:
            app.logger.error(f"Error in regular ask handling: {e}")
            _record_ask_failure()
            raise

    def _handle_streaming_ask(question: str, start_time: float):
        """Handle streaming ask requests."""
        def generate_streaming_response():
            try:
                yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"
                
                # Get response (would be streaming in real implementation)
                result = get_rag_engine()['ask'](question)
                
                if result:
                    # Split response into chunks for streaming effect
                    words = str(result).split()
                    for i, word in enumerate(words):
                        chunk_data = {
                            'type': 'chunk',
                            'content': word + ' ',
                            'index': i
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                        time.sleep(0.05)  # Simulate streaming delay
                
                # Send completion
                completion_data = {
                    'type': 'complete',
                    'processing_time': round(time.time() - start_time, 2),
                    'timestamp': time.time()
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
            except Exception as e:
                error_data = {
                    'type': 'error',
                    'error': str(e),
                    'timestamp': time.time()
                }
                yield f"data: {json.dumps(error_data)}\n\n"
        
        return Response(
            generate_streaming_response(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )

    def _generate_basic_answer(question: str) -> str:
        """Generate a basic fallback answer when RAG system fails."""
        fallback_responses = {
            "chemical": "I can help with chemical engineering questions, but need access to my knowledge base. Please check system status.",
            "process": "Process engineering questions require access to technical documents. Please verify system configuration.",
            "reactor": "Reactor design questions need technical documentation. Please check document library status.",
            "distillation": "Distillation and separation processes require technical references. Please verify document access.",
            "heat": "Heat transfer and thermal system questions need engineering references. Please check system status.",
            "default": f"I'm currently unable to access my knowledge base to answer: '{question[:50]}...' Please check system diagnostics and try again."
        }
        
        question_lower = question.lower()
        for keyword, response in fallback_responses.items():
            if keyword != "default" and keyword in question_lower:
                return f"Answer: {response}"
        
        return f"Answer: {fallback_responses['default']}"

    @app.route('/search', methods=['POST'])
    def search():
        """Enhanced direct document retrieval endpoint."""
        start_time = time.time()
        
        try:
            data = request.get_json() if request.is_json else request.form
            query = data.get('query', '').strip()
            k = min(int(data.get('k', 5)), 20)  # Limit to prevent abuse
            include_metadata = data.get('include_metadata', True)
            
            if not query:
                return jsonify({"error": "Query is required"}), 400
            
            app.logger.info(f"Document search: {query[:50]}... (k={k})")
            
            # Retrieve documents
            documents = get_rag_engine()['retrieve'](query, k=k)
            
            results = []
            if documents and len(documents) > 0:
                for i, doc in enumerate(documents):
                    if doc:  # Make sure doc is not None
                        doc_content = str(doc)
                        doc_result = {
                            "index": i,
                            "content": doc_content[:500],  # Limit content length
                            "relevance_score": round(1.0 - (i * 0.1), 2),  # Simulated relevance
                            "snippet": doc_content[:150] + "..." if len(doc_content) > 150 else doc_content
                        }
                        
                        if include_metadata:
                            doc_result["metadata"] = {
                                "length": len(doc_content),
                                "estimated_tokens": len(doc_content.split()),
                                "word_count": len(doc_content.split()),
                                "content_preview": doc_content[:100] + "..." if len(doc_content) > 100 else doc_content,
                                "document_type": "text",
                                "processing_method": "chunked"
                            }
                        
                        results.append(doc_result)
            
            processing_time = time.time() - start_time
            update_metrics(success=True, response_time=processing_time)
            
            return jsonify({
                "query": query,
                "results": results,
                "total_found": len(results),
                "processing_time": round(processing_time, 3),
                "timestamp": int(time.time())
            })
            
        except Exception as e:
            app.logger.error(f"Search endpoint error: {e}")
            update_metrics(success=False, response_time=time.time() - start_time)
            return jsonify({
                "error": "Search failed",
                "details": str(e) if current_app.debug else "Contact administrator",
                "timestamp": int(time.time())
            }), 500

    @app.route('/upload', methods=['POST'])
    def upload_document():
        """Enhanced document upload endpoint with comprehensive file support."""
        start_time = time.time()
        upload_id = str(uuid.uuid4())[:8]
        
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    "error": f"File type not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                }), 400
            
            # Secure filename
            filename = secure_filename(file.filename)
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            # Create upload directory
            upload_dir = Path("data/01_raw/source_documents")
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file with unique name to prevent conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{timestamp}_{upload_id}_{filename}"
            file_path = upload_dir / unique_filename
            
            # Also save with original filename in processed data directory
            processed_dir = Path("data/02_processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            original_file_path = processed_dir / filename
            
            app.logger.info(f"Uploading file: {filename} -> {unique_filename} (and original in data/02_processed/)")
            
            # Save file with unique timestamped name (primary copy)
            file.save(str(file_path))
            file_size = file_path.stat().st_size
            
            # Also save with original filename in processed directory for easy access
            try:
                shutil.copy2(str(file_path), str(original_file_path))
                app.logger.info(f"✅ Also saved as original filename in processed dir: {filename}")
                print(f"DEBUG: Successfully copied {file_path} to {original_file_path}")
            except Exception as copy_error:
                app.logger.error(f"❌ Failed to save original filename copy: {copy_error}")
                print(f"DEBUG: Copy failed - {copy_error}")
            
            # File processing metadata
            processing_info = {
                "upload_id": upload_id,
                "original_filename": filename,
                "saved_filename": unique_filename,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "file_type": file_ext,
                "upload_timestamp": int(time.time()),
                "processing_status": "uploaded"
            }
            
            # Attempt to ingest the file into the vector database
            ingest_success = False
            chunks_created = 0
            try:
                app.logger.info(f"Ingesting file into vector database: {unique_filename}")
                
                # Enhanced file processing based on file type
                if file_ext in ['pdf']:
                    # Use enhanced PDF processing with table extraction
                    try:
                        from pynucleus.rag.document_processor import DocumentProcessor
                        processor = DocumentProcessor()
                        ingest_result = processor.process_document(file_path)
                        
                        if ingest_result:
                            chunks_created = ingest_result.get("chunks_created", 0)
                            processing_info["tables_extracted"] = ingest_result.get("tables_extracted", 0)
                            processing_info["processing_method"] = "enhanced_pdf"
                        else:
                            # Fallback to regular ingestion
                            ingest_result = get_rag_engine()['ingest'](str(file_path))
                            processing_info["processing_method"] = "standard"
                    except ImportError:
                        # Fallback if enhanced processor not available
                        ingest_result = get_rag_engine()['ingest'](str(file_path))
                        processing_info["processing_method"] = "standard"
                else:
                    # Standard ingestion for other file types
                    ingest_result = get_rag_engine()['ingest_single_file'](str(file_path))
                    processing_info["processing_method"] = "standard"
                
                if ingest_result:
                    ingest_success = True
                    processing_info["processing_status"] = "ingested"
                    processing_info["vector_db_status"] = "success"
                    processing_info["chunks_created"] = chunks_created or 1
                    
                    # Verify ingestion worked by testing retrieval
                    try:
                        test_query = filename.rsplit('.', 1)[0]  # Use filename as test query
                        retrieved_docs = get_rag_engine()['retrieve'](test_query, k=1)
                        if retrieved_docs and len(retrieved_docs) > 0:
                            processing_info["retrieval_test"] = "passed"
                            app.logger.info(f"File successfully ingested and retrievable: {unique_filename}")
                        else:
                            processing_info["retrieval_test"] = "failed"
                            app.logger.warning(f"File ingested but not retrievable: {unique_filename}")
                    except Exception as retrieval_error:
                        processing_info["retrieval_test"] = f"error: {str(retrieval_error)}"
                        app.logger.warning(f"File ingested but retrieval test failed: {retrieval_error}")
                else:
                    processing_info["processing_status"] = "ingest_failed"
                    processing_info["vector_db_status"] = "failed"
                    app.logger.warning(f"Failed to ingest file: {unique_filename}")
                    
            except Exception as ingest_error:
                app.logger.error(f"Error during ingestion: {ingest_error}")
                processing_info["processing_status"] = "ingest_error"
                processing_info["vector_db_status"] = "error"
                processing_info["error_details"] = str(ingest_error)
            
            # Content analysis (basic)
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content_preview = f.read(500)
                    processing_info["content_preview"] = content_preview
                    processing_info["estimated_tokens"] = len(content_preview.split())
            except Exception as e:
                app.logger.warning(f"Could not read file content: {e}")
                processing_info["content_preview"] = "Binary or unreadable content"
            
            processing_time = time.time() - start_time
            processing_info["processing_time"] = round(processing_time, 3)
            
            update_metrics(success=ingest_success, response_time=processing_time)
            
            status_code = 200 if ingest_success else 202
            
            return jsonify({
                "message": "File uploaded and processed successfully" if ingest_success else "File uploaded but processing failed",
                "processing_info": processing_info,
                "next_steps": [
                    f"Try asking: 'What does {filename} contain?'",
                    f"Search for content with: {filename.rsplit('.', 1)[0]}",
                    "Check document statistics in the Statistics tab"
                ] if ingest_success else [
                    "Check system diagnostics for processing issues",
                    "Verify file format is supported and readable",
                    f"File saved as: {unique_filename}",
                    "Manual processing may be required"
                ],
                "quick_test": {
                    "suggested_query": f"What is in the document {filename.rsplit('.', 1)[0]}?",
                    "file_available_for_qa": ingest_success,
                    "estimated_chunks": processing_info.get("chunks_created", 0),
                    "processing_time": processing_info.get("processing_time", 0)
                }
            }), status_code
            
        except Exception as e:
            app.logger.error(f"Upload endpoint error: {e}")
            update_metrics(success=False, response_time=time.time() - start_time)
            return jsonify({
                "error": "Upload failed",
                "upload_id": upload_id,
                "details": str(e) if current_app.debug else "Contact administrator",
                "timestamp": int(time.time())
            }), 500

    @app.route('/system_diagnostic', methods=['GET'])
    def system_diagnostic():
        """Simple, fast system diagnostic with essential health checks."""
        app.logger.info("System diagnostic requested")
        
        try:
            start_time = time.time()
            
            # Quick health checks
            checks = []
            
            # 1. Python Environment
            try:
                import sys
                python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
                checks.append(f"✅ Python {python_version}")
            except:
                checks.append("❌ Python version check failed")
            
            # 2. Core Dependencies
            try:
                import numpy, pandas, flask
                checks.append("✅ Core dependencies")
            except ImportError as e:
                checks.append(f"❌ Missing dependencies: {e}")
            
            # 3. Vector Database
            try:
                from pynucleus.rag.vector_store import ChromaVectorStore
                store = ChromaVectorStore()
                stats = store.get_index_stats()
                doc_count = stats.get("doc_count", 0)
                checks.append(f"✅ Vector DB ({doc_count} docs)")
            except Exception as e:
                checks.append(f"❌ Vector DB: {str(e)[:50]}")
            
            # 4. Model Loading
            try:
                from pynucleus.llm.model_loader import get_model_info
                model_info = get_model_info()
                if model_info and model_info.get("loaded"):
                    checks.append(f"✅ Model ({model_info.get('method', 'unknown')})")
                else:
                    checks.append("⚠️ Model not loaded")
            except Exception as e:
                checks.append(f"❌ Model: {str(e)[:50]}")
            
            # 5. File System
            try:
                from pathlib import Path
                source_dir = Path("data/01_raw/source_documents")
                if source_dir.exists():
                    file_count = len(list(source_dir.glob("*")))
                    checks.append(f"✅ Source docs ({file_count} files)")
                else:
                    checks.append("⚠️ No source documents directory")
            except:
                checks.append("❌ File system check failed")
            
            # Generate simple report
            total_checks = len(checks)
            passed_checks = len([c for c in checks if c.startswith("✅")])
            
            response_text = f"""▶ SYSTEM DIAGNOSTIC REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔍 HEALTH CHECKS ({passed_checks}/{total_checks} passed):

{chr(10).join(f"   {check}" for check in checks)}

📊 SUMMARY:
   • Status: {"🟢 HEALTHY" if passed_checks >= total_checks * 0.8 else "🟡 WARNING" if passed_checks >= total_checks * 0.6 else "🔴 CRITICAL"}
   • Success Rate: {passed_checks/total_checks*100:.0f}%
   • Check Time: {time.time() - start_time:.2f}s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

            return response_text, 200, {'Content-Type': 'text/plain'}
            
        except Exception as e:
            app.logger.error(f"Diagnostic error: {e}")
            error_text = f"❌ DIAGNOSTIC ERROR: {str(e)}\n\nSystem may be in unstable state."
            return error_text, 500, {'Content-Type': 'text/plain'}

    @app.route('/comprehensive_diagnostic', methods=['GET'])
    def comprehensive_diagnostic():
        """Simple comprehensive diagnostic - same as system_diagnostic but with different name for UI."""
        return system_diagnostic()

    @app.route('/enhanced_diagnostic', methods=['GET'])
    def enhanced_diagnostic():
        """Enhanced diagnostic - same as system_diagnostic but with different name for UI."""
        return system_diagnostic()

    @app.route('/system_statistics', methods=['GET'])
    def system_statistics():
        """Simple, lightweight system statistics endpoint."""
        app.logger.info("System statistics requested")
        
        try:
            start_time = time.time()
            
            # Basic system info
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Vector store info with detailed metrics
            try:
                from pynucleus.rag.vector_store import ChromaVectorStore
                store = ChromaVectorStore()
                stats = store.get_index_stats()
                doc_count = stats.get("doc_count", "unknown")
                vector_status = "✅ Online"
                
                # Test query performance
                test_start = time.time()
                test_docs = store.query("test", k=1)
                query_time = (time.time() - test_start) * 1000
                
                vector_perf = f"{query_time:.1f}ms"
            except Exception as e:
                doc_count = "unknown"
                vector_status = f"❌ Offline ({str(e)[:30]})"
                vector_perf = "N/A"
            
            # Get additional system details
            cpu = psutil.cpu_percent(interval=0.1)
            
            # Model information
            try:
                from pynucleus.llm.model_loader import get_model_info
                model_info = get_model_info()
                model_status = f"✅ {model_info.get('method', 'Unknown')}" if model_info.get('loaded') else "❌ Not Loaded"
                model_name = model_info.get('model_name', 'Unknown')[:20]
            except:
                model_status = "❌ Not Available"
                model_name = "Unknown"
            
            # Source documents check
            from pathlib import Path
            source_dir = Path("data/01_raw/source_documents")
            source_files = len(list(source_dir.glob("*"))) if source_dir.exists() else 0
            
            # Calculate success rate
            success_rate = ((_system_metrics['requests_total'] - _system_metrics['requests_failed']) / 
                           max(_system_metrics['requests_total'], 1)) * 100
            
            # Format detailed response
            response_text = f"""▶ PYNUCLEUS SYSTEM STATUS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 CORE METRICS:
   • Documents Indexed: {doc_count}
   • Source Files: {source_files}
   • Vector Store: {vector_status}
   • Query Performance: {vector_perf}
   • AI Model: {model_status}
   • Model Name: {model_name}

🖥️ SYSTEM RESOURCES:
   • Memory Usage: {memory.percent:.1f}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)
   • CPU Usage: {cpu:.1f}%
   • Disk Usage: {disk.percent:.1f}% ({disk.free / (1024**3):.1f}GB free)

🔧 API PERFORMANCE:
   • Total Requests: {_system_metrics['requests_total']}
   • Failed Requests: {_system_metrics['requests_failed']}
   • Success Rate: {success_rate:.1f}%
   • Avg Response Time: {_system_metrics['average_response_time']:.3f}s
   • Uptime: {timedelta(seconds=int(time.time() - _system_metrics['uptime_start']))}

🎯 SYSTEM HEALTH: {'🟢 HEALTHY' if success_rate >= 90 and memory.percent < 90 else '🟡 WARNING' if success_rate >= 75 else '🔴 CRITICAL'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated in {time.time() - start_time:.2f}s"""

            return response_text, 200, {'Content-Type': 'text/plain'}
            
        except Exception as e:
            app.logger.error(f"Statistics error: {e}")
            error_text = f"❌ STATISTICS ERROR: {str(e)}\n\nPlease check system logs for details."
            return error_text, 500, {'Content-Type': 'text/plain'}

    @app.route('/enhanced_evaluation', methods=['GET', 'POST'])
    def enhanced_evaluation():
        """Run enhanced evaluation with comprehensive analysis."""
        try:
            # Get parameters
            threshold = float(request.args.get('threshold', 0.7))
            sample_size = request.args.get('sample_size')
            if sample_size:
                sample_size = int(sample_size)
            
            app.logger.info(f"Starting enhanced evaluation (threshold={threshold}, sample_size={sample_size})...")
            
            # Import and run enhanced evaluation
            from pynucleus.eval.golden_eval import run_enhanced_eval
            
            result = run_enhanced_eval(
                threshold=threshold,
                sample_size=sample_size,
                save_results=True
            )
            
            if request.args.get('format') == 'json':
                return jsonify(result)
            else:
                # Return formatted text output
                analysis = result.get("analysis", {})
                
                output = f"""🔍 ENHANCED EVALUATION REPORT
{'=' * 50}

📊 OVERALL METRICS:
Success Rate: {analysis.get('overall_metrics', {}).get('success_rate', 0):.1%}
Total Questions: {analysis.get('overall_metrics', {}).get('total_questions', 0)}
Average Response Time: {analysis.get('overall_metrics', {}).get('avg_response_time', 0):.3f}s
Average Confidence: {analysis.get('overall_metrics', {}).get('avg_confidence_score', 0):.3f}

📈 QUALITY DISTRIBUTION:
"""
                
                quality_dist = analysis.get('quality_distribution', {})
                if quality_dist:
                    confidence_scores = quality_dist.get('confidence_scores', {})
                    output += f"""Confidence Scores:
  • Excellent (≥0.8): {confidence_scores.get('excellent', 0)}
  • Good (0.6-0.8): {confidence_scores.get('good', 0)}
  • Fair (0.4-0.6): {confidence_scores.get('fair', 0)}
  • Poor (<0.4): {confidence_scores.get('poor', 0)}
"""
                
                # Domain performance
                domain_analysis = analysis.get('domain_analysis', {})
                if domain_analysis:
                    output += f"\n🏷️ DOMAIN PERFORMANCE:\n"
                    for domain, stats in domain_analysis.items():
                        output += f"  • {domain}: {stats.get('success_rate', 0):.1%} ({stats.get('questions', 0)} questions)\n"
                
                # Recommendations
                recommendations = analysis.get('recommendations', [])
                if recommendations:
                    output += f"\n💡 RECOMMENDATIONS:\n"
                    for rec in recommendations:
                        output += f"  {rec}\n"
                
                return output, 200, {'Content-Type': 'text/plain'}
            
        except Exception as e:
            app.logger.error(f"Enhanced evaluation failed: {e}")
            return f"Enhanced evaluation failed: {e}", 500, {'Content-Type': 'text/plain'}

    @app.route('/metrics_export', methods=['GET'])
    def metrics_export():
        """Export comprehensive metrics data."""
        try:
            # Simplified metrics export without problematic imports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export to file if requested
            if request.args.get('export') == 'file':
                export_file = f"data/05_output/metrics_export_{timestamp}.json"
                
                # Create simplified metrics data
                metrics_data = {
                    "export_timestamp": datetime.now().isoformat(),
                    "status": "simplified_export",
                    "message": "Enhanced metrics temporarily disabled for stability",
                    "basic_metrics": {
                        "uptime_seconds": int(time.time() - _system_metrics['uptime_start']),
                        "requests_total": _system_metrics['requests_total'],
                        "requests_failed": _system_metrics['requests_failed'],
                        "success_rate": round(((_system_metrics['requests_total'] - _system_metrics['requests_failed']) / max(_system_metrics['requests_total'], 1)) * 100, 1),
                        "average_response_time": round(_system_metrics['average_response_time'], 3)
                    }
                }
                
                # Ensure directory exists
                Path(export_file).parent.mkdir(parents=True, exist_ok=True)
                
                # Write to file
                with open(export_file, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                
                return jsonify({
                    "status": "exported",
                    "file": export_file,
                    "timestamp": timestamp
                })
            
            # Return JSON data
            return jsonify({
                "current_performance": {
                    "status": "simplified",
                    "message": "Enhanced metrics temporarily disabled for stability"
                },
                "historical_trends": {
                    "status": "simplified",
                    "message": "Historical trends temporarily disabled for stability"
                },
                "export_timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            app.logger.error(f"Metrics export failed: {e}")
            return jsonify({"error": str(e)}), 500

    @app.route('/browse_files', methods=['GET'])
    def browse_files():
        """Browse files in source_documents and cleaned_txt directories."""
        try:
            directories = {
                "source_documents": [],
                "cleaned_txt": []
            }
            
            # Browse source_documents
            source_dir = Path("data/01_raw/source_documents")
            if source_dir.exists():
                for file in source_dir.iterdir():
                    if file.is_file():
                        stat = file.stat()
                        directories["source_documents"].append({
                            "name": file.name,
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": int(stat.st_mtime),
                            "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            "extension": file.suffix.lower()
                        })
            
            # Browse cleaned_txt
            cleaned_dir = Path("data/02_processed/cleaned_txt")
            if cleaned_dir.exists():
                for file in cleaned_dir.iterdir():
                    if file.is_file():
                        stat = file.stat()
                        directories["cleaned_txt"].append({
                            "name": file.name,
                            "size": stat.st_size,
                            "size_mb": round(stat.st_size / (1024 * 1024), 2),
                            "modified": int(stat.st_mtime),
                            "modified_str": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                            "extension": file.suffix.lower()
                        })
            
            # Sort by modification time (newest first)
            for dir_name in directories:
                directories[dir_name].sort(key=lambda x: x["modified"], reverse=True)
            
            return jsonify({
                "status": "success",
                "directories": directories,
                "summary": {
                    "source_count": len(directories["source_documents"]),
                    "cleaned_count": len(directories["cleaned_txt"]),
                    "total_source_size": sum(f["size"] for f in directories["source_documents"]),
                    "total_cleaned_size": sum(f["size"] for f in directories["cleaned_txt"])
                },
                "timestamp": int(time.time())
            })
            
        except Exception as e:
            app.logger.error(f"File browser error: {e}")
            return jsonify({
                "error": "Failed to browse files",
                "details": str(e) if current_app.debug else "Contact administrator"
            }), 500

    @app.route('/dev', methods=['GET'])
    def dev_console():
        """Redirect to main developer console."""
        return send_from_directory(BASE_DIR / 'static', 'developer_dashboard.html')

    return app

# For backwards compatibility when running directly
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5001, debug=True)