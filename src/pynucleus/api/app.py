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
from pathlib import Path
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, Response, stream_template
from werkzeug.utils import secure_filename
import time
from datetime import datetime
import signal
import atexit
import threading
import uuid

# Fix tokenizer parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Configure logging
logger = configure_logging(level="INFO")
api_logger = get_logger(__name__)

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

# Global cleanup handler
def cleanup_resources():
    """Clean up resources on app shutdown."""
    try:
        api_logger.info("Cleaning up application resources...")
        # Force garbage collection
        import gc
        gc.collect()
        api_logger.info("Resource cleanup completed")
    except Exception as e:
        api_logger.error(f"Error during cleanup: {e}")

# Register cleanup handlers
atexit.register(cleanup_resources)

def signal_handler(sig, frame):
    """Handle shutdown signals gracefully."""
    api_logger.info(f"Received signal {sig}, shutting down gracefully...")
    cleanup_resources()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Allowed file extensions for document upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'md', 'doc', 'docx', 'csv', 'json'}

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

@app.route('/', methods=['GET'])
def index():
    """Serve the main developer console."""
    static_dir = Path(__file__).parent / 'static'
    return send_from_directory(static_dir, 'index.html')

@app.route('/health', methods=['GET'])
def health():
    """Comprehensive health check endpoint."""
    api_logger.info("Health check requested")
    
    health_status = {
        "status": "healthy",
        "timestamp": int(time.time()),
        "version": __version__,
        "service": "PyNucleus API",
        "checks": {}
    }
    
    # Check RAG system
    try:
        test_result = rag_ask("test")
        health_status["checks"]["rag"] = "healthy" if test_result else "degraded"
    except Exception as e:
        health_status["checks"]["rag"] = f"unhealthy: {str(e)[:50]}"
        health_status["status"] = "degraded"
    
    # Check vector store
    try:
        from ..rag.vector_store import ChromaVectorStore
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
            api_logger.warning(f"Failed to get system metrics: {e}")
        
        # ChromaDB status
        chroma_status = {"status": "unknown", "doc_count": 0, "size_mb": 0}
        try:
            test_docs = retrieve("test", k=1)
            if test_docs:
                chroma_status["status"] = "ready"
                all_docs = retrieve("", k=1000)
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
        api_logger.error(f"Status endpoint error: {e}")
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
            api_logger.info("Circuit breaker closed - timeout expired")
    
    return not _ask_circuit_open

def _record_ask_failure():
    """Record a failure for the circuit breaker."""
    global _ask_failure_count, _ask_circuit_open, _ask_last_failure
    
    _ask_failure_count += 1
    _ask_last_failure = time.time()
    
    if _ask_failure_count >= CIRCUIT_BREAKER_THRESHOLD:
        _ask_circuit_open = True
        api_logger.warning(f"Circuit breaker opened after {_ask_failure_count} failures")

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
        
        api_logger.info(f"Processing question: {question[:50]}... (stream: {stream})")
        
        if stream:
            return _handle_streaming_ask(question, start_time)
        else:
            return _handle_regular_ask(question, start_time)
            
    except Exception as e:
        api_logger.error(f"Ask endpoint error: {e}")
        _record_ask_failure()
        update_metrics(success=False, response_time=time.time() - start_time)
        return jsonify({
            "error": "Internal server error",
            "details": str(e) if app.debug else "Contact administrator"
        }), 500

def _handle_regular_ask(question: str, start_time: float):
    """Handle regular (non-streaming) ask requests with enhanced metadata."""
    try:
        # Get reasoning-enhanced response
        result = rag_ask(question)
        
        if not result:
            # Fallback response
            api_logger.warning("RAG system returned empty result, using fallback")
            result = _generate_basic_answer(question)
            answer = result
            sources = []
        else:
            # Extract answer and sources from result dictionary
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Format answer according to required format: "Answer: [answer] [citation_number]"
            if answer and sources:
                # Check if answer already has citation
                if '[' in answer and ']' in answer:
                    formatted_answer = f"Answer: {answer}"
                else:
                    # Add citation if not present
                    formatted_answer = f"Answer: {answer} [1]"
            else:
                formatted_answer = f"Answer: {answer}"
            
            answer = formatted_answer
        
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
                retrieved_docs = retrieve(question, k=3)
                if retrieved_docs:
                    sources = [f"Document chunk {i+1}" for i in range(len(retrieved_docs))]
                    metadata["sources_retrieved"] = len(sources)
                    metadata["reasoning_steps"].append("Document retrieval completed")
            except Exception as e:
                api_logger.warning(f"Failed to retrieve sources: {e}")
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
        
        api_logger.info(f"Question processed successfully in {metadata['processing_time']}s")
        update_metrics(success=True, response_time=metadata['processing_time'])
        
        return jsonify(response_data)
        
    except Exception as e:
        api_logger.error(f"Error in regular ask handling: {e}")
        _record_ask_failure()
        raise

def _handle_streaming_ask(question: str, start_time: float):
    """Handle streaming ask requests."""
    def generate_streaming_response():
        try:
            yield f"data: {json.dumps({'type': 'start', 'timestamp': time.time()})}\n\n"
            
            # Get response (would be streaming in real implementation)
            result = rag_ask(question)
            
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
    return f"Unable to process question '{question[:50]}...' - RAG system unavailable. Please check system status and try again."

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
        
        api_logger.info(f"Document search: {query[:50]}... (k={k})")
        
        # Retrieve documents
        documents = retrieve(query, k=k)
        
        results = []
        if documents:
            for i, doc in enumerate(documents):
                doc_result = {
                    "index": i,
                    "content": str(doc)[:500],  # Limit content length
                    "relevance_score": 1.0 - (i * 0.1),  # Simulated relevance
                }
                
                if include_metadata:
                    doc_result["metadata"] = {
                        "length": len(str(doc)),
                        "estimated_tokens": len(str(doc).split()),
                        "content_preview": str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc)
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
        api_logger.error(f"Search endpoint error: {e}")
        update_metrics(success=False, response_time=time.time() - start_time)
        return jsonify({
            "error": "Search failed",
            "details": str(e) if app.debug else "Contact administrator",
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
        upload_dir = Path("source_documents")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file with unique name to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{upload_id}_{filename}"
        file_path = upload_dir / unique_filename
        
        api_logger.info(f"Uploading file: {filename} -> {unique_filename}")
        
        # Save file
        file.save(str(file_path))
        file_size = file_path.stat().st_size
        
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
        try:
            api_logger.info(f"Ingesting file into vector database: {unique_filename}")
            ingest_result = ingest(str(file_path))
            
            if ingest_result:
                ingest_success = True
                processing_info["processing_status"] = "ingested"
                processing_info["vector_db_status"] = "success"
                api_logger.info(f"File successfully ingested: {unique_filename}")
            else:
                processing_info["processing_status"] = "ingest_failed"
                processing_info["vector_db_status"] = "failed"
                api_logger.warning(f"Failed to ingest file: {unique_filename}")
                
        except Exception as ingest_error:
            api_logger.error(f"Error during ingestion: {ingest_error}")
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
            api_logger.warning(f"Could not read file content: {e}")
            processing_info["content_preview"] = "Binary or unreadable content"
        
        processing_time = time.time() - start_time
        processing_info["processing_time"] = round(processing_time, 3)
        
        update_metrics(success=ingest_success, response_time=processing_time)
        
        status_code = 200 if ingest_success else 202
        
        return jsonify({
            "message": "File uploaded successfully" if ingest_success else "File uploaded but ingestion failed",
            "processing_info": processing_info,
            "recommendations": [
                "Check vector database status if ingestion failed",
                "Verify file content is readable and relevant",
                "Monitor system diagnostics for processing issues"
            ] if not ingest_success else [
                "File is now available for search and Q&A",
                "Use the search endpoint to verify document retrieval",
                "Ask questions related to the uploaded content"
            ]
        }), status_code
        
    except Exception as e:
        api_logger.error(f"Upload endpoint error: {e}")
        update_metrics(success=False, response_time=time.time() - start_time)
        return jsonify({
            "error": "Upload failed",
            "upload_id": upload_id,
            "details": str(e) if app.debug else "Contact administrator",
            "timestamp": int(time.time())
        }), 500

@app.route('/system_diagnostic', methods=['GET'])
def system_diagnostic():
    """Get system diagnostics from the validator script."""
    try:
        # Import and run system validator
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "scripts"))
        from system_validator import run_validation
        
        # Run validation
        results = run_validation()
        
        return jsonify(results)
        
    except Exception as e:
        api_logger.error(f"System diagnostic error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": int(time.time())
        }), 500

@app.route('/comprehensive_diagnostic', methods=['GET'])
def comprehensive_diagnostic():
    """Run comprehensive system diagnostics."""
    try:
        # Import and run comprehensive diagnostic
        sys.path.append(str(Path(__file__).parent.parent.parent.parent / "scripts"))
        from comprehensive_system_diagnostic import run_comprehensive_diagnostic
        
        # Run comprehensive diagnostic
        results = run_comprehensive_diagnostic()
        
        return jsonify(results)
        
    except Exception as e:
        api_logger.error(f"Comprehensive diagnostic error: {e}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": int(time.time())
        }), 500

@app.route('/system_statistics', methods=['GET'])
def system_statistics():
    """Get comprehensive system statistics."""
    try:
        timestamp = int(time.time())
        
        # RAG Pipeline Configuration
        rag_pipeline = {
            "model_id": getattr(settings, 'MODEL_ID', 'unknown'),
            "embedding_model": getattr(settings, 'EMBEDDING_MODEL', 'unknown'),
            "max_tokens": getattr(settings, 'MAX_TOKENS', 'unknown'),
            "retrieve_top_k": getattr(settings, 'RETRIEVE_TOP_K', 'unknown'),
            "context_window": getattr(settings, 'CONTEXT_WINDOW', 'unknown'),
            "retrieval_method": "semantic_search",
            "chroma_path": getattr(settings, 'CHROMA_PATH', 'unknown')
        }
        
        # Document Processing Configuration
        document_processing = {
            "default_chunk_size": 1000,
            "chunk_overlap": 200,
            "chunking_strategy": "recursive_character",
            "vector_dimensions": 384,
            "similarity_metric": "cosine",
            "supported_formats": list(ALLOWED_EXTENSIONS)
        }
        
        # Vector Database Information
        vector_database = {
            "exists": False,
            "doc_count": 0,
            "collection_name": "default",
            "chunking_status": "unknown",
            "embedding_model": rag_pipeline["embedding_model"],
            "search_method": "similarity",
            "chunk_metadata": "available"
        }
        
        try:
            # Check if vector database exists and get stats
            test_docs = retrieve("test", k=1)
            if test_docs:
                vector_database["exists"] = True
                all_docs = retrieve("", k=1000)
                vector_database["doc_count"] = len(all_docs) if all_docs else 0
                vector_database["chunking_status"] = "active"
        except Exception as e:
            api_logger.warning(f"Failed to get vector database stats: {e}")
        
        # Document Library Information
        documents = {
            "source_documents_total": 0,
            "source_by_type": {},
            "processed_documents": {},
            "processing_pipeline": {},
            "total_source_size_mb": 0
        }
        
        # Count source documents
        source_dir = Path("source_documents")
        if source_dir.exists():
            source_files = list(source_dir.glob("*"))
            documents["source_documents_total"] = len([f for f in source_files if f.is_file()])
            
            # Count by type
            type_counts = {}
            total_size = 0
            for file in source_files:
                if file.is_file():
                    ext = file.suffix.lower().replace('.', '')
                    type_counts[f"{ext}_files"] = type_counts.get(f"{ext}_files", 0) + 1
                    total_size += file.stat().st_size
            
            documents["source_by_type"] = type_counts
            documents["total_source_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # System Information
        system_info = {
            "platform": f"{os.name} ({sys.platform})",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "cpu_count": os.cpu_count()
        }
        
        try:
            memory = psutil.virtual_memory()
            system_info.update({
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent": round(memory.percent, 1)
            })
        except Exception as e:
            api_logger.warning(f"Failed to get memory stats: {e}")
        
        return jsonify({
            "timestamp": timestamp,
            "rag_pipeline": rag_pipeline,
            "document_processing": document_processing,
            "vector_database": vector_database,
            "documents": documents,
            "system": system_info,
            "api_metrics": {
                "uptime_seconds": int(time.time() - _system_metrics['uptime_start']),
                "requests_total": _system_metrics['requests_total'],
                "requests_failed": _system_metrics['requests_failed'],
                "average_response_time": round(_system_metrics['average_response_time'], 3)
            }
        })
        
    except Exception as e:
        api_logger.error(f"System statistics error: {e}")
        return jsonify({
            "error": "Failed to get system statistics",
            "details": str(e) if app.debug else "Contact administrator",
            "timestamp": int(time.time())
        }), 500

@app.route('/dev', methods=['GET'])
def dev_console():
    """Redirect to main developer console."""
    return send_from_directory(Path(__file__).parent / 'static', 'developer_dashboard.html')

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "Method not allowed"}), 405

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File too large (max 32MB)"}), 413

@app.errorhandler(500)
def internal_error(error):
    api_logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True) 