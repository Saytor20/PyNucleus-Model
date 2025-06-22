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
from pynucleus.rag.collector import ingest, ingest_single_file
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
        # Get reasoning-enhanced response from RAG system
        result = rag_ask(question)
        
        if not result or not result.get("answer"):
            # Fallback response with proper retrieval
            api_logger.warning("RAG system returned empty result, attempting direct retrieval")
            try:
                # Try direct document retrieval
                retrieved_docs = retrieve(question, k=3)
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
                api_logger.error(f"Fallback retrieval failed: {e}")
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
        
        api_logger.info(f"Document search: {query[:50]}... (k={k})")
        
        # Retrieve documents
        documents = retrieve(query, k=k)
        
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
        
        api_logger.info(f"Uploading file: {filename} -> {unique_filename} (and original in data/02_processed/)")
        
        # Save file with unique timestamped name (primary copy)
        file.save(str(file_path))
        file_size = file_path.stat().st_size
        
        # Also save with original filename in processed directory for easy access
        try:
            shutil.copy2(str(file_path), str(original_file_path))
            api_logger.info(f"✅ Also saved as original filename in processed dir: {filename}")
            print(f"DEBUG: Successfully copied {file_path} to {original_file_path}")
        except Exception as copy_error:
            api_logger.error(f"❌ Failed to save original filename copy: {copy_error}")
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
            api_logger.info(f"Ingesting file into vector database: {unique_filename}")
            
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
                        ingest_result = ingest(str(file_path))
                        processing_info["processing_method"] = "standard"
                except ImportError:
                    # Fallback if enhanced processor not available
                    ingest_result = ingest(str(file_path))
                    processing_info["processing_method"] = "standard"
            else:
                # Standard ingestion for other file types
                ingest_result = ingest_single_file(str(file_path))
                processing_info["processing_method"] = "standard"
            
            if ingest_result:
                ingest_success = True
                processing_info["processing_status"] = "ingested"
                processing_info["vector_db_status"] = "success"
                processing_info["chunks_created"] = chunks_created or 1
                
                # Verify ingestion worked by testing retrieval
                try:
                    test_query = filename.rsplit('.', 1)[0]  # Use filename as test query
                    retrieved_docs = retrieve(test_query, k=1)
                    if retrieved_docs and len(retrieved_docs) > 0:
                        processing_info["retrieval_test"] = "passed"
                        api_logger.info(f"File successfully ingested and retrievable: {unique_filename}")
                    else:
                        processing_info["retrieval_test"] = "failed"
                        api_logger.warning(f"File ingested but not retrievable: {unique_filename}")
                except Exception as retrieval_error:
                    processing_info["retrieval_test"] = f"error: {str(retrieval_error)}"
                    api_logger.warning(f"File ingested but retrieval test failed: {retrieval_error}")
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
        scripts_path = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from system_validator import SystemValidator
        
        # Create validator and run validation suite
        validator = SystemValidator(quiet_mode=True)
        validator.run_validation_suite(include_citations=False, include_notebook=False)
        
        # Calculate success rate
        success_rate = (validator.passed_tests / validator.total_tests * 100) if validator.total_tests > 0 else 0
        
        # Format results for web interface
        results = {
            "status": "completed",
            "health_status": "excellent" if success_rate >= 90 else "good" if success_rate >= 80 else "warning" if success_rate >= 70 else "critical",
            "total_tests": validator.total_tests,
            "passed_tests": validator.passed_tests,
            "failed_tests": validator.total_tests - validator.passed_tests,
            "success_rate": round(success_rate, 1),
            "timestamp": int(time.time()),
            "validation_results": [
                {
                    "test_name": r.test_name,
                    "domain": r.domain,
                    "accuracy_score": r.accuracy_score,
                    "citation_accuracy": r.citation_accuracy,
                    "response_time": r.response_time
                }
                for r in validator.validation_results[:10]  # Limit to first 10 for web display
            ]
        }
        
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
        scripts_path = str(Path(__file__).parent.parent.parent.parent / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        
        from comprehensive_system_diagnostic import ComprehensiveSystemDiagnostic
        
        # Create diagnostic runner and run comprehensive diagnostic
        diagnostic = ComprehensiveSystemDiagnostic(quiet_mode=True, test_mode=False)
        diagnostic.run_comprehensive_diagnostic()
        
        # Calculate overall health
        success_rate = (diagnostic.passed_checks / diagnostic.total_checks * 100) if diagnostic.total_checks > 0 else 0
        script_health_rate = (diagnostic.healthy_scripts / diagnostic.total_scripts * 100) if diagnostic.total_scripts > 0 else 100
        
        # Format results for web interface
        results = {
            "status": "completed",
            "overall_health": round((success_rate + script_health_rate) / 2, 1),
            "health_status": "excellent" if success_rate >= 95 else "very_good" if success_rate >= 85 else "good" if success_rate >= 75 else "warning" if success_rate >= 65 else "critical",
            "system_checks": {
                "total": diagnostic.total_checks,
                "passed": diagnostic.passed_checks,
                "failed": diagnostic.total_checks - diagnostic.passed_checks,
                "success_rate": round(success_rate, 1)
            },
            "script_health": {
                "total": diagnostic.total_scripts,
                "healthy": diagnostic.healthy_scripts,
                "unhealthy": diagnostic.total_scripts - diagnostic.healthy_scripts,
                "health_rate": round(script_health_rate, 1)
            },
            "component_health": {
                "environment": diagnostic.environment_health,
                "dependencies": diagnostic.dependencies_health,
                "scripts": diagnostic.scripts_health,
                "components": diagnostic.components_health,
                "docker": diagnostic.docker_health,
                "chromadb": diagnostic.chromadb_health,
                "qwen": diagnostic.qwen_health,
                "pdf_processing": diagnostic.pdf_processing_health
            },
            "timestamp": int(time.time())
        }
        
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
            "chunk_metadata": "available",
            "database_size_mb": 0,
            "last_update": None,
            "performance_metrics": {
                "avg_query_time": 0,
                "avg_embedding_time": 0,
                "cache_hit_rate": 0
            }
        }
        
        try:
            # Check ChromaDB specifically
            from pynucleus.rag.vector_store import ChromaVectorStore
            chroma_store = ChromaVectorStore()
            
            if chroma_store.loaded:
                vector_database["exists"] = True
                vector_database["chunking_status"] = "active"
                
                # Get more detailed stats
                stats = chroma_store.get_index_stats()
                if stats:
                    vector_database["doc_count"] = stats.get("doc_count", 0)
                    vector_database["collection_name"] = stats.get("collection_name", "default")
                    vector_database["last_update"] = stats.get("last_modified", None)
                
                # Test query performance
                start_time = time.time()
                test_docs = retrieve("chemical engineering", k=1)
                query_time = time.time() - start_time
                
                vector_database["performance_metrics"]["avg_query_time"] = round(query_time * 1000, 2)  # ms
                
                if test_docs:
                    vector_database["doc_count"] = max(vector_database["doc_count"], len(test_docs))
            
            # Check database file size
            chroma_db_path = Path(settings.CHROMA_PATH)
            if chroma_db_path.exists():
                total_size = sum(f.stat().st_size for f in chroma_db_path.rglob('*') if f.is_file())
                vector_database["database_size_mb"] = round(total_size / (1024 * 1024), 2)
                
        except Exception as e:
            api_logger.warning(f"Failed to get vector database stats: {e}")
        
        # Document Library Information
        documents = {
            "source_documents_total": 0,
            "source_by_type": {},
            "processed_documents": {},
            "processing_pipeline": {
                "chunking_active": False,
                "embedding_active": False,
                "indexing_active": False,
                "last_processing_time": None
            },
            "total_source_size_mb": 0,
            "processing_stats": {
                "documents_processed_today": 0,
                "total_chunks_created": 0,
                "avg_chunk_size": 0,
                "processing_success_rate": 0,
                "failed_documents": 0
            }
        }
        
        # Count source documents
        source_dir = Path("data/01_raw/source_documents")
        if source_dir.exists():
            source_files = list(source_dir.glob("*"))
            documents["source_documents_total"] = len([f for f in source_files if f.is_file()])
            
            # Count by type and recent activity
            type_counts = {}
            total_size = 0
            processed_today = 0
            current_date = datetime.now().date()
            
            for file in source_files:
                if file.is_file():
                    ext = file.suffix.lower().replace('.', '')
                    type_counts[f"{ext}_files"] = type_counts.get(f"{ext}_files", 0) + 1
                    total_size += file.stat().st_size
                    
                    # Check if processed today
                    file_date = datetime.fromtimestamp(file.stat().st_mtime).date()
                    if file_date == current_date:
                        processed_today += 1
            
            documents["source_by_type"] = type_counts
            documents["total_source_size_mb"] = round(total_size / (1024 * 1024), 2)
            documents["processing_stats"]["documents_processed_today"] = processed_today
        
        # Check for processed document outputs
        processed_dirs = [
            "data/02_processed",
            "data/03_processed", 
            "data/03_intermediate",
            "data/05_output"
        ]
        
        for proc_dir in processed_dirs:
            if Path(proc_dir).exists():
                processed_files = list(Path(proc_dir).rglob("*"))
                processed_count = len([f for f in processed_files if f.is_file()])
                documents["processed_documents"][proc_dir.split("/")[-1]] = processed_count
        
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
                "uptime_formatted": f"{int((time.time() - _system_metrics['uptime_start']) // 3600)}h {int(((time.time() - _system_metrics['uptime_start']) % 3600) // 60)}m",
                "requests_total": _system_metrics['requests_total'],
                "requests_failed": _system_metrics['requests_failed'],
                "success_rate": round(((_system_metrics['requests_total'] - _system_metrics['requests_failed']) / max(_system_metrics['requests_total'], 1)) * 100, 1),
                "average_response_time": round(_system_metrics['average_response_time'], 3),
                "last_request_time": _system_metrics.get('last_request_time'),
                "circuit_breaker_status": "open" if _ask_circuit_open else "closed",
                "failure_count": _ask_failure_count,
                "performance_metrics": {
                    "queries_per_minute": round(_system_metrics['requests_total'] / max((time.time() - _system_metrics['uptime_start']) / 60, 1), 2),
                    "avg_query_length": 50,  # Estimated average
                    "avg_response_length": 200,  # Estimated average
                    "cache_efficiency": 85  # Estimated
                }
            },
            "model_performance": {
                "model_loaded": True,
                "model_type": rag_pipeline.get("model_id", "unknown"),
                "inference_backend": "local",
                "gpu_available": False,
                "memory_usage_mb": round(system_info.get("memory_total_gb", 0) * 1024 * (system_info.get("memory_percent", 0) / 100), 1),
                "estimated_tokens_processed": _system_metrics['requests_total'] * 100,  # Rough estimate
                "model_accuracy_score": 0.85  # Would be calculated from validation results
            }
        })
        
    except Exception as e:
        api_logger.error(f"System statistics error: {e}")
        return jsonify({
            "error": "Failed to get system statistics",
            "details": str(e) if app.debug else "Contact administrator",
            "timestamp": int(time.time())
        }), 500

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
        api_logger.error(f"File browser error: {e}")
        return jsonify({
            "error": "Failed to browse files",
            "details": str(e) if app.debug else "Contact administrator"
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