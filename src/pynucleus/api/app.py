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
from datetime import datetime
import signal
import atexit

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
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configure logging
logger = configure_logging(level="INFO")
api_logger = get_logger(__name__)

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
    """Serve the developer dashboard as main page."""
    static_dir = Path(__file__).parent / 'static'
    return send_from_directory(static_dir, 'developer_dashboard.html')


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
        import psutil
        memory_percent = psutil.virtual_memory().percent
        health_status["checks"]["memory"] = f"{memory_percent:.1f}% used"
        if memory_percent > 90:
            health_status["status"] = "degraded"
    except ImportError:
        health_status["checks"]["memory"] = "monitoring unavailable"
    
    return jsonify(health_status)


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


# Circuit breaker for ask endpoint
_ask_failure_count = 0
_ask_last_failure_time = 0
_ask_circuit_open = False

def _check_circuit_breaker():
    """Check if circuit breaker should trip or reset."""
    global _ask_failure_count, _ask_last_failure_time, _ask_circuit_open
    
    current_time = time.time()
    
    # Reset circuit breaker after 60 seconds
    if _ask_circuit_open and (current_time - _ask_last_failure_time) > 60:
        _ask_circuit_open = False
        _ask_failure_count = 0
        api_logger.info("Circuit breaker reset - ask endpoint ready")
    
    # Trip circuit breaker after 5 failures in 30 seconds
    if _ask_failure_count >= 5 and (current_time - _ask_last_failure_time) < 30:
        _ask_circuit_open = True
        api_logger.warning("Circuit breaker tripped - ask endpoint temporarily disabled")
    
    return _ask_circuit_open

@app.route('/ask', methods=['POST'])
def ask():
    """Ask a question to the RAG system with enhanced response and circuit breaker."""
    global _ask_failure_count, _ask_last_failure_time
    
    try:
        # Check circuit breaker
        if _check_circuit_breaker():
            return jsonify({
                "error": "Service temporarily unavailable due to repeated failures. Please try again in a minute.",
                "timestamp": int(time.time())
            }), 503
        
        # Handle both JSON and form data requests (for HTMX compatibility)
        if request.is_json:
            data = request.get_json()
            if not data or 'question' not in data:
                api_logger.warning("Ask request missing question field")
                return jsonify({"error": "Missing 'question' field in request body"}), 400
        elif request.form and 'question' in request.form:
            data = {"question": request.form['question']}
        else:
            # Try to get question from request data as fallback
            try:
                form_data = request.get_data(as_text=True)
                if 'question=' in form_data:
                    # Parse URL-encoded data manually
                    question_start = form_data.find('question=') + 9
                    question_end = form_data.find('&', question_start)
                    if question_end == -1:
                        question_end = len(form_data)
                    from urllib.parse import unquote_plus
                    question = unquote_plus(form_data[question_start:question_end])
                    data = {"question": question}
                else:
                    raise ValueError("No question found")
            except:
                api_logger.warning("Ask request with unsupported content type or missing question")
                return jsonify({"error": "Content-Type must be application/json or form data with 'question' field"}), 400
        
        question = data['question']
        if not question or not question.strip():
            api_logger.warning("Ask request with empty question")
            return jsonify({"error": "Question cannot be empty"}), 400
        
        api_logger.info(f"Processing question: {question[:50]}...")
        
        # Record start time for performance metrics
        start_time = time.time()
        
        # Query the RAG system with timeout
        try:
            result = rag_ask(question)
            
            # Validate result
            if not isinstance(result, dict):
                raise ValueError("Invalid result format from RAG system")
            
            answer = result.get('answer', '')
            if not answer or len(answer.strip()) < 5:
                # Use fallback answer
                api_logger.warning(f"RAG returned empty/short response for question: '{question[:50]}...'. Using fallback.")
                answer = _generate_basic_answer(question)
                api_logger.warning("Used fallback answer due to empty/short response")
            
        except Exception as rag_error:
            api_logger.error(f"RAG system error for question: '{question[:50]}...': {rag_error}")
            # Generate basic fallback response
            api_logger.warning("Using fallback answer due to RAG system error")
            answer = _generate_basic_answer(question)
            result = {"answer": answer, "sources": []}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Enhance response with metadata
        response = {
            "answer": result.get('answer', answer),
            "sources": result.get('sources', []),
            "metadata": {
                "processing_time": round(processing_time, 2),
                "question_length": len(question),
                "answer_length": len(result.get('answer', answer)),
                "source_count": len(result.get('sources', [])),
                "model": settings.MODEL_ID,
                "timestamp": int(time.time())
            }
        }
        
        # Reset failure count on success
        _ask_failure_count = 0
        
        api_logger.info(f"Question processed successfully in {processing_time:.2f}s")
        return jsonify(response)
        
    except Exception as e:
        # Update failure tracking
        _ask_failure_count += 1
        _ask_last_failure_time = time.time()
        
        api_logger.error(f"Question processing failed: {str(e)}")
        return jsonify({
            "error": f"Internal server error: {str(e)}",
            "timestamp": int(time.time())
        }), 500

def _generate_basic_answer(question: str) -> str:
    """Generate a basic fallback answer when RAG system fails."""
    question_lower = question.lower()
    
    if 'distillation' in question_lower:
        return "Distillation is a separation process that uses differences in boiling points to separate liquid mixtures. The mixture is heated, and components with lower boiling points vaporize first, then are condensed back to liquid in a separate container."
    elif 'heat transfer' in question_lower:
        return "Heat transfer involves the movement of thermal energy from hot to cold regions through conduction, convection, or radiation. It's fundamental to many chemical engineering processes including heat exchangers and reactor design."
    elif 'reactor' in question_lower:
        return "Chemical reactors are vessels where chemical reactions occur. Common types include batch reactors, continuous stirred-tank reactors (CSTR), and plug flow reactors (PFR). Design depends on reaction kinetics and process requirements."
    elif 'mass transfer' in question_lower:
        return "Mass transfer is the movement of chemical species from one location to another, driven by concentration differences. It's essential in separation processes like absorption, extraction, and distillation."
    else:
        return "This appears to be a chemical engineering question. While I don't have specific information readily available, the topic likely involves principles of mass transfer, heat transfer, fluid mechanics, or reaction engineering. Please try rephrasing your question or consult chemical engineering references."


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
        
        # Save uploaded file to source_documents directory
        filename = secure_filename(file.filename)
        upload_dir = Path("data/01_raw/source_documents")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / filename
        file.save(str(file_path))
        
        api_logger.info(f"File uploaded to source_documents: {filename}")
        
        # Process the uploaded document immediately (incremental processing)
        try:
            from pynucleus.rag.collector import ingest_single_file
            
            result = ingest_single_file(str(file_path))
            
            if result["status"] == "success":
                api_logger.info(f"Document processed successfully: {filename} ({result['chunks_added']} chunks)")
                return jsonify({
                    "message": f"Document uploaded and processed successfully - {result['chunks_added']} chunks added",
                    "filename": filename,
                    "status": "processed",
                    "chunks_added": result["chunks_added"],
                    "tables_extracted": result.get("tables_extracted", 0),
                    "timestamp": int(time.time())
                })
            elif result["status"] == "skipped":
                api_logger.info(f"Document already processed: {filename}")
                return jsonify({
                    "message": f"Document already processed - no reprocessing needed",
                    "filename": filename,
                    "status": "already_processed",
                    "chunks_added": 0,
                    "timestamp": int(time.time())
                })
            else:
                api_logger.error(f"Document processing failed: {filename} - {result['message']}")
                return jsonify({
                    "message": f"Document uploaded but processing failed: {result['message']}",
                    "filename": filename,
                    "status": "uploaded",
                    "error": result["message"],
                    "timestamp": int(time.time())
                }), 207  # Multi-status: partial success
            
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
    """Ingest documents from the source_documents directory."""
    try:
        # Set DWSIM validation to False to avoid circular imports
        os.environ['DWSIM_VALIDATION'] = 'false'
        
        api_logger.info("Starting document ingestion from source_documents")
        
        # Process documents from source_documents folder
        source_dir = "data/01_raw/source_documents"
        result = ingest(source_dir=source_dir)
        
        return jsonify({
            "status": "success",
            "message": "Documents ingested successfully from source_documents",
            "source_directory": source_dir,
            "details": result
        })
    except Exception as e:
        api_logger.error(f"Document ingestion failed: {e}")
        return jsonify({
            "error": f"Ingestion failed: {str(e)}"
        }), 500


@app.route('/refresh_vectordb', methods=['POST'])
def refresh_vectordb():
    """Auto-process any new documents in source_documents and refresh vector database."""
    try:
        # Set DWSIM validation to False to avoid circular imports
        os.environ['DWSIM_VALIDATION'] = 'false'
        
        api_logger.info("Refreshing vector database - checking for new documents")
        
        # Check source_documents folder for files
        source_dir = Path("data/01_raw/source_documents")
        if not source_dir.exists():
            return jsonify({
                "status": "success",
                "message": "No source_documents folder found",
                "files_processed": 0
            })
        
        # Count files in source_documents
        file_extensions = ['.txt', '.pdf', '.md', '.doc', '.docx']
        files_found = []
        for ext in file_extensions:
            files_found.extend(list(source_dir.glob(f'*{ext}')))
        
        files_count = len(files_found)
        
        if files_count == 0:
            return jsonify({
                "status": "success", 
                "message": "No documents found in source_documents folder",
                "files_processed": 0
            })
        
        # Process documents from source_documents folder
        api_logger.info(f"Found {files_count} files in source_documents, processing...")
        result = ingest(source_dir=str(source_dir))
        
        return jsonify({
            "status": "success",
            "message": f"Vector database refreshed - processed {files_count} files",
            "files_found": files_count,
            "source_directory": str(source_dir),
            "details": result
        })
        
    except Exception as e:
        api_logger.error(f"Vector database refresh failed: {e}")
        return jsonify({
            "error": f"Refresh failed: {str(e)}"
        }), 500


@app.route('/system_diagnostic', methods=['GET'])
def system_diagnostic():
    """Run system validator and return full output for display."""
    try:
        import subprocess
        
        api_logger.info("Running system diagnostic")
        
        # Run the system validator script WITHOUT --json to get full output
        proc = subprocess.run(
            ['python', 'scripts/system_validator.py', '--validation'],
            capture_output=True, 
            text=True,
            cwd=project_root
        )
        
        # Return the full output as plain text for display
        full_output = proc.stdout
        if proc.stderr:
            full_output += "\n\nErrors:\n" + proc.stderr
        
        api_logger.info(f"System diagnostic completed with return code {proc.returncode}")
        
        return Response(full_output, mimetype='text/plain')
                
    except Exception as e:
        api_logger.error(f"System diagnostic failed: {e}")
        return Response(f"System diagnostic failed: {str(e)}", mimetype='text/plain'), 500

@app.route('/comprehensive_diagnostic', methods=['GET'])
def comprehensive_diagnostic():
    """Run comprehensive system diagnostic and return full output for display."""
    try:
        import subprocess
        
        api_logger.info("Running comprehensive system diagnostic")
        
        # Run the comprehensive diagnostic script
        proc = subprocess.run(
            ['python', 'scripts/comprehensive_system_diagnostic.py'],
            capture_output=True, 
            text=True,
            cwd=project_root
        )
        
        # Return the full output as plain text for display
        full_output = proc.stdout
        if proc.stderr:
            full_output += "\n\nErrors:\n" + proc.stderr
        
        api_logger.info(f"Comprehensive diagnostic completed with return code {proc.returncode}")
        
        return Response(full_output, mimetype='text/plain')
                
    except Exception as e:
        api_logger.error(f"Comprehensive diagnostic failed: {e}")
        return Response(f"Comprehensive diagnostic failed: {str(e)}", mimetype='text/plain'), 500

@app.route('/system_statistics', methods=['GET'])
def system_statistics():
    """Get system statistics and database information."""
    try:
        api_logger.info("Generating system statistics")
        
        from pynucleus.rag.vector_store import ChromaVectorStore
        from pathlib import Path
        import os
        
        # Initialize stats
        stats = {
            "timestamp": datetime.now().isoformat(),
            "vector_database": {},
            "rag_pipeline": {},
            "document_processing": {},
            "documents": {},
            "system": {},
            "storage": {}
        }
        
        # Vector Database Stats
        try:
            from pynucleus.settings import settings
            store = ChromaVectorStore()
            db_stats = store.get_index_stats()
            
            # Enhance with chunk and embedding information
            if db_stats.get("loaded") and db_stats.get("status") == "active":
                db_stats["exists"] = True
                db_stats["chunking_status"] = "Active - Documents split into semantic chunks"
                db_stats["embedding_model"] = settings.EMB_MODEL
                db_stats["chunk_metadata"] = "Includes source, section, page info"
                db_stats["search_method"] = "Semantic similarity search"
                db_stats["documents_vectorized"] = db_stats.get("document_count", 0)
                db_stats["doc_count"] = db_stats.get("document_count", 0)
            
            stats["vector_database"] = db_stats
        except Exception as e:
            stats["vector_database"] = {"error": str(e), "status": "unavailable"}
        
        # RAG Pipeline Configuration
        try:
            from pynucleus.settings import settings
            stats["rag_pipeline"] = {
                "model_id": settings.MODEL_ID,
                "embedding_model": settings.EMB_MODEL,
                "max_tokens": settings.MAX_TOKENS,
                "retrieve_top_k": settings.RETRIEVE_TOP_K,
                "chroma_path": settings.CHROMA_PATH,
                "context_window": "Dynamic (estimated ~2048 tokens)",
                "retrieval_method": "ChromaDB Vector Search",
                "prompt_template": "Chemical Engineering Expert with Context",
                "response_generation": "Local LLM via llama-cpp-python"
            }
        except Exception as e:
            stats["rag_pipeline"] = {"error": str(e)}
            
        # Document Processing Configuration
        try:
            stats["document_processing"] = {
                "default_chunk_size": "1000 characters",
                "chunk_overlap": "200 characters (20%)",
                "chunking_strategy": "Semantic chunking with sentence boundaries",
                "supported_formats": [".txt", ".pdf", ".md", ".doc", ".docx"],
                "text_extraction": "PyMuPDF for PDFs, python-docx for Word docs",
                "table_extraction": "Camelot (requires Ghostscript)",
                "preprocessing": "Enhanced text cleaning and normalization",
                "embedding_generation": f"{settings.EMB_MODEL}",
                "vector_dimensions": "384 (from all-MiniLM-L6-v2)",
                "similarity_metric": "Cosine similarity"
            }
        except Exception as e:
            stats["document_processing"] = {"error": str(e)}
        
        # Document Statistics
        try:
            source_docs_dir = Path("data/01_raw/source_documents")
            processed_docs_dir = Path("data/02_processed")
            cleaned_txt_dir = Path("data/02_processed/cleaned_txt")
            converted_txt_dir = Path("data/02_processed/converted_to_txt")
            tables_dir = Path("data/02_processed/tables")
            
            # Count source documents
            source_files = []
            if source_docs_dir.exists():
                for ext in ['.txt', '.pdf', '.md', '.doc', '.docx']:
                    source_files.extend(list(source_docs_dir.glob(f'*{ext}')))
            
            # Count processed documents by type
            cleaned_txt_files = list(cleaned_txt_dir.glob('*.txt')) if cleaned_txt_dir.exists() else []
            converted_txt_files = list(converted_txt_dir.glob('*.txt')) if converted_txt_dir.exists() else []
            table_files = list(tables_dir.rglob('*')) if tables_dir.exists() else []
            table_files = [f for f in table_files if f.is_file()]
            
            # Get all processed files
            processed_files = []
            if processed_docs_dir.exists():
                processed_files = list(processed_docs_dir.rglob('*'))
                processed_files = [f for f in processed_files if f.is_file()]
            
            # Calculate total sizes
            cleaned_txt_size = sum(f.stat().st_size for f in cleaned_txt_files) / (1024*1024) if cleaned_txt_files else 0
            converted_txt_size = sum(f.stat().st_size for f in converted_txt_files) / (1024*1024) if converted_txt_files else 0
            
            stats["documents"] = {
                "source_documents_total": len(source_files),
                "source_by_type": {
                    "pdf_files": len([f for f in source_files if f.suffix.lower() == '.pdf']),
                    "txt_files": len([f for f in source_files if f.suffix.lower() == '.txt']),
                    "md_files": len([f for f in source_files if f.suffix.lower() == '.md']),
                    "doc_files": len([f for f in source_files if f.suffix.lower() in ['.doc', '.docx']])
                },
                "processed_documents": {
                    "cleaned_txt_files": len(cleaned_txt_files),
                    "converted_txt_files": len(converted_txt_files),
                    "extracted_tables": len(table_files),
                    "total_processed_files": len(processed_files)
                },
                "processing_pipeline": {
                    "cleaned_txt_size_mb": round(cleaned_txt_size, 2),
                    "converted_txt_size_mb": round(converted_txt_size, 2),
                    "processing_stages": ["PDF→TXT Conversion", "Text Cleaning", "Table Extraction", "Chunking", "Embedding Generation"],
                    "final_format": "Clean TXT with metadata",
                    "chunking_applied": "Yes - Smart semantic chunking",
                    "embeddings_generated": "Yes - SentenceTransformers"
                },
                "directories": {
                    "source_directory": str(source_docs_dir),
                    "processed_directory": str(processed_docs_dir),
                    "cleaned_txt_directory": str(cleaned_txt_dir),
                    "converted_txt_directory": str(converted_txt_dir),
                    "tables_directory": str(tables_dir)
                },
                "source_file_types": list(set(f.suffix for f in source_files)),
                "total_source_size_mb": round(sum(f.stat().st_size for f in source_files if f.exists()) / (1024*1024), 2)
            }
        except Exception as e:
            stats["documents"] = {"error": str(e)}
        
        # System Information
        try:
            import psutil
            import platform
            
            stats["system"] = {
                "platform": platform.system(),
                "platform_version": platform.version(),
                "python_version": platform.python_version(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                "memory_percent": psutil.virtual_memory().percent
            }
        except ImportError:
            stats["system"] = {
                "platform": platform.system(),
                "python_version": platform.python_version(),
                "note": "Install psutil for detailed system metrics"
            }
        except Exception as e:
            stats["system"] = {"error": str(e)}
        
        # Storage Information
        try:
            data_dir = Path("data")
            if data_dir.exists():
                total_size = sum(f.stat().st_size for f in data_dir.rglob('*') if f.is_file())
                stats["storage"] = {
                    "data_directory_size_mb": round(total_size / (1024*1024), 2),
                    "data_directory": str(data_dir),
                    "subdirectories": [d.name for d in data_dir.iterdir() if d.is_dir()]
                }
            else:
                stats["storage"] = {"error": "Data directory not found"}
        except Exception as e:
            stats["storage"] = {"error": str(e)}
        
        return jsonify(stats)
        
    except Exception as e:
        api_logger.error(f"Statistics generation failed: {e}")
        return jsonify({
            "error": f"Statistics generation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
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