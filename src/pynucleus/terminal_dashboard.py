"""
Chemical Engineering Terminal Dashboard

A clean, fast Flask-based dashboard with terminal-like black design.
Simple, intuitive, and user-friendly interface for chemical engineering document processing and analysis.
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import shutil
from datetime import datetime
import threading

# Add src directory to Python path
root_dir = Path(__file__).parent.parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

# Import PyNucleus modules
from pynucleus.rag.engine import ask as rag_ask
from pynucleus.rag.document_processor import DocumentProcessor
from pynucleus.rag.collector import ingest
from pynucleus.utils.logger import logger

# Debug logging for path resolution (after logger import)
logger.info(f"Dashboard file: {__file__}")
logger.info(f"Root directory: {root_dir}")
logger.info(f"Root directory absolute: {root_dir.absolute()}")
logger.info(f"Scripts directory: {root_dir / 'scripts'}")
logger.info(f"Scripts directory exists: {(root_dir / 'scripts').exists()}")
logger.info(f"Current working directory: {os.getcwd()}")

app = Flask(__name__)
app.secret_key = 'chemical-engineering-dashboard-2024'

# Global state
chat_history = []
diagnostic_results = None
statistics_results = None
upload_status = ""
processing_status = ""

# Configuration
UPLOAD_FOLDER = Path("data/01_raw")
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'md', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html', 
                         chat_history=chat_history,
                         diagnostic_results=diagnostic_results,
                         statistics_results=statistics_results,
                         upload_status=upload_status,
                         processing_status=processing_status)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A requests"""
    global chat_history
    
    question = request.form.get('question', '').strip()
    if not question:
        flash('Please enter a question', 'error')
        return redirect(url_for('index'))
    
    try:
        # Get RAG response
        result = rag_ask(question)
        
        # Add to chat history
        chat_entry = {
            "question": question,
            "answer": result.get("answer", "No answer generated"),
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0),
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        chat_history.append(chat_entry)
        
        flash('Response generated successfully!', 'success')
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        logger.error(f"Q&A error: {e}")
    
    return redirect(url_for('index'))

@app.route('/clear_history')
def clear_history():
    """Clear chat history"""
    global chat_history
    chat_history = []
    flash('Chat history cleared!', 'success')
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads with improved error handling"""
    global upload_status, processing_status
    
    if 'files' not in request.files:
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('No files selected', 'error')
        return redirect(url_for('index'))
    
    try:
        # Create temporary directory for uploads
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save uploaded files
            saved_files = []
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = temp_path / filename
                    file.save(file_path)
                    saved_files.append(file_path)
            
            if not saved_files:
                flash('No valid files uploaded', 'error')
                return redirect(url_for('index'))
            
            # Process documents with better error handling
            processor = DocumentProcessor()
            processed_count = 0
            tables_extracted = 0
            
            for file_path in saved_files:
                try:
                    if file_path.suffix.lower() == '.pdf':
                        result = processor.process_document(file_path)
                        tables_extracted += result.get("tables_extracted", 0)
                        processed_count += 1
                    else:
                        processed_count += 1
                except Exception as e:
                    logger.error(f"Error processing {file_path.name}: {e}")
                    flash(f'Error processing {file_path.name}: {str(e)}', 'error')
            
            # Ingest documents
            try:
                ingest(str(temp_path))
                upload_status = f"Successfully processed {processed_count} documents"
                if tables_extracted > 0:
                    upload_status += f" (extracted {tables_extracted} tables)"
                flash(upload_status, 'success')
            except Exception as e:
                flash(f'Error during ingestion: {str(e)}', 'error')
                logger.error(f"Ingestion error: {e}")
            
    except Exception as e:
        error_msg = f"Error processing documents: {str(e)}"
        upload_status = error_msg
        flash(error_msg, 'error')
        logger.error(f"Document processing error: {e}")
    
    return redirect(url_for('index'))

@app.route('/run_diagnostics')
def run_diagnostics():
    """Run system diagnostics using the comprehensive_system_diagnostic.py script"""
    global diagnostic_results
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        # Run the comprehensive system diagnostic script
        script_path = root_dir / "scripts" / "comprehensive_system_diagnostic.py"
        
        # Debug logging
        logger.info(f"Root directory: {root_dir}")
        logger.info(f"Script path: {script_path}")
        logger.info(f"Script exists: {script_path.exists()}")
        logger.info(f"Current working directory: {os.getcwd()}")
        
        if not script_path.exists():
            if is_ajax:
                return jsonify({"error": f"System diagnostic script not found at {script_path}"}), 404
            flash(f'System diagnostic script not found at {script_path}', 'error')
            return redirect(url_for('index'))
        
        # Run diagnostic in a separate thread to avoid blocking
        def run_diagnostic():
            global diagnostic_results
            try:
                logger.info(f"Starting diagnostic script: {script_path}")
                logger.info(f"Working directory: {root_dir}")
                
                result = subprocess.run(
                    [sys.executable, str(script_path), "--quiet"],
                    capture_output=True,
                    text=True,
                    cwd=root_dir,
                    timeout=300  # 5 minute timeout
                )
                
                logger.info(f"Diagnostic completed with return code: {result.returncode}")
                
                diagnostic_results = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except subprocess.TimeoutExpired:
                logger.error("Diagnostic timed out")
                diagnostic_results = {
                    "return_code": -1,
                    "stdout": "Diagnostic timed out after 5 minutes",
                    "stderr": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.error(f"Diagnostic error: {e}")
                diagnostic_results = {
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # Start diagnostic in background
        thread = threading.Thread(target=run_diagnostic)
        thread.daemon = True
        thread.start()
        
        if is_ajax:
            return jsonify({"status": "started", "message": "System diagnostics started. Check the Diagnostics tab for results."})
        
        flash('System diagnostics started. Check the Diagnostics tab for results.', 'success')
        
    except Exception as e:
        if is_ajax:
            return jsonify({"error": f"Error starting diagnostics: {str(e)}"}), 500
        flash(f'Error starting diagnostics: {str(e)}', 'error')
        logger.error(f"Diagnostic error: {e}")
    
    return redirect(url_for('index'))

@app.route('/run_validation')
def run_validation():
    """Run system validation using the system_validator.py script"""
    global diagnostic_results
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        # Run the system validator script
        script_path = root_dir / "scripts" / "system_validator.py"
        
        if not script_path.exists():
            if is_ajax:
                return jsonify({"error": f"System validator script not found at {script_path}"}), 404
            flash(f'System validator script not found at {script_path}', 'error')
            return redirect(url_for('index'))
        
        # Run validation in a separate thread
        def run_validation():
            global diagnostic_results
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path), "--quiet"],
                    capture_output=True,
                    text=True,
                    cwd=root_dir,
                    timeout=300  # 5 minute timeout
                )
                
                diagnostic_results = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "validation"
                }
                
            except subprocess.TimeoutExpired:
                diagnostic_results = {
                    "return_code": -1,
                    "stdout": "Validation timed out after 5 minutes",
                    "stderr": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "validation"
                }
            except Exception as e:
                diagnostic_results = {
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "type": "validation"
                }
        
        # Start validation in background
        thread = threading.Thread(target=run_validation)
        thread.daemon = True
        thread.start()
        
        if is_ajax:
            return jsonify({"status": "started", "message": "System validation started. Check the Diagnostics tab for results."})
        
        flash('System validation started. Check the Diagnostics tab for results.', 'success')
        
    except Exception as e:
        if is_ajax:
            return jsonify({"error": f"Error starting validation: {str(e)}"}), 500
        flash(f'Error starting validation: {str(e)}', 'error')
        logger.error(f"Validation error: {e}")
    
    return redirect(url_for('index'))

@app.route('/run_statistics')
def run_statistics():
    """Run system statistics using the system_statistics.py script"""
    global statistics_results
    
    # Check if this is an AJAX request
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'
    
    try:
        # Run the system statistics script
        script_path = root_dir / "scripts" / "system_statistics.py"
        
        if not script_path.exists():
            if is_ajax:
                return jsonify({"error": f"System statistics script not found at {script_path}"}), 404
            flash(f'System statistics script not found at {script_path}', 'error')
            return redirect(url_for('index'))
        
        # Run statistics in a separate thread
        def run_statistics():
            global statistics_results
            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    cwd=root_dir,
                    timeout=120  # 2 minute timeout
                )
                
                statistics_results = {
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
            except subprocess.TimeoutExpired:
                statistics_results = {
                    "return_code": -1,
                    "stdout": "Statistics generation timed out after 2 minutes",
                    "stderr": "",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                statistics_results = {
                    "return_code": -1,
                    "stdout": "",
                    "stderr": str(e),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
        
        # Start statistics in background
        thread = threading.Thread(target=run_statistics)
        thread.daemon = True
        thread.start()
        
        if is_ajax:
            return jsonify({"status": "started", "message": "System statistics generation started. Check the Statistics tab for results."})
        
        flash('System statistics generation started. Check the Statistics tab for results.', 'success')
        
    except Exception as e:
        if is_ajax:
            return jsonify({"error": f"Error starting statistics: {str(e)}"}), 500
        flash(f'Error starting statistics: {str(e)}', 'error')
        logger.error(f"Statistics error: {e}")
    
    return redirect(url_for('index'))

@app.route('/api/diagnostic_results')
def get_diagnostic_results():
    """API endpoint to get diagnostic results"""
    global diagnostic_results
    return jsonify(diagnostic_results if diagnostic_results else {"status": "no_results"})

@app.route('/api/statistics_results')
def get_statistics_results():
    """API endpoint to get statistics results"""
    global statistics_results
    return jsonify(statistics_results if statistics_results else {"status": "no_results"})

@app.route('/api/system_status')
def system_status():
    """API endpoint for system status"""
    try:
        # Check if key components are available
        status = {
            "rag_system": "available",
            "document_processor": "available", 
            "chromadb": "available",
            "llm_model": "available",
            "timestamp": datetime.now().isoformat()
        }
        
        # Test RAG system
        try:
            test_result = rag_ask("test")
            status["rag_system"] = "operational"
        except:
            status["rag_system"] = "error"
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/performance_test', methods=['POST'])
def performance_test():
    """API endpoint for performance testing"""
    try:
        data = request.get_json()
        question = data.get('question', 'What is chemical engineering?')
        
        start_time = time.time()
        result = rag_ask(question)
        response_time = time.time() - start_time
        
        return jsonify({
            "question": question,
            "answer": result.get("answer", ""),
            "response_time": response_time,
            "sources": result.get("sources", []),
            "confidence": result.get("confidence", 0.0)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/test_paths')
def test_paths():
    """Test endpoint to verify script paths"""
    try:
        script_path = root_dir / "scripts" / "comprehensive_system_diagnostic.py"
        return jsonify({
            "dashboard_file": __file__,
            "root_dir": str(root_dir),
            "root_dir_absolute": str(root_dir.absolute()),
            "scripts_dir": str(root_dir / "scripts"),
            "scripts_dir_exists": (root_dir / "scripts").exists(),
            "script_path": str(script_path),
            "script_exists": script_path.exists(),
            "current_working_dir": os.getcwd(),
            "python_executable": sys.executable
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_app():
    """Application factory for production deployment"""
    return app

if __name__ == '__main__':
    # Find available port
    ports = [5000, 5001, 8080, 8000, 3000]
    available_port = None
    
    for port in ports:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                available_port = port
                break
        except OSError:
            continue
    
    if available_port is None:
        print("No available ports found. Exiting.")
        sys.exit(1)
    
    print(f"Starting Chemical Engineering Terminal Dashboard on port {available_port}")
    print(f"Open your browser to: http://localhost:{available_port}")
    
    app.run(host='0.0.0.0', port=available_port, debug=False) 