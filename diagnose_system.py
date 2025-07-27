#!/usr/bin/env python3
"""
PyNucleus System Diagnostic Tool

This script performs comprehensive health checks on the PyNucleus system
to identify and help resolve common issues.
"""

import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Add src to path for imports
sys.path.append('src')

def run_check(name: str, check_func) -> Tuple[bool, str]:
    """Run a diagnostic check and return result."""
    try:
        return True, check_func()
    except Exception as e:
        return False, str(e)

def check_python_environment() -> str:
    """Check Python environment and virtual environment status."""
    python_version = sys.version
    venv_path = os.environ.get('VIRTUAL_ENV', 'Not in virtual environment')
    
    # Check if we're in the expected virtual environment
    expected_venv = str(Path.cwd() / '.venv')
    if venv_path != 'Not in virtual environment':
        if Path(venv_path).resolve() == Path(expected_venv).resolve():
            venv_status = "✅ Correct virtual environment"
        else:
            venv_status = f"⚠️  Unexpected virtual environment: {venv_path}"
    else:
        venv_status = "❌ No virtual environment activated"
    
    return f"Python {python_version.split()[0]}, {venv_status}"

def check_critical_packages() -> str:
    """Check if critical packages are installed and importable."""
    critical_packages = [
        'torch', 'transformers', 'chromadb', 'bitsandbytes', 
        'sentence_transformers', 'pynucleus'
    ]
    
    results = []
    for package in critical_packages:
        try:
            if package == 'pynucleus':
                # Special handling for pynucleus
                from pynucleus import __version__
                version = __version__
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
            results.append(f"✅ {package}: {version}")
        except ImportError as e:
            results.append(f"❌ {package}: Not installed ({e})")
        except Exception as e:
            results.append(f"⚠️  {package}: Error ({e})")
    
    return "\n".join(results)

def check_bitsandbytes() -> str:
    """Test bitsandbytes functionality."""
    try:
        import bitsandbytes as bnb
        from transformers import BitsAndBytesConfig
        
        # Test basic functionality
        config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Check for GPU support warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This might trigger the GPU support warning
            import bitsandbytes.cextension
            
            gpu_warnings = [warning for warning in w if "GPU support" in str(warning.message)]
            
            if gpu_warnings:
                return "⚠️  bitsandbytes: Working but no GPU support (expected on macOS)"
            else:
                return "✅ bitsandbytes: Fully functional"
                
    except Exception as e:
        return f"❌ bitsandbytes: Error - {e}"

def check_chromadb() -> str:
    """Check ChromaDB status and document count."""
    try:
        from pynucleus.utils.telemetry_patch import apply_telemetry_patch
        apply_telemetry_patch()
        
        import chromadb
        from chromadb.config import Settings
        from pynucleus.settings import settings
        
        # Check main ChromaDB instance
        chroma_path = Path(settings.CHROMA_PATH)
        if not chroma_path.exists():
            return f"❌ ChromaDB directory does not exist: {settings.CHROMA_PATH}"
        
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        collections = client.list_collections()
        if not collections:
            return "⚠️  ChromaDB: No collections found (database empty)"
        
        total_docs = 0
        collection_info = []
        for collection in collections:
            count = collection.count()
            total_docs += count
            collection_info.append(f"  • {collection.name}: {count} documents")
        
        if total_docs == 0:
            return "⚠️  ChromaDB: Collections exist but no documents\n" + "\n".join(collection_info)
        else:
            return f"✅ ChromaDB: {total_docs} total documents\n" + "\n".join(collection_info)
            
    except Exception as e:
        return f"❌ ChromaDB: Error - {e}"

def check_model_loading() -> str:
    """Test model loading functionality."""
    try:
        from pynucleus.settings import settings
        from pynucleus.llm.model_loader import get_model_loader
        
        # Get model info without actually loading models (faster)
        loader = get_model_loader()
        model_info = loader.get_model_info()
        
        results = []
        results.append(f"Model ID: {model_info['model_id']}")
        results.append(f"Loading method: {model_info['method']}")
        results.append(f"CUDA available: {model_info['cuda_available']}")
        results.append(f"MPS available: {model_info['mps_available']}")
        
        if model_info['method'] == 'Failed':
            return "❌ Model loading: Failed\n" + "\n".join(results)
        elif model_info['method'] == 'HuggingFace':
            return "✅ Model loading: Working (HuggingFace)\n" + "\n".join(results)
        elif model_info['method'] == 'Optimized':
            return "✅ Model loading: Working (Optimized GGUF)\n" + "\n".join(results)
        else:
            return "⚠️  Model loading: Unknown status\n" + "\n".join(results)
            
    except Exception as e:
        return f"❌ Model loading: Error - {e}"

def check_document_retrieval() -> str:
    """Test document retrieval functionality."""
    try:
        from pynucleus.rag.engine import retrieve
        
        # Test with a simple query
        docs, sources = retrieve("distillation", k=1)
        
        if not docs:
            return "❌ Document retrieval: No documents returned"
        elif len(docs) == 0:
            return "⚠️  Document retrieval: Empty results"
        else:
            return f"✅ Document retrieval: Working ({len(docs)} docs, {len(sources)} sources)"
            
    except Exception as e:
        return f"❌ Document retrieval: Error - {e}"

def check_file_permissions() -> str:
    """Check file permissions for critical directories."""
    critical_paths = [
        'data/03_intermediate/vector_db',
        'cache/models',
        'logs'
    ]
    
    results = []
    for path_str in critical_paths:
        path = Path(path_str)
        if path.exists():
            if os.access(path, os.R_OK | os.W_OK):
                results.append(f"✅ {path_str}: Read/Write OK")
            else:
                results.append(f"❌ {path_str}: Permission denied")
        else:
            results.append(f"⚠️  {path_str}: Does not exist")
    
    return "\n".join(results)

def check_disk_space() -> str:
    """Check available disk space."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(Path.cwd())
        
        # Convert to GB
        total_gb = total // (1024**3)
        used_gb = used // (1024**3)
        free_gb = free // (1024**3)
        
        if free_gb < 1:
            return f"❌ Disk space: Only {free_gb}GB free (need at least 1GB)"
        elif free_gb < 5:
            return f"⚠️  Disk space: {free_gb}GB free (low space warning)"
        else:
            return f"✅ Disk space: {free_gb}GB free ({total_gb}GB total)"
            
    except Exception as e:
        return f"❌ Disk space check failed: {e}"

def main():
    """Run all diagnostic checks."""
    print("🔍 PyNucleus System Diagnostics")
    print("=" * 50)
    print()
    
    checks = [
        ("Python Environment", check_python_environment),
        ("Critical Packages", check_critical_packages),
        ("Bitsandbytes", check_bitsandbytes),
        ("ChromaDB", check_chromadb),
        ("Model Loading", check_model_loading),
        ("Document Retrieval", check_document_retrieval),
        ("File Permissions", check_file_permissions),
        ("Disk Space", check_disk_space),
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"🔍 Checking {check_name}...")
        success, message = run_check(check_name, check_func)
        results[check_name] = (success, message)
        print(f"   {message}")
        print()
    
    # Summary
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All checks passed! Your PyNucleus system is healthy.")
    else:
        print("⚠️  Some issues detected. See details above.")
        print()
        print("🔧 RECOMMENDED ACTIONS:")
        
        for check_name, (success, message) in results.items():
            if not success:
                print(f"   • Fix {check_name}: {message}")
        
        print()
        print("💡 Common solutions:")
        print("   • Run: ./setup_environment.sh")
        print("   • Activate virtual environment: source .venv/bin/activate")
        print("   • Ingest documents: pynucleus -> option 5")
        print("   • Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()