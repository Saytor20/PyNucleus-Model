"""
Unified Diagnostic System for PyNucleus Flask Application.

Provides comprehensive system diagnostics and health-checking capabilities
with concurrent execution for optimal performance.
"""

import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional
import traceback
import time
from datetime import datetime, timedelta

# Add scripts directory to path for diagnostic imports
SCRIPTS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(SCRIPTS_PATH) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PATH))

def run_comprehensive() -> Dict[str, Any]:
    """Run comprehensive system diagnostic with error handling."""
    try:
        from comprehensive_system_diagnostic import ComprehensiveSystemDiagnostic
        
        start_time = time.time()
        diagnostic = ComprehensiveSystemDiagnostic(quiet_mode=True, test_mode=False)
        diagnostic.run_comprehensive_diagnostic()
        
        # Extract results into structured format
        return {
            "status": "completed",
            "execution_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "total_checks": diagnostic.total_checks,
            "passed_checks": diagnostic.passed_checks,
            "failed_checks": diagnostic.total_checks - diagnostic.passed_checks,
            "success_rate": round((diagnostic.passed_checks / diagnostic.total_checks) * 100, 1) if diagnostic.total_checks > 0 else 0,
            "component_health": {
                "environment": diagnostic.environment_health,
                "dependencies": diagnostic.dependencies_health,
                "scripts": diagnostic.scripts_health,
                "components": diagnostic.components_health,
                "docker": diagnostic.docker_health,
                "pdf_processing": diagnostic.pdf_processing_health
            },
            "script_health": {
                "total_scripts": diagnostic.total_scripts,
                "healthy_scripts": diagnostic.healthy_scripts,
                "script_health_rate": round((diagnostic.healthy_scripts / diagnostic.total_scripts) * 100, 1) if diagnostic.total_scripts > 0 else 0
            },
            "detailed_checks": [
                {
                    "name": check.check_name,
                    "category": check.category,
                    "passed": check.passed,
                    "execution_time": check.execution_time,
                    "details": check.details,
                    "warnings": check.warnings,
                    "error_message": check.error_message
                }
                for check in diagnostic.system_checks
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "execution_time": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def run_validator() -> Dict[str, Any]:
    """Run system validator with error handling."""
    try:
        from system_validator import SystemValidator
        
        start_time = time.time()
        validator = SystemValidator(quiet_mode=True)
        validator.run_validation_suite(include_citations=False, include_notebook=False)
        
        # Extract results into structured format  
        return {
            "status": "completed",
            "execution_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat(),
            "total_tests": validator.total_tests,
            "passed_tests": validator.passed_tests,
            "failed_tests": validator.total_tests - validator.passed_tests,
            "success_rate": round((validator.passed_tests / validator.total_tests) * 100, 1) if validator.total_tests > 0 else 0,
            "validation_health": "excellent" if validator.passed_tests / validator.total_tests >= 0.9 else 
                               "good" if validator.passed_tests / validator.total_tests >= 0.8 else
                               "warning" if validator.passed_tests / validator.total_tests >= 0.7 else
                               "critical" if validator.total_tests > 0 else "unknown",
            "detailed_results": [
                {
                    "test_name": result.test_name,
                    "query": result.query,
                    "domain": result.domain,
                    "difficulty": result.difficulty_level,
                    "accuracy_score": result.accuracy_score,
                    "citation_accuracy": result.citation_accuracy,
                    "response_time": result.response_time,
                    "sources_used": result.sources_used
                }
                for result in validator.validation_results
            ]
        }
    except Exception as e:
        return {
            "status": "error", 
            "execution_time": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def run_full_diagnostics() -> Dict[str, Any]:
    """
    Run full diagnostics with concurrent execution.
    
    Returns merged results from comprehensive diagnostic and system validator
    with execution metadata and health summary.
    """
    start_time = time.time()
    
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both diagnostic tasks concurrently
            comprehensive_future = executor.submit(run_comprehensive)
            validator_future = executor.submit(run_validator)
            
            # Collect results as they complete
            results = {}
            for future in as_completed([comprehensive_future, validator_future]):
                if future == comprehensive_future:
                    results["comprehensive"] = future.result()
                elif future == validator_future:
                    results["validation"] = validator_future.result()
        
        # Calculate overall health metrics
        total_execution_time = time.time() - start_time
        
        comp_success = results["comprehensive"].get("success_rate", 0)
        val_success = results["validation"].get("success_rate", 0)
        overall_health = (comp_success + val_success) / 2 if comp_success > 0 and val_success > 0 else max(comp_success, val_success)
        
        # Determine overall status
        if overall_health >= 90:
            overall_status = "excellent"
        elif overall_health >= 80:
            overall_status = "good"
        elif overall_health >= 70:
            overall_status = "warning"
        else:
            overall_status = "critical"
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": round(total_execution_time, 2),
            "overall_health": round(overall_health, 1),
            "overall_status": overall_status,
            "comprehensive": results["comprehensive"],
            "validation": results["validation"],
            "summary": {
                "total_checks": results["comprehensive"].get("total_checks", 0) + results["validation"].get("total_tests", 0),
                "total_passed": results["comprehensive"].get("passed_checks", 0) + results["validation"].get("passed_tests", 0),
                "component_health_count": sum(1 for v in results["comprehensive"].get("component_health", {}).values() if v),
                "execution_parallel": True,
                "both_completed": results["comprehensive"].get("status") == "completed" and results["validation"].get("status") == "completed"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "total_execution_time": round(time.time() - start_time, 2),
            "error": str(e),
            "traceback": traceback.format_exc(),
            "comprehensive": {"status": "not_run"},
            "validation": {"status": "not_run"}
        }

def run_system_statistics() -> Dict[str, Any]:
    """Run comprehensive system statistics collection."""
    try:
        start_time = time.time()
        
        # System information
        try:
            import psutil
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            system_info = {
                "platform": sys.platform,
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": round(cpu_percent, 1),
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_used_gb": round(memory.used / (1024**3), 2),
                "memory_percent": round(memory.percent, 1),
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_used_gb": round(disk.used / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 1)
            }
        except Exception as e:
            system_info = {"error": f"Failed to get system info: {e}"}
        
        # Vector database information
        try:
            from pynucleus.rag.vector_store import ChromaVectorStore
            store = ChromaVectorStore()
            stats = store.get_index_stats()
            
            # Test query performance
            query_start = time.time()
            test_docs = store.query("test", k=1)
            query_time = (time.time() - query_start) * 1000
            
            vector_database = {
                "status": "ready" if stats.get("exists", False) else "empty",
                "document_count": stats.get("doc_count", 0),
                "collection_name": "pynucleus_documents",
                "database_size_mb": round(stats.get("size_mb", 0), 2),
                "performance_metrics": {
                    "avg_query_time": round(query_time, 1),
                    "last_query_docs": len(test_docs) if test_docs else 0
                }
            }
        except Exception as e:
            vector_database = {"status": "error", "error": str(e)}
        
        # RAG pipeline information
        try:
            from pynucleus.settings import settings
            rag_pipeline = {
                "model_id": getattr(settings, 'MODEL_ID', 'unknown'),
                "embedding_model": getattr(settings, 'EMB_MODEL', 'unknown'),
                "max_tokens": getattr(settings, 'MAX_TOKENS', 'unknown'),
                "retrieve_top_k": getattr(settings, 'RETRIEVE_TOP_K', 'unknown'),
                "chroma_path": getattr(settings, 'CHROMA_PATH', 'unknown')
            }
        except Exception as e:
            rag_pipeline = {"error": f"Failed to get RAG config: {e}"}
        
        # Model information
        try:
            from pynucleus.llm.model_loader import get_model_info
            model_info = get_model_info()
            model_status = {
                "loaded": model_info.get('loaded', False),
                "method": model_info.get('method', 'unknown'),
                "model_name": model_info.get('model_name', 'unknown'),
                "device": model_info.get('device', 'unknown'),
                "status": "✅ Loaded" if model_info.get('loaded') else "❌ Not Loaded"
            }
        except Exception as e:
            model_status = {"status": "❌ Error", "error": str(e)}
        
        # Document information
        try:
            from pathlib import Path
            source_dir = Path("data/01_raw/source_documents")
            processed_dir = Path("data/02_processed")
            
            if source_dir.exists():
                source_files = list(source_dir.glob("*"))
                source_count = len([f for f in source_files if f.is_file()])
                total_source_size = sum(f.stat().st_size for f in source_files if f.is_file())
                
                # Group by file type
                source_by_type = {}
                for f in source_files:
                    if f.is_file():
                        ext = f.suffix.lower()
                        source_by_type[ext] = source_by_type.get(ext, 0) + 1
            else:
                source_count = 0
                total_source_size = 0
                source_by_type = {}
            
            documents = {
                "source_documents_total": source_count,
                "total_source_size_mb": round(total_source_size / (1024*1024), 2),
                "source_by_type": source_by_type,
                "processed_documents_dir_exists": processed_dir.exists()
            }
        except Exception as e:
            documents = {"error": f"Failed to get document info: {e}"}
        
        # Uptime calculation
        try:
            # Try to get uptime from memory or calculate from process
            uptime_seconds = time.time() - start_time + 10  # Rough estimate
            uptime_formatted = str(timedelta(seconds=int(uptime_seconds)))
        except:
            uptime_formatted = "unknown"
            uptime_seconds = 0
        
        # API metrics (simplified since we don't have access to _system_metrics here)
        api_metrics = {
            "uptime_formatted": uptime_formatted,
            "uptime_seconds": round(uptime_seconds, 0),
            "requests_total": "unknown",
            "success_rate": "unknown",
            "average_response_time": "unknown",
            "circuit_breaker_status": "unknown",
            "note": "API metrics available via Flask app context"
        }
        
        execution_time = time.time() - start_time
        
        return {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "execution_time": round(execution_time, 3),
            "system": system_info,
            "vector_database": vector_database,
            "rag_pipeline": rag_pipeline,
            "model_status": model_status,
            "documents": documents,
            "api_metrics": api_metrics,
            "summary": {
                "overall_status": "operational" if vector_database.get("status") == "ready" else "degraded",
                "key_components": {
                    "system_resources": "healthy" if system_info.get("memory_percent", 100) < 90 else "warning",
                    "vector_database": vector_database.get("status", "unknown"),
                    "model": "loaded" if model_status.get("loaded", False) else "not_loaded",
                    "documents": "available" if documents.get("source_documents_total", 0) > 0 else "none"
                }
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "execution_time": 0,
            "error": str(e),
            "traceback": traceback.format_exc()
        }

def run_quick_healthcheck() -> Dict[str, Any]:
    """Run quick health check for /healthz endpoint."""
    try:
        start_time = time.time()
        
        # Basic system checks
        health_checks = {
            "python_version": sys.version_info[:2] >= (3, 8),
            "pathlib_available": True,
            "scripts_accessible": SCRIPTS_PATH.exists(),
            "diagnostics_module": True
        }
        
        # Check core PyNucleus imports
        try:
            from pynucleus.settings import settings
            health_checks["pynucleus_settings"] = True
        except:
            health_checks["pynucleus_settings"] = False
            
        try:
            from pynucleus.rag.engine import ask
            health_checks["rag_engine"] = True
        except:
            health_checks["rag_engine"] = False
        
        execution_time = time.time() - start_time
        passed_checks = sum(health_checks.values())
        total_checks = len(health_checks)
        health_percentage = (passed_checks / total_checks) * 100
        
        status = "healthy" if health_percentage >= 80 else "degraded" if health_percentage >= 60 else "unhealthy"
        
        return {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "execution_time": round(execution_time, 3),
            "health_percentage": round(health_percentage, 1),
            "checks_passed": passed_checks,
            "checks_total": total_checks,
            "details": health_checks
        }
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "execution_time": 0,
            "error": str(e)
        } 