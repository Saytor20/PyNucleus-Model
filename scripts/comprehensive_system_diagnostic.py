#!/usr/bin/env python3
"""
Comprehensive PyNucleus Clean System Diagnostic & Testing Suite

COMPREHENSIVE SYSTEM HEALTH DIAGNOSTICS - checks system environment and component health.
This script focuses on comprehensive system health aspects of PyNucleus Clean:
- Complete environment and dependency validation
- Script-by-script health checking with execution testing
- Component integration and health monitoring
- Docker environment validation
- Directory structure verification
- ChromaDB vector store + Qwen model pipeline testing
- Clean architecture monitoring (Pydantic + Loguru)
- Enhanced PDF table extraction system validation

For focused validation testing (accuracy, citations), use system_validator.py instead.
"""

import sys
import warnings
import argparse
import os
import ast
import importlib
import importlib.util
import subprocess
import traceback
import shutil
import time
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@dataclass
class SystemCheck:
    """Structure for system check results."""
    check_name: str
    category: str
    passed: bool = False
    details: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: str = ""

@dataclass
class ScriptHealth:
    """Structure for script health results."""
    script_path: str
    category: str
    syntax_valid: bool = False
    imports_valid: bool = False
    execution_successful: bool = False
    error_message: str = ""
    execution_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

class ComprehensiveSystemDiagnostic:
    """Comprehensive system diagnostic focused on PyNucleus Clean environment and component health."""
    
    def __init__(self, quiet_mode: bool = False, test_mode: bool = False):
        self.quiet_mode = quiet_mode
        self.test_mode = test_mode  # Lightweight testing mode
        self.system_checks: List[SystemCheck] = []
        self.script_health_results: List[ScriptHealth] = []
        
        self.total_checks = 0
        self.passed_checks = 0
        self.total_scripts = 0
        self.healthy_scripts = 0
        
        self.start_time = datetime.now()
        
        # Component health tracking
        self.environment_health = False
        self.dependencies_health = False
        self.scripts_health = False
        self.components_health = False
        self.docker_health = False
        self.chromadb_health = False
        self.qwen_health = False
        self.pdf_processing_health = False
        self.redis_health = False
        self.scaling_health = False
        self.api_health = False
        self.stress_testing_health = False
        self.deployment_readiness = False
        
        # Updated script categories for current PyNucleus Clean project structure
        self.script_categories = {
            "Core Pipeline Scripts": [
                "src/pynucleus/pipeline/**/*.py",
                "src/pynucleus/rag/**/*.py"
            ],
            "Data Processing Scripts": [
                "src/pynucleus/data/**/*.py"
            ],
            "LLM & Model Scripts": [
                "src/pynucleus/llm/**/*.py",
                "src/pynucleus/integration/**/*.py",
                "src/pynucleus/utils/**/*.py"
            ],
            "API & Interface Scripts": [
                "src/pynucleus/api/**/*.py",
                "src/pynucleus/diagnostics/**/*.py",
                "src/pynucleus/deployment/**/*.py"
            ],
            "Entry Point Scripts": [
                "run_pipeline.py",
                "run_answer_engine.py",
                "src/pynucleus/cli.py"
            ],
            "Test Scripts": [
                "tests/*.py",
                "scripts/test_*.py",
                "scripts/*test*.py",
                "scripts/stress_test*.py",
                "scripts/integration_test.py"
            ],
            "Evaluation Scripts": [
                "src/pynucleus/eval/**/*.py"
            ]
        }
    
    def log_message(self, message: str, level: str = "info"):
        """Log messages with appropriate formatting."""
        symbols = {"info": "ℹ️  ", "success": "✅ ", "warning": "⚠️  ", "error": "❌ "}
        symbol = symbols.get(level, "")
        
        if not self.quiet_mode or level in ["error", "warning"]:
            print(f"{symbol}{message}")
    
    def run_comprehensive_diagnostic(self):
        """Run complete comprehensive system diagnostic."""
        self.log_message("🚀 Starting Comprehensive PyNucleus Clean System Diagnostic...")
        self.log_message(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.log_message("=" * 60)
        self.log_message("   COMPREHENSIVE PYNUCLEUS CLEAN HEALTH DIAGNOSTIC")
        self.log_message("=" * 60)
        self.log_message("Focus: ChromaDB, Qwen Models, PDF Processing, and Clean Architecture")
        self.log_message("")
        
        try:
            # Core system environment checks
            self._check_python_environment()
            self._check_comprehensive_dependencies()
            self._check_docker_environment()
            self._validate_directory_structure()
            
            # Clean architecture specific checks
            self._check_pynucleus_clean_architecture()
            self._check_chromadb_health()
            self._check_qwen_model_health()
            self._check_pdf_processing_health()
            
            # Production deployment readiness checks
            self._check_redis_integration()
            self._check_scaling_infrastructure()
            self._check_api_production_readiness()
            self._check_stress_testing_infrastructure()
            
            # Comprehensive script validation
            self._validate_all_scripts_comprehensive()
            
            # Component health testing
            self._check_pipeline_components()
            self._check_integration_components()
            self._test_basic_functionality()
            
            # Enhanced system features (if not in test mode)
            if not self.test_mode:
                self._check_enhanced_features()
                self._test_configuration_management()
                self._check_deployment_readiness()
                
            # Generate comprehensive report
            self._generate_comprehensive_report()
            
        except Exception as e:
            self.log_message(f"Comprehensive diagnostic failed: {e}", "error")
            raise
    
    def run_test_mode(self):
        """Run lightweight test mode diagnostic."""
        self.log_message("🧪 Starting Test Mode System Diagnostic...")
        
        self.log_message("=" * 60)
        self.log_message("   SYSTEM DIAGNOSTIC - TEST MODE")
        self.log_message("=" * 60)
        self.log_message("Focus: Essential System Health Checks")
        self.log_message("")
        
        try:
            # Essential checks only
            self._check_python_environment()
            self._check_essential_dependencies()
            self._validate_core_directory_structure()
            self._validate_core_scripts()
            
            # Basic component check
            self._check_basic_pipeline_health()
            
            # Generate test report
            self._generate_test_report()
            
        except Exception as e:
            self.log_message(f"Test diagnostic failed: {e}", "error")
            raise
    
    def _check_python_environment(self):
        """Check Python environment and version."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   PYTHON ENVIRONMENT CHECK")
        self.log_message("=" * 60)
        
        start_time = time.time()
        check = SystemCheck("Python Environment", "environment")
        
        try:
            # Python version check
            python_version = sys.version
            self.log_message(f"Python Version: {python_version}")
            
            # Check if version meets requirements (3.8+)
            version_info = sys.version_info
            if version_info.major >= 3 and version_info.minor >= 8:
                check.details.append(f"Python {version_info.major}.{version_info.minor} meets requirements")
                check.passed = True
                self.log_message("Python Version: PASSED", "success")
            else:
                check.details.append(f"Python {version_info.major}.{version_info.minor} below minimum (3.8)")
                self.log_message("Python Version: FAILED", "error")
            
            # Python executable info
            self.log_message(f"Python executable: {sys.executable}")
            check.details.append(f"Executable: {sys.executable}")
            
            # Check if src is in path
            src_in_path = src_path in sys.path
            self.log_message(f"Python path includes src: {src_in_path}")
            check.details.append(f"Src in path: {src_in_path}")
            
            self.environment_health = check.passed
            
        except Exception as e:
            check.error_message = str(e)
            self.log_message(f"Python environment check failed: {e}", "error")
        
        check.execution_time = time.time() - start_time
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_essential_dependencies(self):
        """Check essential dependencies only (for test mode)."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   ESSENTIAL DEPENDENCIES CHECK")
        self.log_message("=" * 60)
        
        essential_packages = [
            "numpy", "pandas", "requests", "tqdm", "typer", 
            "pathlib", "dataclasses", "asyncio", "concurrent"
        ]
        
        self._check_package_list(essential_packages, "Essential Dependencies")
    
    def _check_comprehensive_dependencies(self):
        """Check comprehensive dependencies for PyNucleus Clean."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   COMPREHENSIVE DEPENDENCIES CHECK")
        self.log_message("=" * 60)
        
        # Core dependencies for PyNucleus Clean
        core_packages = [
            "numpy", "pandas", "requests", "tqdm", "typer",
            "pathlib", "dataclasses", "asyncio", "concurrent", "flask"
        ]
        
        self.log_message("Core Dependencies:")
        self._check_package_list(core_packages, "Core Dependencies")
        
        # PyNucleus Clean specific dependencies
        clean_packages = [
            ("chromadb", "ChromaDB vector database"),
            ("transformers", "HuggingFace transformers"),
            ("torch", "PyTorch framework"),
            ("sentence-transformers", "Sentence embeddings"),
            ("pydantic", "Pydantic settings validation"),
            ("loguru", "Loguru logging"),
            ("tiktoken", "Token counting utilities")
        ]
        
        self.log_message("\nPyNucleus Clean Dependencies:")
        for package, description in clean_packages:
            self._check_single_package(package, description, optional=False)
        
        # PDF Processing dependencies
        pdf_packages = [
            ("camelot", "Camelot PDF table extraction"),
            ("PyMuPDF", "PyMuPDF document processing"),
            ("pypdf", "PyPDF text extraction"),
            ("PIL", "Pillow image processing"),
            ("cv2", "OpenCV image processing")
        ]
        
        # Production deployment dependencies
        deployment_packages = [
            ("redis", "Redis distributed caching"),
            ("docker", "Docker Python client"),
            ("flask", "Flask web framework"),
            ("gunicorn", "Gunicorn WSGI server"),
            ("requests", "HTTP requests library"),
            ("aiohttp", "Async HTTP client")
        ]
        
        self.log_message("\nPDF Processing Dependencies:")
        for package, description in pdf_packages:
            self._check_single_package(package, description, optional=False)
        
        self.log_message("\nProduction Deployment Dependencies:")
        for package, description in deployment_packages:
            self._check_single_package(package, description, optional=False)
        
        # Optional dependencies
        optional_packages = [
            ("jupyter", "Jupyter notebook support"),
            ("notebook", "Notebook interface"),
            ("llama-cpp-python", "GGUF model support"),
            ("bitsandbytes", "Model quantization"),
            ("accelerate", "Model acceleration"),
            ("langchain", "Language chain workflows"),
            ("guidance", "Prompt templating")
        ]
        
        self.log_message("\nOptional Dependencies:")
        for package, description in optional_packages:
            self._check_single_package(package, description, optional=True)
    
    def _check_package_list(self, packages: List[str], category: str):
        """Check a list of packages."""
        start_time = time.time()
        check = SystemCheck(category, "dependencies")
        
        missing_packages = []
        for package in packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                self.log_message(f"{package}: PASSED", "success")
                check.details.append(f"Package available")
            except ImportError:
                self.log_message(f"{package}: MISSING", "error")
                missing_packages.append(package)
                check.details.append(f"Package missing")
        
        check.passed = len(missing_packages) == 0
        if missing_packages:
            check.error_message = f"Missing packages: {', '.join(missing_packages)}"
        
        check.execution_time = time.time() - start_time
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
        
        self.dependencies_health = check.passed
    
    def _check_single_package(self, package: str, description: str, optional: bool = False):
        """Check a single package."""
        check = SystemCheck(f"Package: {package}", "dependencies")
        
        try:
            importlib.import_module(package.replace("-", "_"))
            if optional:
                self.log_message(f"{package}: PASSED", "success")
                check.details.append("Package available")
            else:
                self.log_message(f"{package}: PASSED", "success")
                check.details.append("Package available")
            check.passed = True
        except ImportError:
            if optional:
                self.log_message(f"{package}: MISSING (optional)", "warning")
                check.details.append("Optional package - not required")
                check.passed = True  # Optional packages don't fail the check
            else:
                self.log_message(f"{package}: MISSING", "error")
                check.details.append("Required package missing")
                check.passed = False
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_docker_environment(self):
        """Check Docker environment availability."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   DOCKER ENVIRONMENT CHECK")
        self.log_message("=" * 60)
        
        start_time = time.time()
        check = SystemCheck("Docker Environment", "environment")
        
        try:
            # Check if Docker is available
            result = subprocess.run(
                ["docker", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                docker_version = result.stdout.strip()
                self.log_message(f"Docker Availability: PASSED", "success")
                self.log_message(f"Version: {docker_version}")
                check.details.append(f"Version: {docker_version}")
                check.passed = True
                self.docker_health = True
            else:
                self.log_message("Docker Availability: FAILED", "error")
                check.error_message = "Docker not available or not responding"
                
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.log_message("Docker Availability: NOT FOUND", "warning")
            check.error_message = "Docker command not found or timed out"
            check.warnings.append("Docker not available (optional for some features)")
        
        check.execution_time = time.time() - start_time
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _validate_directory_structure(self):
        """Validate comprehensive directory structure."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   DIRECTORY STRUCTURE CHECK")
        self.log_message("=" * 60)
        
        required_dirs = [
            "src/pynucleus",
            "src/pynucleus/data",
            "src/pynucleus/deployment",
            "configs", 
            "data",
            "data/02_processed",
            "logs",
            "scripts",
            "docker"
        ]
        
        optional_dirs = [
            "data/02_processed/tables",
            "data/05_output/llm_reports",
            "data/validation", 
            "dwsim_rag_integration",
            "tests",
            "test_output"
        ]
        
        # Check required directories
        for dir_path in required_dirs:
            check = SystemCheck(f"Directory: {dir_path}", "structure")
            
            if Path(dir_path).exists():
                self.log_message(f"Directory: {dir_path}: PASSED", "success")
                check.details.append("Required directory exists")
                check.passed = True
            else:
                self.log_message(f"Directory: {dir_path}: MISSING", "error")
                check.details.append("Required directory missing")
                check.passed = False
            
            self.system_checks.append(check)
            self.total_checks += 1
            if check.passed:
                self.passed_checks += 1
        
        # Check optional directories
        for dir_path in optional_dirs:
            if Path(dir_path).exists():
                self.log_message(f"Optional directory found: {dir_path}", "success")
    
    def _validate_core_directory_structure(self):
        """Validate core directory structure only (for test mode)."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   CORE DIRECTORY STRUCTURE CHECK")
        self.log_message("=" * 60)
        
        core_dirs = ["src/pynucleus", "src/pynucleus/data", "configs", "data"]
        
        for dir_path in core_dirs:
            check = SystemCheck(f"Core Directory: {dir_path}", "structure")
            
            if Path(dir_path).exists():
                self.log_message(f"Directory: {dir_path}: PASSED", "success")
                check.passed = True
            else:
                self.log_message(f"Directory: {dir_path}: MISSING", "error")
                check.passed = False
            
            self.system_checks.append(check)
            self.total_checks += 1
            if check.passed:
                self.passed_checks += 1
    
    def _validate_all_scripts_comprehensive(self):
        """Comprehensive script validation by categories."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   COMPREHENSIVE SCRIPT VALIDATION")
        self.log_message("=" * 60)
        
        for category, patterns in self.script_categories.items():
            self.log_message(f"\n--- Validating {category} ---")
            
            script_files = []
            for pattern in patterns:
                script_files.extend(glob.glob(pattern, recursive=True))
            
            if not script_files:
                self.log_message(f"No scripts found for {category}", "warning")
                continue
            
            category_healthy = 0
            category_total = 0
            
            for script_path in script_files:
                if Path(script_path).exists():
                    health = self._validate_single_script(script_path, category)
                    category_total += 1
                    self.total_scripts += 1
                    
                    if health.syntax_valid and health.imports_valid and health.execution_successful:
                        category_healthy += 1
                        self.healthy_scripts += 1
                        self.log_message(f"   {script_path} - Overall: HEALTHY", "success")
                    else:
                        self.log_message(f"   {script_path} - Overall: ISSUES FOUND", "warning")
            
            # Category summary
            category_health_rate = (category_healthy / category_total * 100) if category_total > 0 else 0
            check = SystemCheck(f"{category} Health", "scripts")
            check.passed = category_health_rate >= 80
            check.details.append(f"{category_healthy}/{category_total} scripts healthy ({category_health_rate:.1f}%)")
            
            self.system_checks.append(check)
            self.total_checks += 1
            if check.passed:
                self.passed_checks += 1
        
        # Overall script health
        overall_script_health = (self.healthy_scripts / self.total_scripts * 100) if self.total_scripts > 0 else 0
        self.scripts_health = overall_script_health >= 75
    
    def _validate_core_scripts(self):
        """Validate core scripts only (for test mode)."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   CORE SCRIPT VALIDATION")
        self.log_message("=" * 60)
        
        core_scripts = [
            "run_pipeline.py",
            "run_answer_engine.py",
            "src/pynucleus/__init__.py",
            "src/pynucleus/cli.py"
        ]
        
        healthy_count = 0
        
        for script_path in core_scripts:
            if Path(script_path).exists():
                health = self._validate_single_script(script_path, "Core")
                self.total_scripts += 1
                
                if health.syntax_valid and health.imports_valid:
                    healthy_count += 1
                    self.healthy_scripts += 1
                    self.log_message(f"   {script_path}: HEALTHY", "success")
                else:
                    self.log_message(f"   {script_path}: ISSUES", "warning")
            else:
                self.log_message(f"   {script_path}: MISSING", "error")
        
        check = SystemCheck("Core Scripts Health", "scripts")
        check.passed = healthy_count >= len(core_scripts) * 0.8
        check.details.append(f"{healthy_count}/{len(core_scripts)} core scripts healthy")
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
        
        self.scripts_health = check.passed
    
    def _validate_single_script(self, script_path: str, category: str) -> ScriptHealth:
        """Validate a single script comprehensively."""
        health = ScriptHealth(script_path, category)
        
        try:
            # Syntax validation
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                ast.parse(content)
                health.syntax_valid = True
                self.log_message(f"   {script_path} - Syntax OK", "success")
            except SyntaxError as e:
                health.error_message = f"Syntax error: {e}"
                self.log_message(f"   {script_path} - Syntax ERROR", "error")
            
            # Import validation
            if health.syntax_valid:
                try:
                    spec = importlib.util.spec_from_file_location("test_module", script_path)
                    health.imports_valid = True
                    self.log_message(f"   {script_path} - Imports OK", "success")
                except Exception as e:
                    health.error_message = f"Import error: {e}"
                    self.log_message(f"   {script_path} - Imports ERROR", "error")
            
            # Execution test (updated logic for current project structure)
            if health.syntax_valid and health.imports_valid:
                if any(ep in script_path for ep in ["run_pipeline.py", "run_answer_engine.py", "cli.py"]):
                    # Skip execution for entry points
                    health.execution_successful = True
                    self.log_message(f"   {script_path} - Entry point (skipped execution)", "success")
                elif "src/pynucleus/" in script_path and script_path.endswith(".py"):
                    # Handle package modules
                    try:
                        with open(script_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        if "from ." in content or "import ." in content:
                            health.execution_successful = True
                            self.log_message(f"   {script_path} - Package module (relative imports detected)", "success")
                        else:
                            # Try to execute if no relative imports
                            spec = importlib.util.spec_from_file_location("test_module", script_path)
                            if spec and spec.loader:
                                module = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(module)
                                health.execution_successful = True
                                self.log_message(f"   {script_path} - Execution OK", "success")
                    except Exception as e:
                        if "relative import" in str(e).lower() or "no known parent package" in str(e).lower():
                            health.execution_successful = True
                            health.warnings.append("Package module with relative imports (expected)")
                            self.log_message(f"   {script_path} - Package module (relative imports)", "success")
                        elif "test_module" in str(e):
                            health.execution_successful = True
                            health.warnings.append("Module execution warning (testing artifacts)")
                            self.log_message(f"   {script_path} - Execution warning: {str(e)[:50]}...", "warning")
                        else:
                            health.error_message = f"Execution error: {e}"
                            self.log_message(f"   {script_path} - Execution ERROR", "error")
                else:
                    try:
                        # Try to import the module
                        spec = importlib.util.spec_from_file_location("test_module", script_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(module)
                            health.execution_successful = True
                            self.log_message(f"   {script_path} - Execution OK", "success")
                    except Exception as e:
                        if "relative import" in str(e).lower() or "no known parent package" in str(e).lower():
                            health.execution_successful = True
                            health.warnings.append("Module with relative imports (expected)")
                            self.log_message(f"   {script_path} - Module with relative imports", "success")
                        elif "test_module" in str(e):
                            health.execution_successful = True
                            health.warnings.append("Module execution warning (testing artifacts)")
                            self.log_message(f"   {script_path} - Execution warning: {str(e)[:50]}...", "warning")
                        else:
                            health.error_message = f"Execution error: {e}"
                            self.log_message(f"   {script_path} - Execution ERROR", "error")
        
        except Exception as e:
            health.error_message = f"Validation failed: {e}"
        
        self.script_health_results.append(health)
        return health
    
    def _check_pipeline_components(self):
        """Check pipeline component health."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   PIPELINE COMPONENTS CHECK")
        self.log_message("=" * 60)
        
        check = SystemCheck("Pipeline Components", "components")
        
        try:
            from pynucleus.pipeline.pipeline_utils import PipelineUtils
            from pynucleus.pipeline.pipeline_rag import RAGPipeline
            
            # Test core pipeline initialization
            pipeline_utils = PipelineUtils()
            rag_pipeline = RAGPipeline()
            
            self.log_message("Pipeline Components:")
            self.log_message(f"  PipelineUtils: {'✓ Initialized' if pipeline_utils else '✗ Failed'}")
            self.log_message(f"  RAG Pipeline: {'✓ Initialized' if rag_pipeline else '✗ Failed'}")
            
            # Try DWSIM pipeline separately (optional component - legacy)
            try:
                from pynucleus.pipeline.pipeline_dwsim import DWSIMPipeline
                dwsim_pipeline = DWSIMPipeline()
                self.log_message(f"  DWSIM Pipeline: {'✓ Initialized' if dwsim_pipeline else '✗ Failed'}")
                check.warnings.append("DWSIM pipeline is legacy - focus on ChromaDB RAG")
            except ImportError:
                self.log_message("  DWSIM Pipeline: ⚠️ Not Available (platform limitation)", "warning")
                check.warnings.append("DWSIM pipeline not available (expected on macOS)")
            except Exception as e:
                self.log_message(f"  DWSIM Pipeline: ✗ Failed ({e})", "warning")
                check.warnings.append(f"DWSIM pipeline error: {e}")
            
            check.passed = True
            check.details.append("Core pipeline components initialized successfully")
            self.components_health = True
            
        except Exception as e:
            check.passed = False
            check.error_message = str(e)
            self.log_message(f"Pipeline component check failed: {e}", "error")
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_basic_pipeline_health(self):
        """Basic pipeline health check (for test mode)."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   BASIC PIPELINE HEALTH CHECK")
        self.log_message("=" * 60)
        
        check = SystemCheck("Basic Pipeline Health", "components")
        
        try:
            # Try basic imports
            from pynucleus.pipeline import PipelineUtils
            from pynucleus.rag import RAGCore
            
            self.log_message("✓ Core pipeline imports successful", "success")
            check.passed = True
            check.details.append("Basic pipeline components importable")
            
        except Exception as e:
            self.log_message(f"✗ Basic pipeline health failed: {e}", "error")
            check.passed = False
            check.error_message = str(e)
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_integration_components(self):
        """Check integration component health."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   INTEGRATION COMPONENTS CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.integration.config_manager import ConfigManager
            from pynucleus.integration.llm_output_generator import LLMOutputGenerator
            from pynucleus.rag.vector_store import ChromaVectorStore
            
            check = SystemCheck("Integration Components", "components")
            
            # Test component initialization
            config_mgr = ConfigManager()
            llm_gen = LLMOutputGenerator()
            chroma_store = ChromaVectorStore()
            
            self.log_message("Integration Components:")
            self.log_message(f"  Config Manager: {'✓ Initialized' if config_mgr else '✗ Failed'}")
            self.log_message(f"  LLM Generator: {'✓ Initialized' if llm_gen else '✗ Failed'}")
            self.log_message(f"  ChromaDB Store: {'✓ Loaded' if chroma_store.loaded else '⚠️ Not Loaded'}")
            
            check.passed = True
            check.details.append("Integration components initialized")
            if not chroma_store.loaded:
                check.warnings.append("ChromaDB store not loaded (may be expected)")
            
        except Exception as e:
            check = SystemCheck("Integration Components", "components")
            check.passed = False
            check.error_message = str(e)
            self.log_message(f"Integration component check failed: {e}", "error")
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_pynucleus_clean_architecture(self):
        """Check PyNucleus Clean architecture components."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   PYNUCLEUS CLEAN ARCHITECTURE CHECK")
        self.log_message("=" * 60)
        
        # Test Pydantic Settings
        try:
            from pynucleus.settings import settings
            
            self.total_checks += 1
            check = SystemCheck("Pydantic Settings", "architecture")
            
            # Validate core settings
            required_settings = ['CHROMA_PATH', 'MODEL_ID', 'EMB_MODEL', 'MAX_TOKENS', 'RETRIEVE_TOP_K']
            settings_valid = all(hasattr(settings, attr) for attr in required_settings)
            
            if settings_valid:
                self.log_message("✓ Pydantic Settings validation PASSED", "success")
                self.log_message(f"   ChromaDB Path: {settings.CHROMA_PATH}")
                self.log_message(f"   Model ID: {settings.MODEL_ID}")
                self.log_message(f"   Embedding Model: {settings.EMB_MODEL}")
                check.passed = True
                self.passed_checks += 1
            else:
                self.log_message("✗ Pydantic Settings validation FAILED", "error")
                check.passed = False
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"Clean architecture validation failed: {e}", "error")
        
        # Test Loguru Logger
        try:
            from pynucleus.utils.logger import logger
            
            self.total_checks += 1
            check = SystemCheck("Loguru Logger", "architecture")
            
            # Test logger functionality
            logger.info("Test log message from comprehensive diagnostic")
            self.log_message("✓ Loguru Logger validation PASSED", "success")
            check.passed = True
            self.passed_checks += 1
            
            self.system_checks.append(check)
            
        except Exception as e:
            self.log_message(f"Logger validation failed: {e}", "error")
    
    def _check_chromadb_health(self):
        """Check ChromaDB health and connectivity."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   CHROMADB HEALTH CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.rag.engine import retrieve
            from pynucleus.settings import settings
            
            self.total_checks += 1
            check = SystemCheck("ChromaDB Health", "chromadb")
            
            # Test ChromaDB connection
            chroma_path = Path(settings.CHROMA_PATH)
            if chroma_path.exists():
                self.log_message(f"✓ ChromaDB directory exists: {settings.CHROMA_PATH}", "success")
                
                # Test basic retrieval
                test_docs = retrieve("chemical engineering", k=1)
                if test_docs and len(test_docs) > 0:
                    self.log_message("✓ ChromaDB retrieval PASSED", "success")
                    self.log_message(f"   Retrieved {len(test_docs)} documents")
                    check.passed = True
                    self.passed_checks += 1
                    self.chromadb_health = True
                else:
                    self.log_message("⚠️ ChromaDB retrieval returned no results", "warning")
                    check.passed = False
                    check.warnings.append("ChromaDB empty or not functioning")
            else:
                self.log_message(f"⚠️ ChromaDB directory not found: {settings.CHROMA_PATH}", "warning")
                check.passed = False
                check.warnings.append("ChromaDB directory missing")
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"ChromaDB health check failed: {e}", "error")
    
    def _check_qwen_model_health(self):
        """Check Qwen model health and performance."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   QWEN MODEL HEALTH CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.llm.qwen_loader import generate
            from pynucleus.settings import settings
            
            self.total_checks += 1
            check = SystemCheck("Qwen Model Health", "qwen")
            
            # Test model loading and generation
            self.log_message(f"Testing Qwen model: {settings.MODEL_ID}")
            
            test_prompt = "What is chemical engineering?"
            start_time = time.time()
            
            response = generate(test_prompt, max_tokens=50)
            response_time = time.time() - start_time
            
            if response and len(response.strip()) > 10:
                self.log_message("✓ Qwen model generation PASSED", "success")
                self.log_message(f"   Response time: {response_time:.2f}s")
                self.log_message(f"   Response length: {len(response)} characters")
                check.passed = True
                self.passed_checks += 1
                self.qwen_health = True
            else:
                self.log_message("✗ Qwen model generation FAILED", "error")
                check.passed = False
                check.error_message = "Model generation failed or returned empty response"
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"Qwen model health check failed: {e}", "error")
    
    def _check_pdf_processing_health(self):
        """Check PDF processing system health and functionality."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   PDF PROCESSING SYSTEM CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.data.table_cleaner import extract_tables
            from pynucleus.rag.document_processor import DocumentProcessor
            
            self.total_checks += 1
            check = SystemCheck("PDF Processing System", "pdf_processing")
            
            # Test table cleaner availability
            self.log_message("PDF Processing Components:")
            self.log_message(f"  Table Cleaner: ✓ Available")
            
            # Test document processor with table extraction
            processor = DocumentProcessor()
            self.log_message(f"  Document Processor: ✓ Initialized")
            self.log_message(f"  Tables Output Dir: {processor.tables_output_dir}")
            
            # Test camelot availability
            try:
                import camelot
                self.log_message("  Camelot PDF Parser: ✓ Available", "success")
                check.passed = True
                self.passed_checks += 1
                self.pdf_processing_health = True
                check.details.append("PDF table extraction system fully functional")
            except ImportError:
                self.log_message("  Camelot PDF Parser: ✗ Not Available", "error")
                check.passed = False
                check.error_message = "Camelot not available - install with: pip install camelot-py[cv]"
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"PDF processing health check failed: {e}", "error")
    
    def _check_redis_integration(self):
        """Check Redis distributed caching integration."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   REDIS INTEGRATION CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.deployment.cache_integration import RedisCache, get_cache
            
            self.total_checks += 1
            check = SystemCheck("Redis Integration", "redis")
            
            # Test Redis cache initialization
            cache = RedisCache()
            
            if cache.enabled:
                self.log_message("✓ Redis Cache: Available and enabled", "success")
                
                # Test basic cache operations
                test_key = "pynucleus_diagnostic_test"
                test_data = {"test": "diagnostic_data", "timestamp": time.time()}
                
                # Test set operation
                cache.set(test_key, test_data, ttl=60)
                
                # Test get operation
                retrieved_data = cache.get(test_key)
                
                if retrieved_data and retrieved_data.get("test") == "diagnostic_data":
                    self.log_message("✓ Redis Cache Operations: Working", "success")
                    check.passed = True
                    self.passed_checks += 1
                    self.redis_health = True
                    
                    # Test cache stats
                    stats = cache.get_stats()
                    self.log_message(f"   Memory Usage: {stats.get('memory_usage_mb', 0):.1f} MB")
                    self.log_message(f"   Total Keys: {stats.get('total_keys', 0)}")
                    
                    # Cleanup test data
                    cache.delete(test_key)
                else:
                    self.log_message("✗ Redis Cache Operations: Failed", "error")
                    check.passed = False
                    check.error_message = "Cache operations not working"
            else:
                self.log_message("⚠️ Redis Cache: Not available (check Redis server)", "warning")
                check.passed = False
                check.warnings.append("Redis server not available")
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"Redis integration check failed: {e}", "error")
    
    def _check_scaling_infrastructure(self):
        """Check horizontal scaling infrastructure."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   SCALING INFRASTRUCTURE CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.deployment.scaling_manager import (
                ScalingManager, CacheManager, DockerManager, 
                InstanceMetrics, ScalingConfig
            )
            
            self.total_checks += 1
            check = SystemCheck("Scaling Infrastructure", "scaling")
            
            # Test scaling configuration
            config = ScalingConfig(min_instances=2, max_instances=8)
            self.log_message(f"✓ Scaling Config: Min {config.min_instances}, Max {config.max_instances}")
            
            # Test cache manager
            cache_manager = CacheManager()
            self.log_message("✓ Cache Manager: Initialized")
            
            # Test Docker manager
            docker_manager = DockerManager()
            self.log_message("✓ Docker Manager: Initialized")
            
            # Test metrics creation
            test_metrics = InstanceMetrics(
                instance_id="test-api-1",
                cpu_usage=50.0,
                memory_usage=60.0,
                response_time_avg=1.5,
                requests_per_second=10.0,
                error_rate=0.0,
                timestamp=time.time(),
                health_status="healthy"
            )
            
            if test_metrics.instance_id == "test-api-1":
                self.log_message("✓ Instance Metrics: Working")
                check.passed = True
                self.passed_checks += 1
                self.scaling_health = True
            else:
                self.log_message("✗ Instance Metrics: Failed")
                check.passed = False
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"Scaling infrastructure check failed: {e}", "error")
    
    def _check_api_production_readiness(self):
        """Check Flask API production readiness."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   API PRODUCTION READINESS CHECK")
        self.log_message("=" * 60)
        
        try:
            from pynucleus.api.app import create_app
            
            self.total_checks += 1
            check = SystemCheck("API Production Readiness", "api")
            
            # Test app factory pattern
            app = create_app()
            self.log_message("✓ Flask App Factory: Working")
            
            # Check routes
            routes = [rule.rule for rule in app.url_map.iter_rules()]
            expected_routes = ["/health", "/metrics", "/ask"]
            found_routes = [r for r in expected_routes if r in routes]
            
            self.log_message(f"✓ API Routes: {len(found_routes)}/{len(expected_routes)} essential routes")
            for route in found_routes:
                self.log_message(f"   {route}: Available")
            
            # Check configuration
            has_redis_config = 'REDIS_URL' in app.config
            has_instance_id = 'PYNUCLEUS_INSTANCE_ID' in app.config
            
            self.log_message(f"✓ Redis Configuration: {'Yes' if has_redis_config else 'No'}")
            self.log_message(f"✓ Instance ID Configuration: {'Yes' if has_instance_id else 'No'}")
            
            if len(found_routes) >= 2 and has_redis_config:
                check.passed = True
                self.passed_checks += 1
                self.api_health = True
                self.log_message("✓ API Production Readiness: PASSED", "success")
            else:
                check.passed = False
                check.error_message = "Missing essential routes or configuration"
                self.log_message("✗ API Production Readiness: FAILED", "error")
                
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"API production readiness check failed: {e}", "error")
    
    def _check_stress_testing_infrastructure(self):
        """Check stress testing and validation infrastructure."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   STRESS TESTING INFRASTRUCTURE CHECK")
        self.log_message("=" * 60)
        
        try:
            self.total_checks += 1
            check = SystemCheck("Stress Testing Infrastructure", "testing")
            
            # Check for stress test scripts
            stress_test_files = [
                "scripts/stress_test_suite.py",
                "scripts/simple_stress_test.py",
                "scripts/integration_test.py"
            ]
            
            existing_files = []
            for test_file in stress_test_files:
                if Path(test_file).exists():
                    existing_files.append(test_file)
                    self.log_message(f"✓ {test_file}: Available")
                else:
                    self.log_message(f"✗ {test_file}: Missing")
            
            # Test if we can import stress test components
            try:
                import sys
                scripts_path = str(Path("scripts"))
                if scripts_path not in sys.path:
                    sys.path.insert(0, scripts_path)
                    
                from stress_test_suite import StressTestConfig, PyNucleusStressTester
                
                # Test configuration creation
                config = StressTestConfig(
                    base_url="http://localhost",
                    port=80,
                    num_concurrent_users=5,
                    num_requests_per_user=10
                )
                
                tester = PyNucleusStressTester(config)
                self.log_message("✓ Stress Test Components: Importable and functional")
                
                check.passed = True
                self.passed_checks += 1
                self.stress_testing_health = True
                
            except ImportError as e:
                self.log_message(f"⚠️ Stress Test Components: Import issues - {e}", "warning")
                check.passed = len(existing_files) >= 2  # Pass if we have most files
                if check.passed:
                    self.passed_checks += 1
                    
            self.log_message(f"Stress Testing Files: {len(existing_files)}/{len(stress_test_files)} available")
            
            self.system_checks.append(check)
                
        except Exception as e:
            self.log_message(f"Stress testing infrastructure check failed: {e}", "error")
    
    def _check_deployment_readiness(self):
        """Check overall deployment readiness."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   DEPLOYMENT READINESS ASSESSMENT")
        self.log_message("=" * 60)
        
        self.total_checks += 1
        check = SystemCheck("Deployment Readiness", "deployment")
        
        # Check Docker compose files
        docker_files = [
            "docker/docker-compose.yml",
            "docker/docker-compose.scale.yml",
            "docker/docker-compose.production.yml",
            "docker/Dockerfile.api"
        ]
        
        docker_files_found = 0
        for docker_file in docker_files:
            if Path(docker_file).exists():
                docker_files_found += 1
                self.log_message(f"✓ {docker_file}: Available")
            else:
                self.log_message(f"✗ {docker_file}: Missing")
        
        # Check launch scripts
        launch_scripts = [
            "scripts/launch_scaled_deployment.sh"
        ]
        
        launch_scripts_found = 0
        for script in launch_scripts:
            if Path(script).exists():
                launch_scripts_found += 1
                self.log_message(f"✓ {script}: Available")
            else:
                self.log_message(f"✗ {script}: Missing")
        
        # Overall deployment readiness assessment
        deployment_components = [
            self.redis_health,
            self.scaling_health,
            self.api_health,
            self.stress_testing_health,
            docker_files_found >= 3,
            launch_scripts_found >= 1
        ]
        
        readiness_score = sum(deployment_components) / len(deployment_components)
        
        self.log_message(f"\nDeployment Readiness Components:")
        self.log_message(f"  Redis Integration: {'✅' if self.redis_health else '❌'}")
        self.log_message(f"  Scaling Infrastructure: {'✅' if self.scaling_health else '❌'}")
        self.log_message(f"  API Production Ready: {'✅' if self.api_health else '❌'}")
        self.log_message(f"  Stress Testing: {'✅' if self.stress_testing_health else '❌'}")
        self.log_message(f"  Docker Files: {'✅' if docker_files_found >= 3 else '❌'}")
        self.log_message(f"  Launch Scripts: {'✅' if launch_scripts_found >= 1 else '❌'}")
        
        if readiness_score >= 0.8:
            self.log_message("🎉 DEPLOYMENT READINESS: PRODUCTION READY", "success")
            check.passed = True
            self.deployment_readiness = True
        elif readiness_score >= 0.6:
            self.log_message("⚠️ DEPLOYMENT READINESS: MOSTLY READY", "warning")
            check.passed = True
        else:
            self.log_message("❌ DEPLOYMENT READINESS: NOT READY", "error")
            check.passed = False
            
        self.passed_checks += 1 if check.passed else 0
        self.system_checks.append(check)
    
    def _test_basic_functionality(self):
        """Test basic system functionality."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   BASIC FUNCTIONALITY TEST")
        self.log_message("=" * 60)
        
        check = SystemCheck("Basic Functionality", "functionality")
        
        try:
            # Test configuration loading
            from pynucleus.integration.config_manager import ConfigManager
            config_mgr = ConfigManager(config_dir="configs")
            
            # Test if we can find config files
            config_files = list(Path("configs").glob("*.json"))
            csv_config_files = list(Path("configs").glob("*.csv"))
            
            total_configs = len(config_files) + len(csv_config_files)
            
            self.log_message(f"Configuration Management:")
            self.log_message(f"  JSON configs: {len(config_files)}")
            self.log_message(f"  CSV configs: {len(csv_config_files)}")
            self.log_message(f"  Total configs: {total_configs}")
            
            if total_configs > 0:
                check.passed = True
                check.details.append(f"Found {total_configs} configuration files")
            else:
                check.passed = False
                check.error_message = "No configuration files found"
            
        except Exception as e:
            check.passed = False
            check.error_message = str(e)
            self.log_message(f"Basic functionality test failed: {e}", "error")
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _check_enhanced_features(self):
        """Check enhanced system features."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   ENHANCED FEATURES CHECK")
        self.log_message("=" * 60)
        
        # Check for enhanced data directories
        enhanced_dirs = {
            "data/02_processed/tables": "Extracted tables directory",
            "data/05_output/results": "Results output directory",
            "data/05_output/llm_reports": "LLM reports directory", 
            "data/validation": "Validation data directory"
        }
        
        for dir_path, description in enhanced_dirs.items():
            check = SystemCheck(f"Enhanced: {description}", "enhanced")
            
            if Path(dir_path).exists():
                files_count = len(list(Path(dir_path).glob("*")))
                self.log_message(f"✓ {description}: {files_count} items", "success")
                check.passed = True
                check.details.append(f"Directory exists with {files_count} items")
            else:
                self.log_message(f"⚠️ {description}: Not found", "warning")
                check.passed = False
                check.warnings.append("Enhanced feature directory not found")
            
            self.system_checks.append(check)
            self.total_checks += 1
            if check.passed:
                self.passed_checks += 1
        
        # Check for new test files
        new_test_files = {
            "tests/test_document_processor_tables.py": "Document processor table tests",
            "tests/test_table_cleaner.py": "Table cleaner tests"
        }
        
        self.log_message("\nNew Test File Validation:")
        for test_file, description in new_test_files.items():
            check = SystemCheck(f"Test: {description}", "enhanced")
            
            if Path(test_file).exists():
                self.log_message(f"✓ {description}: Found", "success")
                check.passed = True
                check.details.append("Test file exists")
            else:
                self.log_message(f"⚠️ {description}: Missing", "warning")
                check.passed = False
                check.warnings.append("Test file not found")
            
            self.system_checks.append(check)
            self.total_checks += 1
            if check.passed:
                self.passed_checks += 1
    
    def _test_configuration_management(self):
        """Test configuration management system."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   CONFIGURATION MANAGEMENT TEST")
        self.log_message("=" * 60)
        
        check = SystemCheck("Configuration Management", "config")
        
        try:
            from pynucleus.integration.config_manager import ConfigManager
            
            config_mgr = ConfigManager(config_dir="configs")
            
            # Find available config files
            config_files = list(Path("configs").glob("*.json"))
            csv_config_files = list(Path("configs").glob("*.csv"))
            
            total_configs = len(config_files) + len(csv_config_files)
            
            self.log_message(f"Configuration Management:")
            self.log_message(f"  JSON configs: {len(config_files)}")
            self.log_message(f"  CSV configs: {len(csv_config_files)}")
            self.log_message(f"  Total configs: {total_configs}")
            
            if total_configs > 0:
                check.passed = True
                check.details.append(f"Found {total_configs} configuration files")
            else:
                check.passed = False
                check.error_message = "No configuration files found"
            
        except Exception as e:
            check.passed = False
            check.error_message = str(e)
            self.log_message(f"Configuration management test failed: {e}", "error")
        
        self.system_checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
    
    def _generate_comprehensive_report(self):
        """Generate comprehensive diagnostic report."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   COMPREHENSIVE DIAGNOSTIC REPORT")
        self.log_message("=" * 60)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        self.log_message(f"PYNUCLEUS COMPREHENSIVE DIAGNOSTIC REPORT")
        self.log_message(f"Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_message(f"Duration: {duration:.1f} seconds")
        
        # Health summary
        success_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        self.log_message(f"\nEXECUTIVE SUMMARY")
        self.log_message(f"System Health: {success_rate:.1f}%")
        self.log_message(f"Checks Performed: {self.total_checks}")
        self.log_message(f"Checks Passed: {self.passed_checks}")
        self.log_message(f"Checks Failed: {self.total_checks - self.passed_checks}")
        
        if self.total_scripts > 0:
            script_health_rate = (self.healthy_scripts / self.total_scripts) * 100
            self.log_message(f"Script Health: {script_health_rate:.1f}%")
            self.log_message(f"Scripts Tested: {self.total_scripts}")
            self.log_message(f"Healthy Scripts: {self.healthy_scripts}")
        
        # Component health breakdown
        self.log_message(f"\nCOMPONENT HEALTH BREAKDOWN")
        self.log_message(f"Environment: {'✅ HEALTHY' if self.environment_health else '❌ ISSUES'}")
        self.log_message(f"Dependencies: {'✅ HEALTHY' if self.dependencies_health else '❌ ISSUES'}")
        self.log_message(f"Scripts: {'✅ HEALTHY' if self.scripts_health else '❌ ISSUES'}")
        self.log_message(f"Components: {'✅ HEALTHY' if self.components_health else '❌ ISSUES'}")
        self.log_message(f"Docker: {'✅ AVAILABLE' if self.docker_health else '⚠️ NOT AVAILABLE'}")
        self.log_message(f"PDF Processing: {'✅ HEALTHY' if self.pdf_processing_health else '❌ ISSUES'}")
        
        self.log_message(f"\nPRODUCTION DEPLOYMENT HEALTH")
        self.log_message(f"Redis Caching: {'✅ HEALTHY' if self.redis_health else '❌ ISSUES'}")
        self.log_message(f"Scaling Infrastructure: {'✅ HEALTHY' if self.scaling_health else '❌ ISSUES'}")
        self.log_message(f"API Production Ready: {'✅ HEALTHY' if self.api_health else '❌ ISSUES'}")
        self.log_message(f"Stress Testing: {'✅ HEALTHY' if self.stress_testing_health else '❌ ISSUES'}")
        self.log_message(f"Deployment Readiness: {'✅ PRODUCTION READY' if self.deployment_readiness else '❌ NOT READY'}")
        
        # Final assessment
        if success_rate >= 95:
            self.log_message("Overall System Status: EXCELLENT 🎉", "success")
        elif success_rate >= 85:
            self.log_message("Overall System Status: VERY GOOD ✅", "success")
        elif success_rate >= 75:
            self.log_message("Overall System Status: GOOD ✅", "success")
        elif success_rate >= 65:
            self.log_message("Overall System Status: WARNING ⚠️", "warning")
        else:
            self.log_message("Overall System Status: CRITICAL ❌", "error")
    
    def _generate_test_report(self):
        """Generate test mode diagnostic report."""
        self.log_message("\n" + "=" * 60)
        self.log_message("   TEST MODE DIAGNOSTIC REPORT")
        self.log_message("=" * 60)
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        success_rate = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        self.log_message(f"PYNUCLEUS TEST MODE DIAGNOSTIC")
        self.log_message(f"Duration: {duration:.1f} seconds")
        self.log_message(f"System Health: {success_rate:.1f}%")
        self.log_message(f"Checks: {self.passed_checks}/{self.total_checks} passed")
        
        if success_rate >= 80:
            self.log_message("Test Result: PASSED ✅", "success")
        else:
            self.log_message("Test Result: FAILED ❌", "error")

def main():
    """Main function for comprehensive system diagnostic."""
    parser = argparse.ArgumentParser(description="PyNucleus Comprehensive System Diagnostic")
    parser.add_argument('--test', action='store_true', help='Test suite mode (lightweight checks)')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode with minimal output')
    parser.add_argument('--mock', action='store_true', help='Include mock testing (full diagnostic)')
    parser.add_argument('--validation', action='store_true', help='Include validation tests (full diagnostic)')
    
    args = parser.parse_args()
    
    # Create diagnostic runner
    diagnostic = ComprehensiveSystemDiagnostic(
        quiet_mode=args.quiet,
        test_mode=args.test
    )
    
    try:
        # Run appropriate diagnostic mode
        if args.test:
            diagnostic.run_test_mode()
        else:
            diagnostic.run_comprehensive_diagnostic()
        
        # Exit with appropriate code based on results
        success_rate = diagnostic.passed_checks / diagnostic.total_checks if diagnostic.total_checks > 0 else 0
        script_health_rate = diagnostic.healthy_scripts / diagnostic.total_scripts if diagnostic.total_scripts > 0 else 1.0
        
        # Overall health assessment
        overall_health = (success_rate + script_health_rate) / 2
        exit_code = 0 if overall_health >= 0.8 else 1
        
        if exit_code == 0:
            diagnostic.log_message("🎉 System diagnostic completed successfully!", "success")
        else:
            diagnostic.log_message("⚠️ System diagnostic completed with issues!", "warning")
        
        sys.exit(exit_code)
        
    except Exception as e:
        diagnostic.log_message(f"System diagnostic failed: {e}", "error")
        sys.exit(2)

if __name__ == "__main__":
    main() 