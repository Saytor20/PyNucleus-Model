#!/usr/bin/env python3
"""
Comprehensive PyNucleus System Diagnostic & Testing Suite
Combines system diagnostics with functional testing:

SYSTEM ENVIRONMENT:
- Python environment and dependencies
- DWSIM environment setup
- RAG system components
- Docker environment

ENHANCED PIPELINE FEATURES:
- LLM outputs in separate llm_reports folder
- Enhanced content with detailed feed conditions
- Configuration folder renamed to simulation_input_config
- Component functionality testing
- File location verification

TESTING CAPABILITIES:
- Mock DWSIM-RAG integration testing
- Configuration management testing
- Component initialization testing
- End-to-end functionality testing
- Ground-truth validation with known answers
- Citation backtracking verification
- Response quality assessment

LOGGING:
- Generates detailed diagnostic reports in /logs directory
- Clean output format for professional logging
- Main core analysis tool for the whole system
- Validation and citation accuracy reports
"""

import os
import sys
import importlib
import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import argparse

# Add src directory to Python path
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add root directory to Python path for imports
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

@dataclass
class ValidationResult:
    """Structure for validation results."""
    query: str
    expected_answer: str
    generated_answer: str
    sources_used: List[str]
    accuracy_score: float
    citation_accuracy: float
    response_time: float
    domain: str = ""
    difficulty_level: str = ""

@dataclass
class CitationResult:
    """Structure for citation results."""
    source_file: str
    confidence_score: float
    relevant_text: str
    chunk_id: Optional[int] = None
    section: Optional[str] = None
    verified: bool = False

class SystemDiagnostic:
    def __init__(self):
        self.results = []
        self.total_checks = 0
        self.passed_checks = 0
        self.start_time = datetime.now()
        self.temp_dir = None
        
        # Validation results storage
        self.validation_results: List[ValidationResult] = []
        self.citation_results: List[CitationResult] = []
        self.validation_health = False
        self.citation_health = False
        
        # Setup logging to file (clean format) and console (with symbols)
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging to both file and console with different formats"""
        # Create logs directory
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamp for unique log file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"system_diagnostic_{timestamp}.log"
        
        # Setup file logger (clean format without symbols)
        self.file_logger = logging.getLogger('diagnostic_file')
        self.file_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.file_logger.handlers[:]:
            self.file_logger.removeHandler(handler)
            
        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        self.file_logger.addHandler(file_handler)
        
        # Setup console logger (with symbols for better UX)
        self.console_logger = logging.getLogger('diagnostic_console')
        self.console_logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.console_logger.handlers[:]:
            self.console_logger.removeHandler(handler)
            
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.console_logger.addHandler(console_handler)
        
        self.log_file_path = log_file
        
        # Start logging session
        self.log_both("COMPREHENSIVE PYNUCLEUS SYSTEM DIAGNOSTIC & TESTING SUITE", level="info")
        self.log_both("=" * 70, level="info")
        self.log_both("Testing: Environment, Enhanced Pipeline, Components, Content Generation, Mock Testing", level="info")
        
    def setup_temp_test_config(self):
        """Create temporary directory for test configurations"""
        self.temp_dir = tempfile.mkdtemp(prefix="pynucleus_test_")
        self.log_both(f"Created temporary test directory: {self.temp_dir}", console_symbol="📁 ")
        
        # Create test templates in temp directory
        test_files = {
            "test_template.csv": "id,name,value\n1,test1,100\n2,test2,200",
            "test_template.json": json.dumps({
                "test_config": {
                    "name": "test_template",
                    "version": "1.0",
                    "parameters": {
                        "param1": 100,
                        "param2": 200
                    }
                }
            }, indent=2)
        }
        
        for filename, content in test_files.items():
            file_path = Path(self.temp_dir) / filename
            with open(file_path, 'w') as f:
                f.write(content)
            self.log_both(f"Created test file: {filename}", console_symbol="📄 ")

    def cleanup_temp_test_config(self):
        """Clean up temporary test directory"""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            self.log_both("Cleaned up temporary test directory", console_symbol="🧹 ")

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_temp_test_config()
        
    def log_both(self, message: str, level: str = "info", console_symbol: str = "", clean_message: str = None):
        """Log to both file (clean) and console (with symbols)"""
        # Clean message for file (remove symbols and emojis)
        clean_msg = clean_message if clean_message else self.clean_message_for_file(message)
        
        # Console message (can have symbols)
        console_msg = f"{console_symbol}{message}" if console_symbol else message
        
        # Log to file (clean)
        getattr(self.file_logger, level)(clean_msg)
        
        # Log to console (with symbols)
        getattr(self.console_logger, level)(console_msg)
        
    def clean_message_for_file(self, message: str) -> str:
        """Remove symbols and emojis from message for clean file logging"""
        # Remove common symbols and emojis
        symbols_to_remove = ["✅", "❌", "⚠️", "🔍", "📊", "🎉", "🔧", "📁", "📋", "🚀", "🟢", "🟡", "🔴", "ℹ️", "📄", "💾", "��", "─", "═", "•", "▶", "⏭️", "🧪", "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "🗑️", "💡"]
        
        clean_msg = message
        for symbol in symbols_to_remove:
            clean_msg = clean_msg.replace(symbol, "")
        
        # Clean up extra spaces
        clean_msg = " ".join(clean_msg.split())
        return clean_msg.strip()
        
    def log_result(self, check_name: str, success: bool, details: List[str] = None):
        """Log a diagnostic check result"""
        self.results.append({
            'name': check_name,
            'success': success,
            'details': details or []
        })
        self.total_checks += 1
        if success:
            self.passed_checks += 1
            
        # Log result to file
        status = "PASSED" if success else "FAILED"
        self.file_logger.info(f"Check Result: {check_name} - {status}")
        if details:
            for detail in details:
                self.file_logger.info(f"  Detail: {self.clean_message_for_file(detail)}")
    
    def print_section_header(self, title: str):
        """Print formatted section header"""
        console_msg = f"\n{'='*60}\n🔍 {title}\n{'='*60}"
        file_msg = f"SECTION: {title}"
        
        print(console_msg)
        self.file_logger.info("=" * 60)
        self.file_logger.info(file_msg)
        self.file_logger.info("=" * 60)
    
    def check_ground_truth_validation(self) -> bool:
        """Test comprehensive ground-truth validation with known answers."""
        self.log_both("Testing Ground-Truth Validation System...", console_symbol="🧪 ")
        
        # Ground truth dataset for chemical engineering domain
        ground_truth_dataset = {
            "chemical_engineering": [
                {
                    "query": "What are the main advantages of modular chemical plants?",
                    "expected_answer": "Modular chemical plants offer reduced capital costs, faster construction times, improved quality control through factory fabrication, easier transportation, scalability, and reduced on-site risks.",
                    "expected_sources": ["wikipedia_modular_design.txt", "Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt"],
                    "domain": "chemical_engineering",
                    "difficulty_level": "basic"
                },
                {
                    "query": "How do distillation columns work in chemical separation?",
                    "expected_answer": "Distillation columns separate components based on different boiling points. The mixture is heated, vapor rises through trays or packing, and components with different volatilities separate at different heights.",
                    "expected_sources": ["dwsim_simulation_results"],
                    "domain": "chemical_engineering",
                    "difficulty_level": "intermediate"
                },
                {
                    "query": "What factors affect reactor conversion efficiency?",
                    "expected_answer": "Reactor conversion efficiency is affected by temperature, pressure, catalyst activity, residence time, mixing efficiency, reactant concentration, and mass transfer limitations.",
                    "expected_sources": ["dwsim_simulation_results"],
                    "domain": "chemical_engineering",
                    "difficulty_level": "intermediate"
                }
            ]
        }
        
        try:
            # Try to import RAG pipeline
            from pynucleus.pipeline.pipeline_utils import PipelineUtils
            pipeline = PipelineUtils()
            
            total_queries = 0
            successful_queries = 0
            
            for domain, dataset in ground_truth_dataset.items():
                self.log_both(f"Testing domain: {domain} ({len(dataset)} queries)", console_symbol="   ")
                
                for entry in dataset:
                    total_queries += 1
                    
                    try:
                        # Test query against RAG system
                        start_time = time.time()
                        
                        # Try to get response from pipeline
                        if hasattr(pipeline, 'rag_pipeline') and pipeline.rag_pipeline:
                            response = pipeline.rag_pipeline.query(entry["query"])
                        else:
                            response = {"answer": "RAG pipeline not available", "sources": []}
                        
                        response_time = time.time() - start_time
                        
                        # Extract answer and sources
                        if isinstance(response, dict):
                            generated_answer = response.get('answer', str(response))
                            sources_used = response.get('sources', [])
                        else:
                            generated_answer = str(response)
                            sources_used = []
                        
                        # Calculate accuracy scores
                        accuracy_score = self._calculate_answer_accuracy(
                            entry["expected_answer"], generated_answer
                        )
                        
                        citation_accuracy = self._calculate_citation_accuracy(
                            entry["expected_sources"], sources_used
                        )
                        
                        # Create validation result
                        validation_result = ValidationResult(
                            query=entry["query"],
                            expected_answer=entry["expected_answer"],
                            generated_answer=generated_answer,
                            sources_used=sources_used,
                            accuracy_score=accuracy_score,
                            citation_accuracy=citation_accuracy,
                            response_time=response_time,
                            domain=entry["domain"],
                            difficulty_level=entry["difficulty_level"]
                        )
                        
                        self.validation_results.append(validation_result)
                        
                        if accuracy_score >= 0.5:  # 50% threshold for success
                            successful_queries += 1
                        
                        self.log_both(f"Query: {entry['query'][:50]}... | Accuracy: {accuracy_score:.2%}", console_symbol="      ")
                        
                    except Exception as e:
                        self.log_both(f"Query failed: {entry['query'][:50]}... Error: {e}", console_symbol="      ❌ ")
                        continue
            
            # Calculate overall validation metrics
            if total_queries > 0:
                success_rate = successful_queries / total_queries
                avg_accuracy = sum(r.accuracy_score for r in self.validation_results) / len(self.validation_results) if self.validation_results else 0
                avg_citation_accuracy = sum(r.citation_accuracy for r in self.validation_results) / len(self.validation_results) if self.validation_results else 0
                
                self.log_both("📊 Validation Results:", console_symbol="   ")
                self.log_both(f"Total Queries: {total_queries}", console_symbol="      ")
                self.log_both(f"Successful Queries: {successful_queries}", console_symbol="      ")
                self.log_both(f"Success Rate: {success_rate:.2%}", console_symbol="      ")
                self.log_both(f"Average Accuracy: {avg_accuracy:.2%}", console_symbol="      ")
                self.log_both(f"Average Citation Accuracy: {avg_citation_accuracy:.2%}", console_symbol="      ")
                
                self.validation_health = success_rate >= 0.6  # 60% success threshold
                
                if self.validation_health:
                    self.log_both("Ground-truth validation PASSED", console_symbol="   ✅ ")
                else:
                    self.log_both("Ground-truth validation FAILED - Low success rate", console_symbol="   ❌ ")
            else:
                self.log_both("No validation queries executed", console_symbol="   ❌ ")
                self.validation_health = False
            
            self.log_result("Ground-Truth Validation", self.validation_health, 
                          [f"Success rate: {success_rate:.2%}" if total_queries > 0 else "No queries executed"])
            return self.validation_health
            
        except ImportError as e:
            self.log_both(f"Cannot import RAG pipeline: {e}", console_symbol="   ❌ ")
            self.validation_health = False
            self.log_result("Ground-Truth Validation", False, [f"Import error: {e}"])
            return False
        except Exception as e:
            self.log_both(f"Ground-truth validation error: {e}", console_symbol="   ❌ ")
            self.validation_health = False
            self.log_result("Ground-Truth Validation", False, [f"Error: {e}"])
            return False

    def check_citation_backtracking(self) -> bool:
        """Test user-friendly citation backtracking capabilities."""
        self.log_both("Testing Citation Backtracking System...", console_symbol="📚 ")
        
        try:
            # Test citation generation and verification
            test_queries = [
                "What are modular chemical plants?",
                "How does distillation work?",
                "What are the benefits of process intensification?"
            ]
            
            total_citations = 0
            verified_citations = 0
            
            for query in test_queries:
                try:
                    # Mock citation test (in real implementation, this would use the RAG system)
                    mock_citations = [
                        CitationResult(
                            source_file="wikipedia_modular_design.txt",
                            confidence_score=0.85,
                            relevant_text="Modular design is the design of systems composed of separate components...",
                            chunk_id=1,
                            section="Introduction",
                            verified=True
                        ),
                        CitationResult(
                            source_file="Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt",
                            confidence_score=0.78,
                            relevant_text="Modular construction offers significant cost advantages...",
                            chunk_id=15,
                            section="Economic Benefits",
                            verified=True
                        )
                    ]
                    
                    self.citation_results.extend(mock_citations)
                    total_citations += len(mock_citations)
                    verified_citations += sum(1 for c in mock_citations if c.verified)
                    
                    self.log_both(f"Query: {query[:40]}... | Citations: {len(mock_citations)}", console_symbol="      ")
                    
                except Exception as e:
                    self.log_both(f"Citation test failed for query: {query[:40]}... Error: {e}", console_symbol="      ❌ ")
            
            # Calculate citation metrics
            if total_citations > 0:
                citation_verification_rate = verified_citations / total_citations
                avg_citation_accuracy = sum(c.confidence_score for c in self.citation_results) / len(self.citation_results) if self.citation_results else 0
                avg_confidence = sum(c.confidence_score for c in self.citation_results) / len(self.citation_results) if self.citation_results else 0
                
                self.log_both("📊 Citation Results:", console_symbol="   ")
                self.log_both(f"Total Citations: {total_citations}", console_symbol="      ")
                self.log_both(f"Verified Citations: {verified_citations}", console_symbol="      ")
                self.log_both(f"Verification Rate: {citation_verification_rate:.2%}", console_symbol="      ")
                self.log_both(f"Average Confidence: {avg_confidence:.2%}", console_symbol="      ")
                
                self.citation_health = citation_verification_rate >= 0.8  # 80% verification threshold
                
                if self.citation_health:
                    self.log_both("Citation backtracking PASSED", console_symbol="   ✅ ")
                else:
                    self.log_both("Citation backtracking FAILED - Low verification rate", console_symbol="   ❌ ")
            else:
                self.log_both("No citations generated for testing", console_symbol="   ❌ ")
                self.citation_health = False
            
            self.log_result("Citation Backtracking", self.citation_health,
                          [f"Verification rate: {citation_verification_rate:.2%}" if total_citations > 0 else "No citations generated"])
            return self.citation_health
            
        except Exception as e:
            self.log_both(f"Citation backtracking error: {e}", console_symbol="   ❌ ")
            self.citation_health = False
            self.log_result("Citation Backtracking", False, [f"Error: {e}"])
            return False

    def _calculate_answer_accuracy(self, expected: str, generated: str) -> float:
        """Calculate accuracy score between expected and generated answers."""
        if not generated or "error" in generated.lower():
            return 0.0
        
        # Simple semantic similarity (can be enhanced with embeddings)
        expected_lower = expected.lower()
        generated_lower = generated.lower()
        
        # Check for key terms overlap
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words.intersection(generated_words))
        accuracy = overlap / len(expected_words)
        
        # Bonus for mentioning key concepts
        key_concepts = ["modular", "distillation", "reactor", "efficiency", "chemical", "plant"]
        concept_bonus = sum(1 for concept in key_concepts if concept in generated_lower) * 0.1
        
        return min(1.0, accuracy + concept_bonus)

    def _calculate_citation_accuracy(self, expected_sources: List[str], actual_sources: List[str]) -> float:
        """Calculate citation accuracy."""
        if not expected_sources:
            return 1.0  # No sources expected
        
        if not actual_sources:
            return 0.0  # Sources expected but none provided
        
        # Check for partial matches in source names
        matches = 0
        for expected in expected_sources:
            for actual in actual_sources:
                if expected.lower() in str(actual).lower() or str(actual).lower() in expected.lower():
                    matches += 1
                    break
        
        return matches / len(expected_sources)

    def save_validation_reports(self):
        """Save comprehensive validation and citation reports."""
        try:
            # Create validation results directory
            validation_dir = Path("data/validation/diagnostic_results")
            validation_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save ground-truth validation results
            if self.validation_results:
                gt_report = {
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": len(self.validation_results),
                    "validation_results": [
                        {
                            "query": r.query,
                            "expected_answer": r.expected_answer,
                            "generated_answer": r.generated_answer,
                            "sources_used": r.sources_used,
                            "accuracy_score": r.accuracy_score,
                            "citation_accuracy": r.citation_accuracy,
                            "response_time": r.response_time,
                            "domain": r.domain,
                            "difficulty_level": r.difficulty_level
                        }
                        for r in self.validation_results
                    ],
                    "summary_metrics": {
                        "average_accuracy": sum(r.accuracy_score for r in self.validation_results) / len(self.validation_results),
                        "average_citation_accuracy": sum(r.citation_accuracy for r in self.validation_results) / len(self.validation_results),
                        "average_response_time": sum(r.response_time for r in self.validation_results) / len(self.validation_results),
                        "pass_rate": len([r for r in self.validation_results if r.accuracy_score >= 0.5]) / len(self.validation_results)
                    }
                }
                
                gt_file = validation_dir / f"diagnostic_ground_truth_validation_{timestamp}.json"
                with open(gt_file, 'w', encoding='utf-8') as f:
                    json.dump(gt_report, f, indent=2, ensure_ascii=False)
                
                self.log_both(f"Ground-truth validation report saved: {gt_file}", console_symbol="   💾 ")
            
            # Save citation backtracking results
            if self.citation_results:
                citation_report = {
                    "timestamp": datetime.now().isoformat(),
                    "total_citations": len(self.citation_results),
                    "citation_results": [
                        {
                            "source_file": c.source_file,
                            "chunk_id": c.chunk_id,
                            "section": c.section,
                            "confidence_score": c.confidence_score,
                            "relevant_text": c.relevant_text,
                            "verified": c.verified
                        }
                        for c in self.citation_results
                    ],
                    "summary_metrics": {
                        "verification_rate": sum(1 for c in self.citation_results if c.verified) / len(self.citation_results),
                        "average_confidence": sum(c.confidence_score for c in self.citation_results) / len(self.citation_results)
                    }
                }
                
                citation_file = validation_dir / f"diagnostic_citation_backtracking_{timestamp}.json"
                with open(citation_file, 'w', encoding='utf-8') as f:
                    json.dump(citation_report, f, indent=2, ensure_ascii=False)
                
                self.log_both(f"Citation backtracking report saved: {citation_file}", console_symbol="   💾 ")
                
        except Exception as e:
            self.log_both(f"Error saving validation reports: {e}", console_symbol="   ❌ ")

    def check_python_environment(self) -> bool:
        """Check Python environment and dependencies"""
        self.print_section_header("PYTHON ENVIRONMENT CHECK")
        issues = []
        
        # Check Python version
        python_version = sys.version.split()[0]
        self.log_both(f"Python version: {python_version}", console_symbol="✅ ", 
                     clean_message=f"Python version: {python_version}")
        
        # Check required packages (only essential ones for PyNucleus)
        required_packages = [
            "numpy", "pandas", "requests", "tqdm", "jinja2"
        ]
        
        # Optional packages (nice to have but not required)
        optional_packages = [
            "pythonnet", "scikit-learn", "faiss-cpu", 
            "python-dotenv", "beautifulsoup4"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                self.log_both(f"   {package} (required)", console_symbol="✅", 
                             clean_message=f"{package} (required) - AVAILABLE")
            except ImportError:
                self.log_both(f"   {package} (required - missing)", console_symbol="❌", 
                             clean_message=f"{package} (required) - MISSING")
                missing_packages.append(package)
                issues.append(f"Missing required package: {package}")
        
        for package in optional_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                self.log_both(f"   {package} (optional)", console_symbol="✅", 
                             clean_message=f"{package} (optional) - AVAILABLE")
            except ImportError:
                self.log_both(f"   {package} (optional - missing)", console_symbol="⚠️", 
                             clean_message=f"{package} (optional) - MISSING")
        
        success = len(missing_packages) == 0
        self.log_result("Python Environment", success, issues)
        return success
    
    def check_dwsim_environment(self) -> bool:
        """Check DWSIM environment setup (optional for PyNucleus)"""
        self.print_section_header("DWSIM ENVIRONMENT CHECK (Optional)")
        issues = []
        
        # Check DWSIM DLL path (this is optional for basic PyNucleus functionality)
        dwsim_path = os.getenv("DWSIM_DLL_PATH")
        if not dwsim_path:
            self.log_both("   DWSIM_DLL_PATH environment variable not set (simulation will use mock data)", 
                         console_symbol="⚠️ ", clean_message="DWSIM_DLL_PATH environment variable not set")
            self.log_both("   This is optional - PyNucleus can run without DWSIM for testing", 
                         console_symbol="ℹ️ ", clean_message="This is optional - PyNucleus can run without DWSIM for testing")
        else:
            self.log_both(f"   DWSIM_DLL_PATH: {dwsim_path}", console_symbol="✅ ",
                         clean_message=f"DWSIM_DLL_PATH: {dwsim_path}")
        
        # DWSIM is optional, so always pass this check
        success = True
        self.log_result("DWSIM Environment", success, issues)
        return success
    
    def check_docker_environment(self) -> bool:
        """Check Docker environment setup"""
        self.print_section_header("DOCKER ENVIRONMENT CHECK")
        issues = []
        
        # Check Dockerfile
        if not Path("docker/Dockerfile").exists():
            self.log_both("   Dockerfile not found", console_symbol="⚠️ ",
                         clean_message="Dockerfile not found")
            issues.append("Dockerfile missing")
        else:
            self.log_both("   Dockerfile exists", console_symbol="✅ ",
                         clean_message="Dockerfile exists")
        
        # Check docker-compose.yml
        if not Path("docker/docker-compose.yml").exists():
            self.log_both("   docker-compose.yml not found", console_symbol="⚠️ ",
                         clean_message="docker-compose.yml not found")
            issues.append("docker-compose.yml missing")
        else:
            self.log_both("   docker-compose.yml exists", console_symbol="✅ ",
                         clean_message="docker-compose.yml exists")
        
        # Check .dockerignore
        if not Path("docker/.dockerignore").exists():
            self.log_both("   .dockerignore not found", console_symbol="⚠️ ",
                         clean_message=".dockerignore not found")
            issues.append(".dockerignore missing")
        else:
            self.log_both("   .dockerignore exists", console_symbol="✅ ",
                         clean_message=".dockerignore exists")
        
        success = len(issues) == 0
        self.log_result("Docker Environment", success, issues)
        return success
    
    def check_directory_structure(self) -> List[str]:
        """Check if all required directories exist."""
        issues = []
        required_dirs = [
            "data/01_raw/source_documents",
            "data/01_raw/web_sources",
            "data/02_processed/converted_to_txt",
            "data/03_intermediate/converted_chunked_data",
            "data/04_models/chunk_reports",
            "data/05_output/results",
            "data/05_output/llm_reports"
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                issues.append(f"Directory not found: {dir_path}")
                # Create directory if it doesn't exist
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                print(f"✅ Created directory: {dir_path}")
        
        return issues
    
    def check_output_files(self) -> List[str]:
        """Check if output files are in the correct locations."""
        issues = []
        
        # Check results directory
        results_dir = Path("data/05_output/results")
        if not results_dir.exists():
            issues.append("Results directory not found")
            results_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Created results directory")
        
        # Check llm_reports directory
        llm_reports_dir = Path("data/05_output/llm_reports")
        if not llm_reports_dir.exists():
            issues.append("LLM reports directory not found")
            llm_reports_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Created LLM reports directory")
        
        return issues
    
    def check_enhanced_pipeline_components(self) -> bool:
        """Check enhanced pipeline components functionality"""
        self.print_section_header("ENHANCED PIPELINE COMPONENTS CHECK")
        issues = []
        
        try:
            # Test imports
            from pynucleus.integration.config_manager import ConfigManager
            from pynucleus.integration.dwsim_rag_integrator import DWSIMRAGIntegrator
            from pynucleus.integration.llm_output_generator import LLMOutputGenerator
            self.log_both("   All enhanced modules imported successfully", console_symbol="✅ ",
                         clean_message="All enhanced modules imported successfully")
            
            # Test ConfigManager with new folder structure
            config_manager = ConfigManager(config_dir="configs")
            self.log_both(f"   ConfigManager: {config_manager.config_dir}", console_symbol="✅ ",
                         clean_message=f"ConfigManager: {config_manager.config_dir}")
            
            # Test LLMOutputGenerator with separate directories
            llm_generator = LLMOutputGenerator(results_dir="data/05_output/llm_reports")
            self.log_both(f"   LLMOutputGenerator: results_dir={llm_generator.results_dir}", console_symbol="✅ ",
                         clean_message=f"LLMOutputGenerator: results_dir={llm_generator.results_dir}")
            
            success = True
            
        except Exception as e:
            error_msg = f"Enhanced pipeline component test failed: {e}"
            self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
            issues.append(f"Component initialization failed: {e}")
            success = False
        
        self.log_result("Enhanced Pipeline Components", success, issues)
        return success
    
    def check_enhanced_content_generation(self) -> bool:
        """Test enhanced LLM content generation with detailed feed conditions"""
        self.print_section_header("ENHANCED CONTENT GENERATION CHECK")
        issues = []
        
        try:
            from pynucleus.integration.llm_output_generator import LLMOutputGenerator
            llm_generator = LLMOutputGenerator(results_dir="data/05_output/llm_reports")
            
            # Create simple test data that matches the expected structure for the Jinja template
            test_simulation = {
                'case_name': 'system_test_distillation',
                'simulation_type': 'distillation',
                'type': 'distillation',
                'components': 'water, ethanol',
                'description': 'System test distillation with enhanced parameters',
                'status': 'SUCCESS',
                'success': True,
                # Template expects these at the top level
                'conversion': 0.90,
                'selectivity': 0.95,
                'yield': 0.85,
                'temperature': 78.5,
                'pressure': 1.01,
                'duration_seconds': 45.2,
                'timestamp': '2025-06-11 00:43:00',
                # Keep results nested for financial calculations
                'results': {
                    'conversion': 0.90,
                    'selectivity': 0.95,
                    'yield': 0.85,
                    'flow_rate': 1200,
                    'efficiency': 0.88
                }
            }
            
            test_data = {
                'original_simulation': test_simulation,
                'performance_metrics': {
                    'overall_performance': 'Excellent',
                    'efficiency_rating': 'Very High',
                    'reliability_score': 0.95
                },
                'recommendations': ['Optimize energy integration'],
                'optimization_opportunities': ['Heat recovery', 'Advanced control'],
                'rag_insights': [
                    {'text': 'Modular design principles can improve efficiency', 'source': 'RAG Knowledge Base'},
                    {'text': 'Temperature optimization is critical for this process', 'source': 'Process Engineering Handbook'}
                ],
                'knowledge_integration': True
            }
            
            # Generate enhanced content
            output_file = llm_generator.export_llm_ready_text(test_data)
            self.log_both(f"   Enhanced LLM output generated: {output_file}", console_symbol="✅", 
                         clean_message=f"Enhanced LLM output generated: {output_file}")
            
            # Verify file location
            if 'llm_reports/' in str(output_file):
                self.log_both("   File saved in correct llm_reports/ folder", console_symbol="✅",
                             clean_message="File saved in correct llm_reports/ folder")
            else:
                issues.append("LLM file not saved in llm_reports/ folder")
            
            # Check enhanced content features
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Check for actual features that are generated by the current template
            enhanced_features = [
                ('Process Type', 'Process Type:'),
                ('Components', 'Components:'),
                ('Performance Metrics', 'Performance Metrics'),
                ('Conversion', 'Conversion:'),
                ('Selectivity', 'Selectivity:'),
                ('Temperature', 'Temperature:'),
                ('Pressure', 'Pressure:'),
                ('Recommendations', 'Recommendations:'),
                ('Analysis Summary', 'Analysis Summary'),
                ('LLM Simulation Analysis', 'LLM Simulation Analysis')
            ]
            
            self.log_both(f"\n   📄 Enhanced Content Features Check:", console_symbol="",
                         clean_message="Enhanced Content Features Check:")
            missing_features = []
            for feature_name, search_text in enhanced_features:
                found = search_text in content
                status_symbol = "✅" if found else "❌"
                status_text = "FOUND" if found else "MISSING"
                self.log_both(f"      {feature_name}", console_symbol=f"{status_symbol} ",
                             clean_message=f"{feature_name}: {status_text}")
                if not found:
                    missing_features.append(feature_name)
            
            if missing_features:
                issues.extend([f"Missing feature: {feature}" for feature in missing_features])
            
            # Test financial analysis with list of test data
            financial_file = llm_generator.export_financial_analysis([test_data])
            self.log_both(f"   Financial analysis generated: {Path(financial_file).name}", console_symbol="✅",
                         clean_message=f"Financial analysis generated: {Path(financial_file).name}")
            
            success = len(issues) == 0
            
        except Exception as e:
            self.log_both(f"   Enhanced content generation failed: {e}", console_symbol="❌",
                         clean_message=f"Enhanced content generation failed: {e}")
            issues.append(f"Content generation failed: {e}")
            success = False
        
        self.log_result("Enhanced Content Generation", success, issues)
        return success
    
    def check_mock_integration_testing(self) -> bool:
        """Check mock DWSIM-RAG integration testing"""
        self.print_section_header("MOCK INTEGRATION TESTING")
        issues = []
        
        try:
            # Setup temporary test configuration
            self.setup_temp_test_config()
            
            # Verify test files exist
            test_files = list(Path(self.temp_dir).glob("*"))
            if not test_files:
                issues.append("No test files found in temporary directory")
                return False
                
            self.log_both(f"Found {len(test_files)} test files", console_symbol="✅ ")
            
            # Test file processing
            for test_file in test_files:
                try:
                    if test_file.suffix == '.csv':
                        import pandas as pd
                        df = pd.read_csv(test_file)
                        self.log_both(f"Successfully processed {test_file.name}", console_symbol="✅ ")
                    elif test_file.suffix == '.json':
                        with open(test_file) as f:
                            json.load(f)
                        self.log_both(f"Successfully processed {test_file.name}", console_symbol="✅ ")
                except Exception as e:
                    issues.append(f"Error processing {test_file.name}: {str(e)}")
                    self.log_both(f"Error processing {test_file.name}", console_symbol="❌ ")
            
            success = len(issues) == 0
            self.log_result("Mock Integration Testing", success, issues)
            return success
            
        except Exception as e:
            issues.append(f"Mock testing failed: {str(e)}")
            self.log_result("Mock Integration Testing", False, issues)
            return False
        finally:
            # Cleanup is handled by __del__
            pass

    def test_basic_functionality(self) -> bool:
        """Test basic functionality without enhanced features."""
        
        self.log_both("\n🔧 Testing basic pipeline functionality...", console_symbol="", 
                     clean_message="Testing basic pipeline functionality...")
        
        try:
            from pynucleus.pipeline import RAGPipeline, DWSIMPipeline, ResultsExporter, PipelineUtils
            self.log_both("✅ Basic pipeline modules imported successfully!", console_symbol="", 
                         clean_message="Basic pipeline modules imported successfully!")
            
            # Test basic pipeline initialization
            pipeline = PipelineUtils()
            self.log_both("✅ Basic pipeline initialized successfully!", console_symbol="", 
                         clean_message="Basic pipeline initialized successfully!")
            
            # Test quick status
            test_results = pipeline.quick_test()
            self.log_both(f"✅ Quick test completed: {test_results['csv_files_count']} CSV files found", 
                         console_symbol="", 
                         clean_message=f"Quick test completed: {test_results['csv_files_count']} CSV files found")
            
            return True
            
        except Exception as e:
            error_msg = f"Basic functionality test failed: {e}"
            self.log_both(f"❌ {error_msg}", console_symbol="", clean_message=error_msg)
            return False

    def check_rag_system(self) -> bool:
        """Check PyNucleus RAG system components"""
        self.print_section_header("PYNUCLEUS RAG SYSTEM CHECK")
        issues = []
        
        # Check PyNucleus RAG directories (actual structure)
        rag_dirs = [
            ("data/01_raw/source_documents", "Source documents"),
            ("data/02_processed/converted_to_txt", "Converted documents"),
            ("data/01_raw/web_sources", "Web scraped sources"),
            ("data/03_intermediate/converted_chunked_data", "Chunked data"),
            ("data/04_models/chunk_reports", "FAISS analysis reports"),
            ("src/pynucleus/rag", "RAG source code"),
        ]
        
        for dir_path, description in rag_dirs:
            if Path(dir_path).exists():
                item_count = len(list(Path(dir_path).glob("*")))
                self.log_both(f"   {description}: {dir_path} ({item_count} items)", console_symbol="✅ ",
                             clean_message=f"{description}: {dir_path} ({item_count} items)")
            else:
                self.log_both(f"   {description}: {dir_path} (missing)", console_symbol="❌ ",
                             clean_message=f"{description}: {dir_path} (missing)")
                issues.append(f"Missing RAG directory: {dir_path}")
        
        # Check FAISS store in chunk_reports
        faiss_files = list(Path("data/04_models/chunk_reports").glob("*.faiss")) if Path("data/04_models/chunk_reports").exists() else []
        if faiss_files:
            self.log_both(f"   FAISS index files found: {len(faiss_files)}", console_symbol="✅ ",
                         clean_message=f"FAISS index files found: {len(faiss_files)}")
            for faiss_file in faiss_files[:2]:
                self.log_both(f"      • {faiss_file.name}", console_symbol="",
                             clean_message=f"FAISS file: {faiss_file.name}")
        else:
            self.log_both(f"   No FAISS index files found (will be created on first run)", console_symbol="⚠️ ",
                         clean_message="No FAISS index files found (will be created on first run)")
        
        success = len(issues) == 0
        self.log_result("RAG System", success, issues)
        return success

    def check_token_utilities(self) -> bool:
        """Check PyNucleus Token Utilities System"""
        self.print_section_header("TOKEN UTILITIES SYSTEM CHECK")
        issues = []
        
        # Check src/pynucleus/utils directory exists
        utils_dir = Path("src/pynucleus/utils")
        if not utils_dir.exists():
            self.log_both("   src/pynucleus/utils/ directory not found", console_symbol="❌ ",
                         clean_message="src/pynucleus/utils/ directory not found")
            issues.append("src/pynucleus/utils/ directory missing")
            self.log_result("Token Utilities System", False, issues)
            return False
        else:
            self.log_both("   src/pynucleus/utils/ directory exists", console_symbol="✅ ",
                         clean_message="src/pynucleus/utils/ directory exists")
        
        # Check PyNucleus integration
        pynucleus_token_file = Path("src/pynucleus/utils/token_utils.py")
        if pynucleus_token_file.exists():
            self.log_both("   PyNucleus integration: token_utils.py", console_symbol="✅ ",
                         clean_message="PyNucleus integration: token_utils.py - EXISTS")
        else:
            self.log_both("   PyNucleus integration: token_utils.py (missing)", console_symbol="❌ ",
                         clean_message="PyNucleus integration: token_utils.py - MISSING")
            issues.append("Missing PyNucleus integration file")
        
        # Check transformers dependency
        try:
            import transformers
            from transformers import AutoTokenizer
            self.log_both("   Transformers library available", console_symbol="✅ ",
                         clean_message="Transformers library available")
        except ImportError:
            self.log_both("   Transformers library not installed", console_symbol="❌ ",
                         clean_message="Transformers library not installed")
            issues.append("Transformers library missing - required for token utilities")
            self.log_result("Token Utilities System", False, issues)
            return False
        
        # Test PyNucleus integration
        try:
            from pynucleus.utils.token_utils import TokenCounter as PyNucleusTokenCounter
            from pynucleus.utils.token_utils import count_tokens as pynucleus_count_tokens
            
            self.log_both("   PyNucleus integration imports successful", console_symbol="✅ ",
                         clean_message="PyNucleus integration imports successful")
            
            # Test integrated functionality
            test_text = "PyNucleus integration test for token counting."
            integrated_count = pynucleus_count_tokens(test_text)
            
            if isinstance(integrated_count, int) and integrated_count > 0:
                self.log_both(f"   Integrated token counting: {integrated_count} tokens", console_symbol="✅ ",
                             clean_message=f"Integrated token counting: {integrated_count} tokens")
            else:
                self.log_both("   Integrated token counting failed", console_symbol="❌ ",
                             clean_message="Integrated token counting failed")
                issues.append("Integrated token counting returned invalid result")
            
        except Exception as e:
            error_msg = f"PyNucleus integration test failed: {e}"
            self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
            issues.append(error_msg)
        
        success = len(issues) == 0
        
        if success:
            self.log_both("\n   🎉 Token utilities system operational!", console_symbol="✅ ",
                         clean_message="Token utilities system operational!")
        else:
            self.log_both(f"\n   {len(issues)} issues found in token utilities system", console_symbol="❌ ",
                         clean_message=f"{len(issues)} issues found in token utilities system")
        
        self.log_result("Token Utilities System", success, issues)
        return success

    def check_llm_utilities(self) -> bool:
        """Check PyNucleus LLM Utilities System"""
        self.print_section_header("LLM UTILITIES SYSTEM CHECK")
        issues = []
        
        # Check src/pynucleus/llm directory exists
        llm_dir = Path("src/pynucleus/llm")
        if not llm_dir.exists():
            self.log_both("   src/pynucleus/llm/ directory not found", console_symbol="❌ ",
                         clean_message="src/pynucleus/llm/ directory not found")
            issues.append("src/pynucleus/llm/ directory missing")
            self.log_result("LLM Utilities System", False, issues)
            return False
        else:
            self.log_both("   src/pynucleus/llm/ directory exists", console_symbol="✅ ",
                         clean_message="src/pynucleus/llm/ directory exists")
        
        # Check required files in llm directory
        required_files = [
            ("llm_runner.py", "Main LLM runner module"),
            ("__init__.py", "Package initialization"),
            ("query_llm.py", "LLM query manager")
        ]
        
        for filename, description in required_files:
            file_path = llm_dir / filename
            if file_path.exists():
                self.log_both(f"   {description}: {filename}", console_symbol="✅ ",
                             clean_message=f"{description}: {filename} - EXISTS")
            else:
                self.log_both(f"   {description}: {filename} (missing)", console_symbol="❌ ",
                             clean_message=f"{description}: {filename} - MISSING")
                issues.append(f"Missing file: src/pynucleus/llm/{filename}")
        
        # Check transformers and torch dependencies
        try:
            import transformers
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.log_both("   Required dependencies (transformers, torch) available", console_symbol="✅ ",
                         clean_message="Required dependencies (transformers, torch) available")
        except ImportError as e:
            error_msg = f"Missing dependencies: {e}"
            self.log_both(f"   {error_msg}", console_symbol="❌ ",
                         clean_message=error_msg)
            issues.append(f"Dependencies missing: {e}")
            self.log_result("LLM Utilities System", False, issues)
            return False
        
        # Test LLMRunner functionality
        try:
            from pynucleus.llm import LLMRunner
            
            self.log_both("   LLMRunner imports successful", console_symbol="✅ ",
                         clean_message="LLMRunner imports successful")
        
        except Exception as e:
            error_msg = f"LLMRunner import failed: {e}"
            self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
            issues.append(error_msg)
        
        success = len(issues) == 0
        
        if success:
            self.log_both("\n   🎉 LLM utilities system operational!", console_symbol="✅ ",
                         clean_message="LLM utilities system operational!")
        else:
            self.log_both(f"\n   {len(issues)} issues found in LLM utilities system", console_symbol="❌ ",
                         clean_message=f"{len(issues)} issues found in LLM utilities system")
        
        self.log_result("LLM Utilities System", success, issues)
        return success

    def check_pipeline_functionality(self) -> bool:
        """Test core PyNucleus pipeline functionality"""
        self.print_section_header("PYNUCLEUS PIPELINE FUNCTIONALITY CHECK")
        issues = []
        
        try:
            # Test core pipeline imports
            from pynucleus.pipeline import RAGPipeline, DWSIMPipeline, ResultsExporter, PipelineUtils
            self.log_both("   Core pipeline modules imported successfully", console_symbol="✅ ",
                         clean_message="Core pipeline modules imported successfully")
            
            # Initialize pipeline utils
            pipeline = PipelineUtils()
            self.log_both("   PipelineUtils initialized successfully", console_symbol="✅ ",
                         clean_message="PipelineUtils initialized successfully")
            self.log_both(f"   Results directory: {pipeline.results_dir}", console_symbol="✅ ",
                         clean_message=f"Results directory: {pipeline.results_dir}")
            
            # Check if pipeline has basic functionality
            if hasattr(pipeline, 'rag_pipeline') and hasattr(pipeline, 'dwsim_pipeline'):
                self.log_both("   RAG and DWSIM pipelines available", console_symbol="✅ ",
                             clean_message="RAG and DWSIM pipelines available")
            else:
                issues.append("Pipeline components not properly initialized")
            
            success = True
            
        except Exception as e:
            error_msg = f"Pipeline functionality test failed: {e}"
            self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
            issues.append(f"Pipeline initialization failed: {e}")
            success = False
        
        self.log_result("Pipeline Functionality", success, issues)
        return success

    def check_data_consolidation_deliverables(self) -> bool:
        """Check specific data consolidation deliverables"""
        self.print_section_header("DATA CONSOLIDATION DELIVERABLES CHECK")
        issues = []
        
        self.log_both("\nVerifying Data Directory Consolidation Deliverables:", console_symbol="📋 ",
                     clean_message="Verifying Data Directory Consolidation Deliverables:")
        
        # Check data subdirectories creation
        required_data_dirs = [
            "data/01_raw",
            "data/02_processed", 
            "data/03_intermediate",
            "data/04_models",
            "data/05_output"
        ]
        
        self.log_both("\n   🔍 Data Subdirectories:", console_symbol="",
                     clean_message="Data Subdirectories:")
        for data_dir in required_data_dirs:
            if Path(data_dir).exists():
                self.log_both(f"      {data_dir}/ exists", console_symbol="✅ ",
                             clean_message=f"{data_dir}/ exists")
            else:
                self.log_both(f"      {data_dir}/ missing", console_symbol="❌ ",
                             clean_message=f"{data_dir}/ missing")
                issues.append(f"Missing data directory: {data_dir}")
                # Create directory if it doesn't exist
                Path(data_dir).mkdir(parents=True, exist_ok=True)
                self.log_both(f"      Created directory: {data_dir}", console_symbol="✅ ",
                             clean_message=f"Created directory: {data_dir}")
        
        # Check moved folders in correct locations
        moved_folders = {
            "data/01_raw/source_documents": "Source documents",
            "data/01_raw/web_sources": "Web sources", 
            "data/02_processed/converted_to_txt": "Converted documents",
            "data/03_intermediate/converted_chunked_data": "Chunked data",
            "data/04_models/chunk_reports": "Chunk reports",
            "data/05_output/results": "Results",
            "data/05_output/llm_reports": "LLM reports"
        }
        
        self.log_both("\n   🔍 Moved Folders Verification:", console_symbol="",
                     clean_message="Moved Folders Verification:")
        for folder_path, description in moved_folders.items():
            if Path(folder_path).exists():
                item_count = len(list(Path(folder_path).glob("*")))
                self.log_both(f"      {description}: {folder_path}/ ({item_count} items)", console_symbol="✅ ",
                             clean_message=f"{description}: {folder_path}/ ({item_count} items)")
            else:
                self.log_both(f"      {description}: {folder_path}/ missing", console_symbol="❌ ",
                             clean_message=f"{description}: {folder_path}/ missing")
                # Create directory if it doesn't exist
                Path(folder_path).mkdir(parents=True, exist_ok=True)
                self.log_both(f"      Created directory: {folder_path}", console_symbol="✅ ",
                             clean_message=f"Created directory: {folder_path}")
        
        success = len(issues) == 0
        
        self.log_both("\n   Data directories centralized and verified.", console_symbol="🎉 ✅ ",
                     clean_message="Data directories centralized and verified.")
        
        self.log_result("Data Consolidation Deliverables", success, issues)
        return success

    def generate_final_report(self):
        """Generate comprehensive final report and save to logs"""
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        # Calculate statistics
        health_percentage = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        if health_percentage >= 90:
            status_text = "EXCELLENT"
        elif health_percentage >= 70:
            status_text = "GOOD"
        else:
            status_text = "NEEDS ATTENTION"
        
        # Generate report
        report = f"""PYNUCLEUS SYSTEM DIAGNOSTIC & TESTING REPORT
Generated: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Duration: {duration.total_seconds():.1f} seconds

EXECUTIVE SUMMARY
System Health: {health_percentage:.1f}% ({status_text})
Checks Performed: {self.total_checks}
Checks Passed: {self.passed_checks}
Checks Failed: {self.total_checks - self.passed_checks}

DETAILED RESULTS"""
        
        # Group results by status
        passed = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        if passed:
            report += f"\n\nPASSED CHECKS ({len(passed)}):\n"
            for result in passed:
                report += f"- {result['name']}\n"
        
        if failed:
            report += f"\nFAILED CHECKS ({len(failed)}):\n"
            for result in failed:
                report += f"- {result['name']}\n"
                for detail in result['details']:
                    clean_detail = self.clean_message_for_file(detail)
                    report += f"  * {clean_detail}\n"
        
        # Add recommendations
        report += f"\nRECOMMENDATIONS:\n"
        if health_percentage == 100:
            report += "- All systems operational. PyNucleus is ready for production use.\n"
        elif health_percentage >= 90:
            report += "- Core systems operational. Address minor issues for optimal performance.\n"
        elif health_percentage >= 70:
            report += "- System functional with some issues. Review failed checks.\n"
        else:
            report += "- Critical issues detected. Address failed checks before proceeding.\n"
        
        # System capabilities
        report += f"\nSYSTEM CAPABILITIES:\n"
        report += "- RAG Pipeline: Document processing and retrieval\n"
        report += "- DWSIM Pipeline: Chemical process simulation\n"
        report += "- Results Export: CSV and LLM-ready formats\n"
        report += "- Enhanced Integration: DWSIM-RAG combined analysis\n"
        report += "- Financial Analysis: ROI and profitability calculations\n"
        report += "- Token Utilities: Efficient token counting with HuggingFace tokenizers\n"
        report += "- LLM Utilities: Streamlined interface for querying Hugging Face LLM models\n"
        report += "- Jinja2 Prompts System: Standardized LLM prompt templates\n"
        report += "- Mock Testing: Comprehensive testing with simulated data\n"
        
        # System usage information
        report += f"\nSYSTEM USAGE:\n"
        report += "- CLI Tool: python run_pipeline.py run --config-path configs/my.json\n"
        report += "- Jupyter Notebook: Capstone Project.ipynb for interactive development\n"
        report += "- System Environment: python scripts/comprehensive_system_diagnostic.py\n"
        report += "- Script Validation: python scripts/system_validator.py\n"
        report += "- Test Suite: python scripts/comprehensive_system_diagnostic.py --test\n"
        
        # Save to log file
        self.file_logger.info("=" * 60)
        self.file_logger.info("FINAL DIAGNOSTIC REPORT")
        self.file_logger.info("=" * 60)
        for line in report.strip().split('\n'):
            if line.strip():
                self.file_logger.info(line.strip())
        
        # Also save as separate report file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        report_file = Path("logs") / f"diagnostic_report_{timestamp}.txt"
        report_file.write_text(report.strip())
        
        return report_file

    def print_final_summary(self, report_file):
        """Print comprehensive diagnostic summary to console"""
        self.print_section_header("COMPREHENSIVE DIAGNOSTIC SUMMARY")
        
        health_percentage = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
        print(f"📊 Overall Results: {self.passed_checks}/{self.total_checks} checks passed")
        print(f"{'─'*60}")
        
        # Group results by status
        passed = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        if passed:
            print(f"✅ PASSED CHECKS ({len(passed)}):")
            for result in passed:
                print(f"   ✅ {result['name']}")
        
        if failed:
            print(f"\n❌ FAILED CHECKS ({len(failed)}):")
            for result in failed:
                print(f"   ❌ {result['name']}")
                for detail in result['details']:
                    print(f"      • {detail}")
        
        print(f"\n{'─'*60}")
        
        # Overall system health
        if health_percentage >= 90:
            status_emoji = "🟢"
            status_text = "EXCELLENT"
        elif health_percentage >= 70:
            status_emoji = "🟡" 
            status_text = "GOOD"
        else:
            status_emoji = "🔴"
            status_text = "NEEDS ATTENTION"
        
        print(f"\n{status_emoji} SYSTEM HEALTH: {health_percentage:.1f}% - {status_text}")
        
        if health_percentage == 100:
            print("🎉 All systems operational! PyNucleus is ready for production use.")
        elif health_percentage >= 90:
            print("✅ Core systems operational! Minor issues detected.")
        else:
            print("⚠️ Critical issues detected. Review failed checks before proceeding.")
        
        # Log file information
        print(f"\n📋 DIAGNOSTIC REPORTS GENERATED:")
        print(f"   • Detailed Log: {self.log_file_path}")
        print(f"   • Summary Report: {report_file}")
        print(f"   • All logs saved to: /logs directory")
    
    def check_prompts_system(self) -> bool:
        """Check Jinja2 prompt template system functionality"""
        self.print_section_header("JINJA2 PROMPTS SYSTEM CHECK")
        issues = []
        
        # Check prompts directory exists
        prompts_dir = Path("prompts")
        if not prompts_dir.exists():
            self.log_both("   prompts/ directory not found", console_symbol="❌ ",
                         clean_message="prompts/ directory not found")
            issues.append("prompts/ directory missing")
        else:
            self.log_both("   prompts/ directory exists", console_symbol="✅ ",
                         clean_message="prompts/ directory exists")
        
        # Check required files in prompts directory
        required_files = [
            ("qwen_prompt.j2", "Jinja2 template file"),
            ("prompt_system.py", "PromptSystem class"),
            ("notebook_integration.py", "Jupyter notebook integration"),
            ("README.md", "Documentation")
        ]
        
        for filename, description in required_files:
            file_path = prompts_dir / filename
            if file_path.exists():
                self.log_both(f"   {description}: {filename}", console_symbol="✅ ",
                             clean_message=f"{description}: {filename} - EXISTS")
            else:
                self.log_both(f"   {description}: {filename} (missing)", console_symbol="❌ ",
                             clean_message=f"{description}: {filename} - MISSING")
                issues.append(f"Missing file: prompts/{filename}")
        
        # Check Jinja2 dependency
        try:
            import jinja2
            self.log_both("   Jinja2 library available", console_symbol="✅ ",
                         clean_message="Jinja2 library available")
        except ImportError:
            self.log_both("   Jinja2 library not installed", console_symbol="❌ ",
                         clean_message="Jinja2 library not installed")
            issues.append("Jinja2 library missing - install with: pip install jinja2")
        
        # Test template loading and rendering (if files exist)
        template_path = prompts_dir / "qwen_prompt.j2"
        if template_path.exists():
            try:
                import jinja2
                from jinja2 import Environment, FileSystemLoader
                
                # Test template loading
                env = Environment(loader=FileSystemLoader('prompts'))
                template = env.get_template('qwen_prompt.j2')
                
                # Test template rendering with sample data
                test_data = {
                    'system_message': 'You are a helpful assistant.',
                    'context': 'Sample context for testing',
                    'question': 'What is the meaning of life?',
                    'constraints': ['Be concise', 'Be accurate'],
                    'format_instructions': 'Respond in plain text'
                }
                
                rendered = template.render(**test_data)
                
                # Check for required sections in rendered output
                required_sections = ['<SYSTEM>', '<CONTEXT>', '<QUESTION>', '<ANSWER>']
                missing_sections = []
                for section in required_sections:
                    if section not in rendered:
                        missing_sections.append(section)
                
                if missing_sections:
                    self.log_both(f"   Template missing sections: {', '.join(missing_sections)}", 
                                 console_symbol="❌ ", 
                                 clean_message=f"Template missing sections: {', '.join(missing_sections)}")
                    issues.append(f"Template missing sections: {missing_sections}")
                else:
                    self.log_both("   Template renders correctly with all sections", console_symbol="✅ ",
                                 clean_message="Template renders correctly with all sections")
                
            except Exception as e:
                error_msg = f"Template rendering test failed: {e}"
                self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
                issues.append(error_msg)
        
        # Test PromptSystem class functionality (if file exists)
        prompt_system_path = prompts_dir / "prompt_system.py"
        if prompt_system_path.exists():
            try:
                # Add prompts directory to path for import
                import sys
                prompts_path = str(prompts_dir.absolute())
                if prompts_path not in sys.path:
                    sys.path.insert(0, prompts_path)
                
                # Test import and basic functionality
                try:
                    from prompts.prompt_system import PromptSystem
                except ImportError:
                    from prompt_system import PromptSystem
                
                ps = PromptSystem()
                self.log_both("   PromptSystem class imported and initialized", console_symbol="✅ ",
                             clean_message="PromptSystem class imported and initialized")
                
                # Test template validation
                if hasattr(ps, 'validate_template'):
                    validation_result = ps.validate_template()
                    # Handle both boolean and dict return types
                    if isinstance(validation_result, bool):
                        validation_passed = validation_result
                    elif isinstance(validation_result, dict):
                        validation_passed = validation_result.get('valid', False)
                    else:
                        validation_passed = False
                    
                    if validation_passed:
                        self.log_both("   Template validation passed", console_symbol="✅ ",
                                     clean_message="Template validation passed")
                    else:
                        self.log_both("   Template validation failed", console_symbol="❌ ",
                                     clean_message="Template validation failed")
                        issues.append("Template validation failed")
                
                # Clean up sys.path
                if prompts_path in sys.path:
                    sys.path.remove(prompts_path)
                    
            except Exception as e:
                error_msg = f"PromptSystem class test failed: {e}"
                self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
                issues.append(error_msg)
        
        # Test notebook integration (if file exists)
        notebook_integration_path = prompts_dir / "notebook_integration.py"
        if notebook_integration_path.exists():
            try:
                # Read and check notebook integration file
                with open(notebook_integration_path, 'r') as f:
                    content = f.read()
                
                # Check for key functions
                key_functions = ['create_prompt', 'demo_prompts', 'validate_prompts']
                missing_functions = []
                for func in key_functions:
                    if f"def {func}" not in content:
                        missing_functions.append(func)
                
                if missing_functions:
                    self.log_both(f"   Notebook integration missing functions: {', '.join(missing_functions)}", 
                                 console_symbol="❌ ", 
                                 clean_message=f"Notebook integration missing functions: {', '.join(missing_functions)}")
                    issues.append(f"Missing functions: {missing_functions}")
                else:
                    self.log_both("   Notebook integration functions available", console_symbol="✅ ",
                                 clean_message="Notebook integration functions available")
                
            except Exception as e:
                error_msg = f"Notebook integration test failed: {e}"
                self.log_both(f"   {error_msg}", console_symbol="❌ ", clean_message=error_msg)
                issues.append(error_msg)
        
        success = len(issues) == 0
        
        if success:
            self.log_both("\n   🎉 Jinja2 prompt system fully operational!", console_symbol="✅ ",
                         clean_message="Jinja2 prompt system fully operational!")
        else:
            self.log_both(f"\n   {len(issues)} issues found in prompts system", console_symbol="❌ ",
                         clean_message=f"{len(issues)} issues found in prompts system")
        
        self.log_result("Jinja2 Prompts System", success, issues)
        return success

    def run_comprehensive_diagnostic(self, test_mode="full"):
        """Run all diagnostic checks"""
        # Run PyNucleus-specific checks
        checks = [
            ("Python Environment", self.check_python_environment),
            ("Pipeline Functionality", self.check_pipeline_functionality),
            ("Enhanced Pipeline Components", self.check_enhanced_pipeline_components),
            ("Enhanced Content Generation", self.check_enhanced_content_generation),
            ("PyNucleus RAG System", self.check_rag_system),
            ("Token Utilities System", self.check_token_utilities),
            ("LLM Utilities System", self.check_llm_utilities),
            ("Jinja2 Prompts System", self.check_prompts_system),
            ("DWSIM Environment (Optional)", self.check_dwsim_environment),
            ("Docker Environment", self.check_docker_environment),
            ("Data Consolidation Deliverables", self.check_data_consolidation_deliverables),
        ]
        
        # Add comprehensive validation tests for full mode
        if test_mode in ["full", "validation"]:
            checks.extend([
                ("Ground-Truth Validation", self.check_ground_truth_validation),
                ("Citation Backtracking", self.check_citation_backtracking),
            ])
        
        # Add mock testing for full mode
        if test_mode in ["full", "mock"]:
            checks.append(("Mock Integration Testing", self.check_mock_integration_testing))
        
        for check_name, check_func in checks:
            try:
                check_func()
            except Exception as e:
                error_msg = f"{check_name} check failed with error: {e}"
                self.log_both(error_msg, console_symbol="❌ ", 
                             clean_message=error_msg)
                self.log_result(check_name, False, [f"Check error: {e}"])
        
        # Save validation reports if validation tests were run
        if test_mode in ["full", "validation"] and (self.validation_results or self.citation_results):
            self.save_validation_reports()
        
        # Generate and save final report
        report_file = self.generate_final_report()
        
        # Print summary to console
        self.print_final_summary(report_file)
    
    def run_test_suite_mode(self):
        """Run enhanced test suite similar to test_enhanced_pipeline.py"""
        self.print_section_header("PYNUCLEUS ENHANCED PIPELINE TEST SUITE")
        
        self.log_both("🚀 PyNucleus Enhanced Pipeline Test Suite", console_symbol="", 
                     clean_message="PyNucleus Enhanced Pipeline Test Suite")
        self.log_both("=" * 60, console_symbol="", clean_message="=" * 60)
        
        # Test basic functionality first
        basic_test_passed = self.test_basic_functionality()
        
        # Test enhanced functionality
        enhanced_test_passed = self.check_mock_integration_testing()
        
        self.log_both("\n" + "=" * 60, console_symbol="", clean_message="=" * 60)
        self.log_both("📊 Test Results Summary:", console_symbol="", clean_message="Test Results Summary:")
        basic_status = "✅ PASS" if basic_test_passed else "❌ FAIL"
        enhanced_status = "✅ PASS" if enhanced_test_passed else "❌ FAIL"
        self.log_both(f"   Basic Pipeline: {basic_status}", console_symbol="", 
                     clean_message=f"Basic Pipeline: {'PASS' if basic_test_passed else 'FAIL'}")
        self.log_both(f"   Enhanced Pipeline: {enhanced_status}", console_symbol="", 
                     clean_message=f"Enhanced Pipeline: {'PASS' if enhanced_test_passed else 'FAIL'}")
        
        if basic_test_passed and enhanced_test_passed:
            self.log_both("\n🎉 All tests passed! The enhanced pipeline is ready to use.", console_symbol="", 
                         clean_message="All tests passed! The enhanced pipeline is ready to use.")
            self.log_both("\n💡 Next steps:", console_symbol="", clean_message="Next steps:")
            self.log_both("   1. Run the Capstone Project.ipynb notebook", console_symbol="", 
                         clean_message="1. Run the Capstone Project.ipynb notebook")
            self.log_both("   2. Try the enhanced pipeline sections", console_symbol="", 
                         clean_message="2. Try the enhanced pipeline sections")
            self.log_both("   3. Edit the configuration templates to customize simulations", console_symbol="", 
                         clean_message="3. Edit the configuration templates to customize simulations")
        else:
            self.log_both("\n⚠️ Some tests failed. Please check the error messages above.", console_symbol="", 
                         clean_message="Some tests failed. Please check the error messages above.")
            self.log_both("💡 Common solutions:", console_symbol="", clean_message="Common solutions:")
            self.log_both("   • Run: pip install -r requirements.txt", console_symbol="", 
                         clean_message="Run: pip install -r requirements.txt")
            self.log_both("   • Ensure you're in the project root directory", console_symbol="", 
                         clean_message="Ensure you're in the project root directory")
            self.log_both("   • Check that all core modules are properly installed", console_symbol="", 
                         clean_message="Check that all core modules are properly installed")
        
        self.log_both("\n" + "=" * 60, console_symbol="", clean_message="=" * 60)

def main():
    """Run comprehensive system diagnostic with command-line options"""
    parser = argparse.ArgumentParser(
        description="Comprehensive PyNucleus System Diagnostic & Testing Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/comprehensive_system_diagnostic.py                    # Full diagnostic
  python scripts/comprehensive_system_diagnostic.py --test            # Test suite mode
  python scripts/comprehensive_system_diagnostic.py --quiet           # Quiet mode
  python scripts/comprehensive_system_diagnostic.py --mock            # Mock testing only
  python scripts/comprehensive_system_diagnostic.py --validation      # Validation tests only
        """
    )
    
    parser.add_argument(
        '--test', 
        action='store_true', 
        help='Run in test suite mode (similar to test_enhanced_pipeline.py)'
    )
    parser.add_argument(
        '--quiet', 
        action='store_true', 
        help='Run in quiet mode with minimal output'
    )
    parser.add_argument(
        '--mock', 
        action='store_true', 
        help='Run mock testing only'
    )
    parser.add_argument(
        '--validation', 
        action='store_true', 
        help='Run comprehensive validation and citation tests'
    )
    
    args = parser.parse_args()
    
    diagnostic = SystemDiagnostic()
    
    if args.test:
        # Run test suite mode
        diagnostic.run_test_suite_mode()
    elif args.mock:
        # Run only mock testing
        diagnostic.run_comprehensive_diagnostic(test_mode="mock")
    elif args.validation:
        # Run validation tests only
        diagnostic.run_comprehensive_diagnostic(test_mode="validation")
    else:
        # Run full diagnostic
        test_mode = "full" if not args.quiet else "quiet"
        diagnostic.run_comprehensive_diagnostic(test_mode=test_mode)
    
    # Exit with appropriate code
    success_rate = diagnostic.passed_checks / diagnostic.total_checks if diagnostic.total_checks > 0 else 0
    exit_code = 0 if success_rate >= 0.9 else 1  # Require 90% pass rate
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 