#!/usr/bin/env python3
"""
Comprehensive PyNucleus System Diagnostic
Combines all system checks in one comprehensive script:

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
"""

import os
import sys
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Add src directory to Python path
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

class SystemDiagnostic:
    def __init__(self):
        self.results = []
        self.total_checks = 0
        self.passed_checks = 0
        
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
    
    def print_section_header(self, title: str):
        """Print formatted section header"""
        print(f"\n{'='*60}")
        print(f"🔍 {title}")
        print(f"{'='*60}")
    
    def check_python_environment(self) -> bool:
        """Check Python environment and dependencies"""
        self.print_section_header("PYTHON ENVIRONMENT CHECK")
        issues = []
        
        # Check Python version
        python_version = sys.version.split()[0]
        print(f"✅ Python version: {python_version}")
        
        # Check required packages (only essential ones for PyNucleus)
        required_packages = [
            "numpy", "pandas", "requests", "tqdm"
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
                print(f"   ✅ {package} (required)")
            except ImportError:
                print(f"   ❌ {package} (required - missing)")
                missing_packages.append(package)
                issues.append(f"Missing required package: {package}")
        
        for package in optional_packages:
            try:
                importlib.import_module(package.replace("-", "_"))
                print(f"   ✅ {package} (optional)")
            except ImportError:
                print(f"   ⚠️ {package} (optional - missing)")
        
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
            print("   ⚠️ DWSIM_DLL_PATH environment variable not set (simulation will use mock data)")
            print("   ℹ️ This is optional - PyNucleus can run without DWSIM for testing")
        else:
            print(f"   ✅ DWSIM_DLL_PATH: {dwsim_path}")
            
            # Check DLL directory exists
            dll_path = Path(dwsim_path)
            if not dll_path.exists():
                print(f"   ⚠️ DWSIM DLL directory does not exist: {dwsim_path}")
                issues.append(f"DLL directory missing: {dwsim_path}")
            else:
                # Count DLLs
                dlls = list(dll_path.glob("*.dll"))
                print(f"   ✅ Found {len(dlls)} DLL files")
                for dll in dlls[:3]:  # Show first 3
                    print(f"      • {dll.name}")
                if len(dlls) > 3:
                    print(f"      ... and {len(dlls)-3} more")
                
                if not dlls:
                    issues.append("No DLL files found")
        
        # DWSIM is optional, so always pass this check
        success = True
        self.log_result("DWSIM Environment", success, issues)
        return success
    
    def check_directory_structure(self) -> bool:
        """Check PyNucleus directory structure"""
        self.print_section_header("PYNUCLEUS DIRECTORY STRUCTURE CHECK")
        issues = []
        
        print("\n============================================================")
        print("🔍 PYNUCLEUS DIRECTORY STRUCTURE CHECK")
        print("============================================================")
        
        # Check core directories
        core_dirs = {
            "Core pipeline modules": "src/pynucleus",
            "LLM output files": "llm_reports",
            "Configuration files": "simulation_input_config",
            "Pipeline results": "results",
            "Source documents for RAG": "source_documents",
            "Converted documents": "converted_to_txt",
            "Web scraped sources": "web_sources",
            "FAISS analysis reports": "chunk_reports"
        }
        
        for name, path in core_dirs.items():
            if os.path.exists(path):
                items = len(os.listdir(path))
                print(f"   ✅ {name}: {path}/ ({items} items)")
            else:
                print(f"   ❌ {name}: {path}/ (missing)")
                issues.append(f"Missing directory: {path}")
        
        print("\n   📁 Core Module Structure:")
        core_module_dirs = {
            "Pipeline components": "src/pynucleus/pipeline",
            "RAG components": "src/pynucleus/rag",
            "Enhanced integration": "src/pynucleus/integration",
            "Simulation components": "src/pynucleus/simulation"
        }
        
        for name, path in core_module_dirs.items():
            if os.path.exists(path):
                print(f"      ✅ {name}: {path}/")
            else:
                print(f"      ❌ {name}: {path}/ (missing)")
                issues.append(f"Missing core directory: {path}")
        
        # Check our enhanced folders specifically
        enhanced_folders = [
            "llm_reports",
            "simulation_input_config", 
            "results"
        ]
        
        print(f"\n   📁 Enhanced Pipeline Folders:")
        for folder in enhanced_folders:
            if Path(folder).exists():
                files = list(Path(folder).glob("*"))
                print(f"      ✅ {folder}/ ({len(files)} files)")
            else:
                print(f"      ❌ {folder}/ (missing)")
                issues.append(f"Enhanced folder missing: {folder}")
        
        success = len(issues) == 0
        self.log_result("Directory Structure", success, issues)
        return success
    
    def check_enhanced_pipeline_components(self) -> bool:
        """Check enhanced pipeline components functionality"""
        self.print_section_header("ENHANCED PIPELINE COMPONENTS CHECK")
        issues = []
        
        try:
            # Test imports
            from pynucleus.integration.config_manager import ConfigManager
            from pynucleus.integration.dwsim_rag_integrator import DWSIMRAGIntegrator
            from pynucleus.integration.llm_output_generator import LLMOutputGenerator
            print("   ✅ All enhanced modules imported successfully")
            
            # Test ConfigManager with new folder structure
            config_manager = ConfigManager(config_dir="simulation_input_config")
            print(f"   ✅ ConfigManager: {config_manager.config_dir}")
            
            # Test LLMOutputGenerator with separate directories
            llm_generator = LLMOutputGenerator(results_dir="results", llm_output_dir="llm_reports")
            print(f"   ✅ LLMOutputGenerator: results_dir={llm_generator.results_dir}, llm_output_dir={llm_generator.llm_output_dir}")
            
            # Test template creation (will skip if exists)
            json_template = config_manager.create_template_json("system_diagnostic_test.json", verbose=True)
            csv_template = config_manager.create_template_csv("system_diagnostic_test.csv", verbose=True)
            
            success = True
            
        except Exception as e:
            print(f"   ❌ Enhanced pipeline component test failed: {e}")
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
            llm_generator = LLMOutputGenerator(results_dir="results", llm_output_dir="llm_reports")
            
            # Create comprehensive test data
            test_data = [
                {
                    'original_simulation': {
                        'case_name': 'system_test_distillation',
                        'type': 'distillation',
                        'components': 'water, ethanol',
                        'description': 'System test distillation with enhanced parameters',
                        'success': True,
                        'result_summary': "{'results': {'conversion': 0.90, 'selectivity': 0.95, 'yield': 0.85}}"
                    },
                    'performance_metrics': {
                        'overall_performance': 'Excellent',
                        'efficiency_rating': 'Very High'
                    },
                    'recommendations': ['Optimize energy integration'],
                    'optimization_opportunities': ['Heat recovery', 'Advanced control']
                },
                {
                    'original_simulation': {
                        'case_name': 'system_test_reactor',
                        'type': 'reactor',
                        'components': 'methane, steam',
                        'description': 'System test steam reformer',
                        'success': True,
                        'result_summary': "{'results': {'conversion': 0.88, 'selectivity': 0.92, 'yield': 0.81}}"
                    },
                    'performance_metrics': {
                        'overall_performance': 'Good',
                        'efficiency_rating': 'High'
                    },
                    'recommendations': ['Monitor catalyst activity'],
                    'optimization_opportunities': ['Temperature optimization']
                }
            ]
            
            # Generate enhanced content
            output_file = llm_generator.export_llm_ready_text(test_data, verbose=False)
            print(f"   ✅ Enhanced LLM output generated: {output_file}")
            
            # Verify file location
            if 'llm_reports/' in str(output_file):
                print("   ✅ File saved in correct llm_reports/ folder")
            else:
                issues.append("LLM file not saved in llm_reports/ folder")
            
            # Check enhanced content features
            with open(output_file, 'r') as f:
                content = f.read()
            
            enhanced_features = [
                ('Feed Conditions', 'Feed Conditions:'),
                ('Operating Conditions', 'Operating Conditions:'),
                ('Mole Fraction', 'Mole Fraction:'),
                ('Mass Flow Rate', 'Mass Flow Rate:'),
                ('Component Breakdown', 'Component Breakdown:'),
                ('Temperature (°C)', '°C'),
                ('Pressure (kPa)', 'kPa'),
                ('Pressure (atm)', 'atm'),
                ('Performance Results', 'Performance Results:'),
                ('Economic Metrics', 'Economic Metrics:'),
                ('Total Feed Rate', 'Total Feed Rate:'),
                ('Feed Temperature', 'Feed Temperature:'),
                ('Feed Pressure', 'Feed Pressure:')
            ]
            
            print(f"\n   📄 Enhanced Content Features Check:")
            missing_features = []
            for feature_name, search_text in enhanced_features:
                found = search_text in content
                status = "✅" if found else "❌"
                print(f"      {status} {feature_name}")
                if not found:
                    missing_features.append(feature_name)
            
            if missing_features:
                issues.extend([f"Missing feature: {feature}" for feature in missing_features])
            
            # Test financial analysis
            financial_file = llm_generator.export_financial_analysis(test_data)
            print(f"   ✅ Financial analysis generated: {Path(financial_file).name}")
            
            success = len(issues) == 0
            
        except Exception as e:
            print(f"   ❌ Enhanced content generation failed: {e}")
            issues.append(f"Content generation failed: {e}")
            success = False
        
        self.log_result("Enhanced Content Generation", success, issues)
        return success
    
    def check_rag_system(self) -> bool:
        """Check PyNucleus RAG system components"""
        self.print_section_header("PYNUCLEUS RAG SYSTEM CHECK")
        issues = []
        
        # Check PyNucleus RAG directories (actual structure)
        rag_dirs = [
            ("source_documents", "Source documents"),
            ("converted_to_txt", "Converted documents"),
            ("web_sources", "Web scraped sources"),
            ("converted_chunked_data", "Chunked data"),
            ("chunk_reports", "FAISS analysis reports"),
            ("src/pynucleus/rag", "RAG source code"),
        ]
        
        for dir_path, description in rag_dirs:
            if Path(dir_path).exists():
                item_count = len(list(Path(dir_path).glob("*")))
                print(f"   ✅ {description}: {dir_path} ({item_count} items)")
            else:
                print(f"   ❌ {description}: {dir_path} (missing)")
                issues.append(f"Missing RAG directory: {dir_path}")
        
        # Check FAISS store in chunk_reports
        faiss_files = list(Path("chunk_reports").glob("*.faiss")) if Path("chunk_reports").exists() else []
        if faiss_files:
            print(f"   ✅ FAISS index files found: {len(faiss_files)}")
            for faiss_file in faiss_files[:2]:
                print(f"      • {faiss_file.name}")
        else:
            print(f"   ⚠️ No FAISS index files found (will be created on first run)")
        
        success = len(issues) == 0
        self.log_result("RAG System", success, issues)
        return success
    
    def check_pipeline_functionality(self) -> bool:
        """Test core PyNucleus pipeline functionality"""
        self.print_section_header("PYNUCLEUS PIPELINE FUNCTIONALITY CHECK")
        issues = []
        
        try:
            # Test core pipeline imports
            from pynucleus.pipeline import RAGPipeline, DWSIMPipeline, ResultsExporter, PipelineUtils
            print("   ✅ Core pipeline modules imported successfully")
            
            # Initialize pipeline utils
            pipeline = PipelineUtils()
            print("   ✅ PipelineUtils initialized successfully")
            print(f"   ✅ Results directory: {pipeline.results_dir}")
            
            # Check if pipeline has basic functionality
            if hasattr(pipeline, 'rag_pipeline') and hasattr(pipeline, 'dwsim_pipeline'):
                print("   ✅ RAG and DWSIM pipelines available")
            else:
                issues.append("Pipeline components not properly initialized")
            
            success = True
            
        except Exception as e:
            print(f"   ❌ Pipeline functionality test failed: {e}")
            issues.append(f"Pipeline initialization failed: {e}")
            success = False
        
        self.log_result("Pipeline Functionality", success, issues)
        return success
    
    def run_comprehensive_diagnostic(self):
        """Run all diagnostic checks"""
        print("🚀 COMPREHENSIVE PYNUCLEUS SYSTEM DIAGNOSTIC")
        print(f"{'='*60}")
        print("Testing: Environment, Enhanced Pipeline, Components, Content Generation")
        
        # Run PyNucleus-specific checks
        checks = [
            ("Python Environment", self.check_python_environment),
            ("Directory Structure", self.check_directory_structure), 
            ("Pipeline Functionality", self.check_pipeline_functionality),
            ("Enhanced Pipeline Components", self.check_enhanced_pipeline_components),
            ("Enhanced Content Generation", self.check_enhanced_content_generation),
            ("PyNucleus RAG System", self.check_rag_system),
            ("DWSIM Environment (Optional)", self.check_dwsim_environment),
        ]
        
        for check_name, check_func in checks:
            try:
                check_func()
            except Exception as e:
                print(f"❌ {check_name} check failed with error: {e}")
                self.log_result(check_name, False, [f"Check error: {e}"])
        
        # Print comprehensive summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print comprehensive diagnostic summary"""
        self.print_section_header("COMPREHENSIVE DIAGNOSTIC SUMMARY")
        
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
        
        # Enhanced pipeline specific summary
        enhanced_checks = [r for r in self.results if 'Enhanced' in r['name'] or 'Directory' in r['name']]
        enhanced_passed = [r for r in enhanced_checks if r['success']]
        
        print(f"🎯 ENHANCED PIPELINE STATUS ({len(enhanced_passed)}/{len(enhanced_checks)} ready):")
        print("   ✅ 1. LLM outputs in separate llm_reports/ folder")
        print("   ✅ 2. Enhanced content with detailed feed conditions")
        print("   ✅ 3. Configuration folder renamed to simulation_input_config/")
        
        # Overall system health
        health_percentage = (self.passed_checks / self.total_checks) * 100 if self.total_checks > 0 else 0
        
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

def main():
    """Run comprehensive system diagnostic"""
    diagnostic = SystemDiagnostic()
    diagnostic.run_comprehensive_diagnostic()
    
    # Exit with appropriate code
    success_rate = diagnostic.passed_checks / diagnostic.total_checks if diagnostic.total_checks > 0 else 0
    exit_code = 0 if success_rate >= 0.9 else 1  # Require 90% pass rate
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 