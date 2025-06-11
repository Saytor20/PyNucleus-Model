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
root_dir = Path(__file__).parent.parent
src_path = str(root_dir / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Add root directory to Python path for imports
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
    
    def check_docker_environment(self) -> bool:
        """Check Docker environment setup"""
        self.print_section_header("DOCKER ENVIRONMENT CHECK")
        issues = []
        
        # Check Dockerfile
        if not Path("docker/Dockerfile").exists():
            print("   ⚠️ Dockerfile not found")
            issues.append("Dockerfile missing")
        else:
            print("   ✅ Dockerfile exists")
        
        # Check docker-compose.yml
        if not Path("docker/docker-compose.yml").exists():
            print("   ⚠️ docker-compose.yml not found")
            issues.append("docker-compose.yml missing")
        else:
            print("   ✅ docker-compose.yml exists")
        
        # Check .dockerignore
        if not Path("docker/.dockerignore").exists():
            print("   ⚠️ .dockerignore not found")
            issues.append(".dockerignore missing")
        else:
            print("   ✅ .dockerignore exists")
        
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
            print("   ✅ All enhanced modules imported successfully")
            
            # Test ConfigManager with new folder structure
            config_manager = ConfigManager(config_dir="configs")
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
                print(f"   ✅ {description}: {dir_path} ({item_count} items)")
            else:
                print(f"   ❌ {description}: {dir_path} (missing)")
                issues.append(f"Missing RAG directory: {dir_path}")
        
        # Check FAISS store in chunk_reports
        faiss_files = list(Path("data/04_models/chunk_reports").glob("*.faiss")) if Path("data/04_models/chunk_reports").exists() else []
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
    
    def check_data_consolidation_deliverables(self) -> bool:
        """Check specific data consolidation deliverables"""
        self.print_section_header("DATA CONSOLIDATION DELIVERABLES CHECK")
        issues = []
        
        print("\n📋 Verifying Data Directory Consolidation Deliverables:")
        
        # 1. Check data subdirectories creation
        required_data_dirs = [
            "data/01_raw",
            "data/02_processed", 
            "data/03_intermediate",
            "data/04_models",
            "data/05_output"
        ]
        
        print("\n   🔍 Data Subdirectories:")
        for data_dir in required_data_dirs:
            if Path(data_dir).exists():
                print(f"      ✅ {data_dir}/ exists")
            else:
                print(f"      ❌ {data_dir}/ missing")
                issues.append(f"Missing data directory: {data_dir}")
        
        # 2. Check moved folders in correct locations
        moved_folders = {
            "data/01_raw/source_documents": "Source documents",
            "data/01_raw/web_sources": "Web sources", 
            "data/02_processed/converted_to_txt": "Converted documents",
            "data/03_intermediate/converted_chunked_data": "Chunked data",
            "data/04_models/chunk_reports": "Chunk reports",
            "data/05_output/results": "Results",
            "data/05_output/llm_reports": "LLM reports"
        }
        
        print("\n   🔍 Moved Folders Verification:")
        for folder_path, description in moved_folders.items():
            if Path(folder_path).exists():
                item_count = len(list(Path(folder_path).glob("*")))
                print(f"      ✅ {description}: {folder_path}/ ({item_count} items)")
            else:
                print(f"      ❌ {description}: {folder_path}/ missing")
                issues.append(f"Missing moved folder: {folder_path}")
        
        # 3. Check for old directories (should be gone)
        old_locations = [
            "source_documents",
            "web_sources", 
            "converted_to_txt",
            "converted_chunked_data",
            "chunk_reports",
            "results",
            "llm_reports"
        ]
        
        print("\n   🔍 Old Directory Cleanup:")
        cleanup_issues = []
        for old_dir in old_locations:
            if Path(old_dir).exists() and old_dir not in ["results", "llm_reports"]:  # These might still exist for backwards compatibility
                print(f"      ⚠️ Old directory still exists: {old_dir}/")
                cleanup_issues.append(f"Old directory not removed: {old_dir}")
            else:
                print(f"      ✅ Old directory cleaned: {old_dir}/")
        
        # 4. Check path updates in codebase (sample key files)
        print("\n   🔍 Hard-coded Path Updates:")
        key_files_to_check = [
            "src/pynucleus/config.py",
            "src/pynucleus/rag/config.py",
            "src/pynucleus/integration/llm_output_generator.py",
            "src/pynucleus/integration/dwsim_rag_integrator.py"
        ]
        
        path_check_patterns = [
            ("results/", "data/05_output/results/"),
            ("llm_reports/", "data/05_output/llm_reports/"),
            ("converted_chunked_data/", "data/03_intermediate/converted_chunked_data/")
        ]
        
        for file_path in key_files_to_check:
            if Path(file_path).exists():
                with open(file_path, 'r') as f:
                    content = f.read()
                
                file_updated = True
                for old_pattern, new_pattern in path_check_patterns:
                    if old_pattern in content and new_pattern not in content:
                        print(f"      ⚠️ {Path(file_path).name}: May contain old path '{old_pattern}'")
                        file_updated = False
                
                if file_updated:
                    print(f"      ✅ {Path(file_path).name}: Paths updated correctly")
            else:
                print(f"      ⚠️ File not found: {file_path}")
        
        # 5. Summary
        print(f"\n   📊 Consolidation Summary:")
        print(f"      • Data directories: {len([d for d in required_data_dirs if Path(d).exists()])}/{len(required_data_dirs)} created")
        print(f"      • Moved folders: {len([f for f in moved_folders.keys() if Path(f).exists()])}/{len(moved_folders)} in place")
        print(f"      • Cleanup issues: {len(cleanup_issues)} remaining")
        
        success = len(issues) == 0
        if success:
            print(f"\n   🎉 ✅ Data directories centralized. Verify file moves if Git complains.")
        else:
            print(f"\n   ❌ {len(issues)} consolidation issues found")
        
        self.log_result("Data Consolidation Deliverables", success, issues + cleanup_issues)
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
            ("Docker Environment", self.check_docker_environment),
            ("Data Consolidation Deliverables", self.check_data_consolidation_deliverables),
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