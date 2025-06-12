"""
Pipeline Utilities Module

Contains utility functions and classes for pipeline management including:
- Pipeline orchestration
- Common utility functions
- Status monitoring
- Configuration management
- DWSIM-RAG integration
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

# Add project root to path
sys.path.append(os.path.abspath('.'))

from .pipeline_rag import RAGPipeline
from .pipeline_dwsim import DWSIMPipeline
from .pipeline_export import ResultsExporter

# Try importing logging config, but don't fail if not available
try:
    from ..utils.logging_config import setup_logging
except ImportError:
    # Fallback logging setup
    def setup_logging():
        import logging
        logging.basicConfig(level=logging.INFO)

class PipelineUtils:
    """Utility class for managing the complete PyNucleus pipeline with DWSIM-RAG integration."""
    
    def __init__(self, results_dir: str = "data/05_output/results", 
                 llm_output_dir: str = "data/05_output/llm_reports",
                 enable_dwsim_integration: bool = True):
        """Initialize pipeline utilities with specified directories."""
        self.results_dir = Path(results_dir)
        self.llm_output_dir = Path(llm_output_dir)
        self.enable_dwsim_integration = enable_dwsim_integration
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.llm_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline components with DWSIM integration
        self.rag_pipeline = RAGPipeline(results_dir, enable_dwsim_integration=enable_dwsim_integration)
        self.dwsim_pipeline = DWSIMPipeline(results_dir)
        self.exporter = ResultsExporter(results_dir)
        
        print(f"🔧 Pipeline Utils initialized with results dir: {self.results_dir}")
        if enable_dwsim_integration:
            print(f"🔗 DWSIM-RAG integration enabled")
    
    def run_complete_pipeline(self):
        """Run the complete PyNucleus pipeline - RAG + DWSIM + Export with integration."""
        print("🚀 Running complete PyNucleus pipeline...")
        start_time = datetime.now()
        
        try:
            # Clear previous results
            self.rag_pipeline.clear_results()
            self.dwsim_pipeline.clear_results()
            
            # First, run DWSIM simulations to generate data for integration
            print("🔬 Step 1: Running DWSIM simulations...")
            dwsim_data = self.dwsim_pipeline.run_simulations()
            dwsim_stats = self.dwsim_pipeline.get_statistics()
            
            # Run RAG pipeline (which will automatically integrate DWSIM data if available)
            print("\n📚 Step 2: Running RAG pipeline with DWSIM integration...")
            self.rag_pipeline.run_pipeline()
            rag_data = self.rag_pipeline.test_queries()
            rag_stats = self.rag_pipeline.get_statistics()
            
            # Export DWSIM simulation results
            print("\n💾 Step 3: Exporting results...")
            exported_files = self.exporter.export_simulation_only(dwsim_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Print integration status
            if rag_stats.get("has_simulation_data", False):
                print(f"\n🎉 Complete integrated pipeline finished in {duration:.1f} seconds!")
                print(f"📊 Unified Knowledge Base:")
                print(f"   ├── Total Chunks: {rag_stats.get('total_chunks', 0):,}")
                print(f"   ├── Document Chunks: {rag_stats.get('document_chunks', 0):,}")
                print(f"   └── Simulation Chunks: {rag_stats.get('simulation_chunks', 0):,}")
            else:
                print(f"✅ Complete pipeline finished in {duration:.1f} seconds!")
            
            return {
                'rag_data': rag_data,
                'dwsim_data': dwsim_data,
                'rag_stats': rag_stats,
                'dwsim_stats': dwsim_stats,
                'exported_files': exported_files,
                'duration': duration,
                'integration_enabled': self.enable_dwsim_integration,
                'has_integrated_data': rag_stats.get("has_simulation_data", False)
            }
            
        except Exception as e:
            print(f"❌ Pipeline error: {str(e)}")
            return None
    
    def run_rag_only(self):
        """Run only the RAG pipeline with DWSIM integration."""
        print("📚 Running RAG pipeline with DWSIM integration...")
        
        try:
            self.rag_pipeline.clear_results()
            self.rag_pipeline.run_pipeline()
            rag_data = self.rag_pipeline.test_queries()
            rag_stats = self.rag_pipeline.get_statistics()
            
            # Export RAG results
            exported_file = self.exporter.export_rag_results(rag_data, "rag_only_results.csv")
            stats_file = self.exporter.export_statistics(rag_stats, "rag_only_statistics.csv")
            
            if rag_stats.get("has_simulation_data", False):
                print("✅ RAG pipeline with DWSIM integration completed!")
                print(f"📊 Integrated Knowledge Base: {rag_stats.get('total_chunks', 0):,} chunks")
            else:
                print("✅ RAG-only pipeline completed!")
            
            return {
                'rag_data': rag_data,
                'rag_stats': rag_stats,
                'exported_files': [exported_file, stats_file],
                'integration_enabled': self.enable_dwsim_integration,
                'has_integrated_data': rag_stats.get("has_simulation_data", False)
            }
            
        except Exception as e:
            print(f"❌ RAG pipeline error: {str(e)}")
            return None
    
    def run_dwsim_only(self):
        """Run only the DWSIM simulations and export results."""
        print("🔬 Running DWSIM simulations only...")
        
        try:
            self.dwsim_pipeline.clear_results()
            dwsim_data = self.dwsim_pipeline.run_simulations()
            dwsim_stats = self.dwsim_pipeline.get_statistics()
            
            # Export DWSIM results
            exported_file = self.exporter.export_dwsim_results(dwsim_data, "dwsim_only_results.csv")
            stats_file = self.exporter.export_statistics(dwsim_stats, "dwsim_only_statistics.csv")
            
            print("✅ DWSIM-only pipeline completed!")
            return {
                'dwsim_data': dwsim_data,
                'dwsim_stats': dwsim_stats,
                'exported_files': [exported_file, stats_file]
            }
            
        except Exception as e:
            print(f"❌ DWSIM pipeline error: {str(e)}")
            return None
    
    def view_results_summary(self):
        """View comprehensive results summary including integration status."""
        print("\n📊 Pipeline Results Summary")
        print("=" * 60)
        
        # Get RAG statistics
        rag_stats = self.rag_pipeline.get_statistics()
        dwsim_stats = self.dwsim_pipeline.get_statistics()
        
        # RAG Section
        print("📚 RAG Knowledge Base:")
        if "error" not in rag_stats:
            print(f"   Total Chunks: {rag_stats.get('total_chunks', 0):,}")
            
            if rag_stats.get("has_simulation_data", False):
                print(f"   ├── Document Chunks: {rag_stats.get('document_chunks', 0):,}")
                print(f"   └── Simulation Chunks: {rag_stats.get('simulation_chunks', 0):,}")
                print(f"   Integration: ✅ {rag_stats.get('integration_status', 'Unknown')}")
            else:
                print(f"   Document Sources: {len(rag_stats.get('sources', []))}")
                print(f"   Integration: ⚪ Documents only")
                
            print(f"   Avg Chunk Size: {rag_stats.get('avg_chunk_size', 0):.1f} chars")
        else:
            print(f"   ❌ {rag_stats.get('error', 'Unknown error')}")
        
        # DWSIM Section
        print(f"\n🔬 DWSIM Simulations:")
        if "error" not in dwsim_stats:
            print(f"   Total Simulations: {dwsim_stats.get('total_simulations', 0)}")
            print(f"   Success Rate: {dwsim_stats.get('success_rate', 0):.1f}%")
            print(f"   Avg Duration: {dwsim_stats.get('average_duration', 0):.2f}s")
        else:
            print(f"   ❌ {dwsim_stats.get('error', 'Unknown error')}")
        
        # Output Files
        print(f"\n📁 Output Files:")
        output_files = list(self.results_dir.glob("*.csv"))
        if output_files:
            for file in output_files:
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.1f} KB)")
        else:
            print("   No output files found")
        
        print("=" * 60)
    
    def quick_test(self):
        """Quick test of pipeline components and integration status."""
        print("🧪 Quick Pipeline Test")
        print("-" * 30)
        
        # Test RAG pipeline
        rag_stats = self.rag_pipeline.get_statistics()
        print(f"📚 RAG: {rag_stats.get('total_chunks', 0)} chunks available")
        
        if rag_stats.get("has_simulation_data", False):
            print(f"🔗 Integration: ✅ {rag_stats.get('simulation_chunks', 0)} simulation chunks")
        else:
            print(f"🔗 Integration: ⚪ Documents only")
        
        # Test DWSIM pipeline
        dwsim_stats = self.dwsim_pipeline.get_statistics()
        print(f"🔬 DWSIM: {dwsim_stats.get('total_simulations', 0)} simulations")
        
        # Test output directory
        csv_files = list(self.results_dir.glob("*.csv"))
        print(f"📁 Output: {len(csv_files)} CSV files")
        
        return {
            'rag_chunks': rag_stats.get('total_chunks', 0),
            'simulation_chunks': rag_stats.get('simulation_chunks', 0),
            'dwsim_simulations': dwsim_stats.get('total_simulations', 0),
            'csv_files_count': len(csv_files),
            'integration_enabled': rag_stats.get("has_simulation_data", False)
        }
    
    def print_pipeline_status(self):
        """Print detailed pipeline status including integration capabilities."""
        print("\n🔧 PyNucleus Pipeline Status")
        print("=" * 60)
        
        # Pipeline Configuration
        print("⚙️ Configuration:")
        print(f"   Results Directory: {self.results_dir}")
        print(f"   LLM Output Directory: {self.llm_output_dir}")
        print(f"   DWSIM Integration: {'✅ Enabled' if self.enable_dwsim_integration else '❌ Disabled'}")
        
        # Component Status
        print(f"\n📦 Components:")
        print(f"   RAG Pipeline: {'✅ Ready' if self.rag_pipeline else '❌ Not available'}")
        print(f"   DWSIM Pipeline: {'✅ Ready' if self.dwsim_pipeline else '❌ Not available'}")
        print(f"   Results Exporter: {'✅ Ready' if self.exporter else '❌ Not available'}")
        
        # Data Status
        self.rag_pipeline.print_status()
        
        # Integration Capabilities
        if self.enable_dwsim_integration:
            print(f"\n🔗 Integration Capabilities:")
            print(f"   📄 Document Processing: ✅ Available")
            print(f"   🔬 Simulation Integration: ✅ Available")
            print(f"   🤖 LLM Querying: ✅ Enhanced (docs + sims)")
        else:
            print(f"\n🔗 Integration Capabilities:")
            print(f"   📄 Document Processing: ✅ Available")
            print(f"   🔬 Simulation Integration: ❌ Disabled")
            print(f"   🤖 LLM Querying: ⚪ Standard (docs only)")
        
        print("=" * 60)
    
    def clean_all_results(self):
        """Clean all pipeline results and reset integration."""
        print("🧹 Cleaning all pipeline results...")
        
        # Clear component results
        self.rag_pipeline.clear_results()
        self.dwsim_pipeline.clear_results()
        
        # Clean output files
        output_files = list(self.results_dir.glob("*.csv"))
        for file in output_files:
            try:
                file.unlink()
                print(f"   🗑️ Removed: {file.name}")
            except Exception as e:
                print(f"   ⚠️ Could not remove {file.name}: {e}")
        
        print("✅ Cleanup completed!")


def run_full_pipeline(settings, output_dir: Path) -> Dict[str, Any]:
    """
    Execute the full PyNucleus pipeline from command line with configuration settings.
    
    Args:
        settings: AppSettings object loaded from JSON/CSV via Pydantic
        output_dir: Output directory path
        
    Returns:
        Dictionary with pipeline results and status
    """
    # Add src to Python path for CLI usage
    import sys
    from pathlib import Path as PathLib
    src_path = str(PathLib(__file__).parent.parent.parent.parent / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    from ..utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    start_time = datetime.now()
    
    logger.info("🚀 Starting PyNucleus full pipeline...")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"⚙️ Configuration: {len(settings.simulations)} simulations loaded")
    logger.info(f"🔍 RAG settings: top_k={settings.rag.top_k}, threshold={settings.rag.similarity_threshold}")
    logger.info(f"📝 LLM settings: summary_length={settings.llm.summary_length}")
    
    try:
        # Initialize pipeline with DWSIM integration enabled
        pipeline = PipelineUtils(results_dir=str(output_dir), enable_dwsim_integration=True)
        
        # Always run complete pipeline for CLI
        logger.info("🔄 Running complete pipeline (RAG + DWSIM + Export + Integration)")
        results = pipeline.run_complete_pipeline()
        
        if results:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Add CLI-specific metadata
            results.update({
                'cli_execution': True,
                'settings_used': {
                    'simulations_count': len(settings.simulations),
                    'rag_config': settings.rag.model_dump(),
                    'llm_config': settings.llm.model_dump()
                },
                'output_directory': str(output_dir),
                'total_duration': duration,
                'execution_time': start_time.isoformat(),
                'completion_time': end_time.isoformat()
            })
            
            logger.info(f"✅ Pipeline completed successfully in {duration:.1f}s")
            logger.info(f"📊 Results: {len(results.get('rag_data', []))} RAG, {len(results.get('dwsim_data', []))} DWSIM")
            logger.info(f"📁 Files exported: {len(results.get('exported_files', []))}")
            
            if results.get('has_integrated_data', False):
                logger.info("🔗 DWSIM-RAG integration successful")
            
            print(f'✅  CLI available:  python run_pipeline.py --config-path configs/my.json')
            
            return results
        else:
            logger.error("❌ Pipeline execution returned no results")
            return {
                'success': False,
                'error': 'Pipeline execution failed',
                'cli_execution': True,
                'output_directory': str(output_dir)
            }
            
    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'cli_execution': True,
            'output_directory': str(output_dir)
        }

def create_pipeline(results_dir="data/05_output/results", enable_integration=True):
    """Factory function to create a new pipeline instance with integration support."""
    return PipelineUtils(results_dir, enable_dwsim_integration=enable_integration)

def quick_run():
    """Quick function to run the complete pipeline with default settings and integration."""
    pipeline = PipelineUtils(enable_dwsim_integration=True)
    return pipeline.run_complete_pipeline()

def status_check():
    """Quick function to check pipeline status."""
    pipeline = PipelineUtils()
    pipeline.print_pipeline_status()
    return pipeline.get_pipeline_status() 