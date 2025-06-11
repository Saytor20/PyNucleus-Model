"""
Pipeline Utilities Module

Contains utility functions and classes for pipeline management including:
- Pipeline orchestration
- Common utility functions
- Status monitoring
- Configuration management
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

from .pipeline_rag import RAGPipeline
from .pipeline_dwsim import DWSIMPipeline
from .pipeline_export import ResultsExporter

class PipelineUtils:
    """Utility class for managing the complete PyNucleus pipeline."""
    
    def __init__(self, results_dir="data/05_output/results"):
        """Initialize Pipeline Utils with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize pipeline components
        self.rag_pipeline = RAGPipeline(results_dir)
        self.dwsim_pipeline = DWSIMPipeline(results_dir)
        self.exporter = ResultsExporter(results_dir)
        
        print(f"🔧 Pipeline Utils initialized with results dir: {self.results_dir}")
    
    def run_complete_pipeline(self):
        """Run the complete PyNucleus pipeline - RAG + DWSIM + Export."""
        print("🚀 Running complete PyNucleus pipeline...")
        start_time = datetime.now()
        
        try:
            # Clear previous results
            self.rag_pipeline.clear_results()
            self.dwsim_pipeline.clear_results()
            
            # Run RAG pipeline
            self.rag_pipeline.run_pipeline()
            rag_data = self.rag_pipeline.test_queries()
            rag_stats = self.rag_pipeline.get_statistics()
            
            # Run DWSIM simulations
            dwsim_data = self.dwsim_pipeline.run_simulations()
            dwsim_stats = self.dwsim_pipeline.get_statistics()
            
            # Export DWSIM simulation results only (simplified)
            exported_files = self.exporter.export_simulation_only(dwsim_data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            print(f"✅ Complete pipeline finished in {duration:.1f} seconds!")
            return {
                'rag_data': rag_data,
                'dwsim_data': dwsim_data,
                'rag_stats': rag_stats,
                'dwsim_stats': dwsim_stats,
                'exported_files': exported_files,
                'duration': duration
            }
            
        except Exception as e:
            print(f"❌ Pipeline error: {str(e)}")
            return None
    
    def run_rag_only(self):
        """Run only the RAG pipeline and export results."""
        print("📚 Running RAG pipeline only...")
        
        try:
            self.rag_pipeline.clear_results()
            self.rag_pipeline.run_pipeline()
            rag_data = self.rag_pipeline.test_queries()
            rag_stats = self.rag_pipeline.get_statistics()
            
            # Export RAG results
            exported_file = self.exporter.export_rag_results(rag_data, "rag_only_results.csv")
            stats_file = self.exporter.export_statistics(rag_stats, "rag_only_statistics.csv")
            
            print("✅ RAG-only pipeline completed!")
            return {
                'rag_data': rag_data,
                'rag_stats': rag_stats,
                'exported_files': [exported_file, stats_file]
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
    
    def quick_test(self):
        """Quick test function to verify everything is working."""
        print("⚡ Running quick test...")
        
        # Test each component
        print(f"📁 Results directory: {self.results_dir}")
        
        # Check RAG pipeline
        rag_data_count = len(self.rag_pipeline.get_results())
        print(f"📚 RAG data points: {rag_data_count}")
        
        # Check DWSIM pipeline
        dwsim_data_count = len(self.dwsim_pipeline.get_results())
        print(f"🔬 DWSIM data points: {dwsim_data_count}")
        
        # Check CSV files
        csv_files = list(self.results_dir.glob('*.csv'))
        print(f"📈 CSV files available: {len(csv_files)}")
        for f in csv_files:
            print(f"   • {f.name}")
        
        # Test export functionality
        self.exporter.quick_test()
        
        return {
            'rag_data_count': rag_data_count,
            'dwsim_data_count': dwsim_data_count,
            'csv_files_count': len(csv_files)
        }
    
    def view_results_summary(self):
        """View a summary of exported results."""
        self.exporter.view_summary()
    
    def clean_all_results(self):
        """Clean all results and start fresh."""
        print("🗑️ Cleaning all results...")
        
        # Clear pipeline data
        self.rag_pipeline.clear_results()
        self.dwsim_pipeline.clear_results()
        
        # Clean exported files
        self.exporter.clean_results()
        
        print("✅ All results cleaned!")
    
    def get_pipeline_status(self):
        """Get current status of all pipeline components."""
        status = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'results_directory': str(self.results_dir),
            'rag_pipeline': {
                'initialized': self.rag_pipeline.manager is not None,
                'documents_loaded': self.rag_pipeline.documents is not None,
                'results_count': len(self.rag_pipeline.get_results())
            },
            'dwsim_pipeline': {
                'available': self.dwsim_pipeline.dwsim_available,
                'initialized': self.dwsim_pipeline.service is not None,
                'results_count': len(self.dwsim_pipeline.get_results())
            },
            'exported_files': self.exporter.get_latest_files()
        }
        
        return status
    
    def print_pipeline_status(self):
        """Print a formatted pipeline status report."""
        status = self.get_pipeline_status()
        
        print(f"\n📊 Pipeline Status Report - {status['timestamp']}")
        print(f"📁 Results Directory: {status['results_directory']}")
        
        # RAG status
        rag = status['rag_pipeline']
        print(f"\n📚 RAG Pipeline:")
        print(f"   • Initialized: {'✅' if rag['initialized'] else '❌'}")
        print(f"   • Documents Loaded: {'✅' if rag['documents_loaded'] else '❌'}")
        print(f"   • Results Count: {rag['results_count']}")
        
        # DWSIM status
        dwsim = status['dwsim_pipeline']
        print(f"\n🔬 DWSIM Pipeline:")
        print(f"   • Available: {'✅' if dwsim['available'] else '❌'}")
        print(f"   • Initialized: {'✅' if dwsim['initialized'] else '❌'}")
        print(f"   • Results Count: {dwsim['results_count']}")
        
        # Files status
        files = status['exported_files']
        print(f"\n📈 Exported Files ({len(files)}):")
        if files:
            for file_info in files[:5]:  # Show latest 5 files
                print(f"   • {file_info['filename']} ({file_info['size_bytes']:,} bytes) - {file_info['modified']}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more files")
        else:
            print("   No exported files found.")

def create_pipeline(results_dir="data/05_output/results"):
    """Factory function to create a new pipeline instance."""
    return PipelineUtils(results_dir)

def quick_run():
    """Quick function to run the complete pipeline with default settings."""
    pipeline = PipelineUtils()
    return pipeline.run_complete_pipeline()

def status_check():
    """Quick function to check pipeline status."""
    pipeline = PipelineUtils()
    pipeline.print_pipeline_status()
    return pipeline.get_pipeline_status() 