"""
Pipeline utilities for PyNucleus system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

class PipelineUtils:
    """Main utility class for PyNucleus pipeline operations."""
    
    def __init__(self, results_dir: str = "data/05_output"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline components
        self.rag_pipeline = None
        self._dwsim_pipeline = None
        
        # Setup basic logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize pipelines
        self._initialize_pipelines()
        
    def _initialize_pipelines(self):
        """Initialize RAG and DWSIM pipeline components."""
        try:
            from .pipeline_rag import RAGPipeline
            from .pipeline_dwsim import DWSIMPipeline
            from ..rag.vector_store import RealFAISSVectorStore
            
            self.rag_pipeline = RAGPipeline(data_dir="data")
            self._dwsim_pipeline = DWSIMPipeline()
            
            # Initialize real FAISS vector store
            self.real_vector_store = RealFAISSVectorStore()
            
            self.logger.info("Pipeline components initialized successfully")
            if self.real_vector_store.loaded:
                self.logger.info("Real FAISS vector store loaded successfully")
            else:
                self.logger.warning("FAISS vector store not loaded - using fallback mode")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {e}")
            self.real_vector_store = None
    
    def print_pipeline_status(self) -> None:
        """Print the current status of all pipeline components."""
        print("\n" + "="*50)
        print("     PYNUCLEUS PIPELINE STATUS")
        print("="*50)
        
        # Check RAG Pipeline
        rag_status = "✅ Initialized" if self.rag_pipeline else "❌ Not Initialized"
        print(f"📚 RAG Pipeline:          {rag_status}")
        
        # Check DWSIM Pipeline
        dwsim_status = "✅ Initialized" if self._dwsim_pipeline else "❌ Not Initialized"
        print(f"🔬 DWSIM Pipeline:        {dwsim_status}")
        
        # Check FAISS Vector Store
        if hasattr(self, 'real_vector_store') and self.real_vector_store:
            faiss_status = "✅ Loaded" if self.real_vector_store.loaded else "⚠️  Not Loaded (Fallback Mode)"
        else:
            faiss_status = "❌ Not Available"
        print(f"🔍 FAISS Vector Store:    {faiss_status}")
        
        # Check Results Directory
        results_exists = self.results_dir.exists()
        results_status = "✅ Available" if results_exists else "❌ Not Found"
        print(f"📁 Results Directory:     {results_status}")
        
        # Count existing results
        if results_exists:
            csv_files = list(self.results_dir.glob("*.csv"))
            json_files = list(self.results_dir.glob("*.json"))
            print(f"   └─ CSV Files:          {len(csv_files)}")
            print(f"   └─ JSON Files:         {len(json_files)}")
        
        # Overall System Health
        components_ready = all([
            self.rag_pipeline is not None,
            self._dwsim_pipeline is not None,
            results_exists
        ])
        
        overall_status = "🟢 READY" if components_ready else "🟡 PARTIAL"
        print(f"\n🎯 Overall Status:        {overall_status}")
        
        if components_ready:
            print("   └─ System ready for pipeline execution")
        else:
            print("   └─ Some components need initialization")
            
        print("="*50)
    
    @property
    def dwsim_pipeline(self):
        """Get DWSIM pipeline with backward compatibility for get_results()."""
        if not hasattr(self, '_dwsim_pipeline') or self._dwsim_pipeline is None:
            return None
        return self._get_dwsim_pipeline_with_compat()
    
    @dwsim_pipeline.setter
    def dwsim_pipeline(self, value):
        """Set DWSIM pipeline."""
        self._dwsim_pipeline = value
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete PyNucleus pipeline (RAG + DWSIM + Export).
        
        Returns:
            Dictionary with complete pipeline results
        """
        start_time = time.time()
        self.logger.info("Starting complete pipeline execution")
        
        try:
            # Step 1: Run RAG analysis
            rag_results = self._run_rag_analysis()
            
            # Step 2: Run DWSIM simulations
            dwsim_results = self._run_dwsim_simulations()
            
            # Step 3: Export results
            exported_files = self._export_results(rag_results, dwsim_results)
            
            duration = time.time() - start_time
            
            results = {
                "success": True,
                "duration": duration,
                "rag_data": rag_results,
                "dwsim_data": dwsim_results,
                "exported_files": exported_files,
                "timestamp": datetime.now().isoformat(),
                "results_dir": str(self.results_dir)
            }
            
            self.logger.info(f"Complete pipeline execution completed in {duration:.1f} seconds")
            return results
            
        except Exception as e:
            self.logger.error(f"Complete pipeline execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def run_rag_only(self) -> Dict[str, Any]:
        """
        Run only the RAG pipeline.
        
        Returns:
            Dictionary with RAG results
        """
        self.logger.info("Running RAG-only pipeline")
        
        try:
            rag_results = self._run_rag_analysis()
            exported_files = self._export_rag_results(rag_results)
            
            return {
                "success": True,
                "rag_data": rag_results,
                "exported_files": exported_files,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"RAG-only pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_dwsim_only(self) -> Dict[str, Any]:
        """
        Run only the DWSIM simulation pipeline.
        
        Returns:
            Dictionary with DWSIM results
        """
        self.logger.info("Running DWSIM-only pipeline")
        
        try:
            dwsim_results = self._run_dwsim_simulations()
            exported_files = self._export_dwsim_results(dwsim_results)
            
            return {
                "success": True,
                "dwsim_data": dwsim_results,
                "exported_files": exported_files,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"DWSIM-only pipeline failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _run_rag_analysis(self) -> List[Dict[str, Any]]:
        """Run RAG analysis with predefined queries using real FAISS vector store."""
        if not self.rag_pipeline:
            raise Exception("RAG pipeline not initialized")
        
        self.logger.info("Starting RAG document analysis...")
        
        # Simulate realistic processing time
        time.sleep(2)  # Simulate document loading and processing
        
        # Load documents first
        self.rag_pipeline.load_documents()
        
        # Predefined queries for chemical engineering
        queries = [
            "What are the key parameters for distillation optimization?",
            "How to improve process efficiency in chemical plants?",
            "Best practices for reactor design in chemical processes?"
        ]
        
        results = []
        for i, query in enumerate(queries):
            self.logger.info(f"Processing RAG query {i+1}/{len(queries)}: {query[:50]}...")
            time.sleep(1)  # Simulate query processing time
            
            # Use real FAISS vector store if available
            if hasattr(self, 'real_vector_store') and self.real_vector_store:
                # Use real FAISS search
                search_results = self.real_vector_store.search(query, top_k=3, similarity_threshold=0.3)
                
                if search_results:
                    # Format results in expected structure
                    answer_parts = []
                    sources = []
                    for search_result in search_results:
                        answer_parts.append(search_result["text"][:200] + "...")
                        sources.append(search_result["source"])
                    
                    result = {
                        "answer": " ".join(answer_parts) if answer_parts else f"No relevant documents found for: {query}",
                        "sources": sources,
                        "confidence": search_results[0]["score"] if search_results else 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "search_results": search_results,
                        "query": query,
                        "vector_store_used": "Real FAISS"
                    }
                else:
                    # No results found
                    result = {
                        "answer": f"No relevant documents found for query: {query}",
                        "sources": [],
                        "confidence": 0.0,
                        "timestamp": datetime.now().isoformat(),
                        "query": query,
                        "vector_store_used": "Real FAISS (no results)"
                    }
            else:
                # Fallback to original RAG pipeline
                result = self.rag_pipeline.query(query)
                result["vector_store_used"] = "Fallback RAG Pipeline"
            
            results.append(result)
        
        self.logger.info(f"RAG analysis completed: {len(results)} queries processed")
        return results
    
    def _run_dwsim_simulations(self) -> List[Dict[str, Any]]:
        """Run DWSIM simulations with predefined configurations."""
        if not hasattr(self, '_dwsim_pipeline') or not self._dwsim_pipeline:
            raise Exception("DWSIM pipeline not initialized")
        
        self.logger.info("Starting DWSIM chemical process simulations...")
        
        # Predefined simulation configurations
        configs = [
            {
                "case_name": "distillation_ethanol_water",
                "type": "distillation",
                "components": "ethanol, water",
                "feed_temperature": 25.0,
                "feed_pressure": 1.0
            },
            {
                "case_name": "methanol_synthesis",
                "type": "reactor",
                "components": "CO, H2, methanol",
                "reaction_temperature": 250.0,
                "reaction_pressure": 50.0
            },
            {
                "case_name": "heat_exchanger_optimization",
                "type": "heat_exchanger",
                "components": "hot_stream, cold_stream",
                "inlet_temperature": 80.0,
                "outlet_temperature": 40.0
            }
        ]
        
        results = []
        for i, config in enumerate(configs):
            self.logger.info(f"Running DWSIM simulation {i+1}/{len(configs)}: {config['case_name']}")
            time.sleep(3)  # Simulate realistic simulation time
            result = self._dwsim_pipeline.run_simulation(config)
            results.append(result)
        
        self.logger.info(f"DWSIM simulations completed: {len(results)} scenarios processed")
        return results
    
    def _export_results(self, rag_results: List[Dict[str, Any]], dwsim_results: List[Dict[str, Any]]) -> List[str]:
        """Export combined results to files."""
        exported_files = []
        
        # Export RAG results
        rag_files = self._export_rag_results(rag_results)
        exported_files.extend(rag_files)
        
        # Export DWSIM results
        dwsim_files = self._export_dwsim_results(dwsim_results)
        exported_files.extend(dwsim_files)
        
        return exported_files
    
    def _export_rag_results(self, rag_results: List[Dict[str, Any]]) -> List[str]:
        """Export RAG results to JSON file."""
        try:
            # Ensure results subdirectory exists
            results_subdir = self.results_dir / "results"
            results_subdir.mkdir(parents=True, exist_ok=True)
            
            export_file = results_subdir / f"rag_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(export_file, 'w') as f:
                json.dump(rag_results, f, indent=2)
            
            self.logger.info(f"RAG results exported to: {export_file}")
            return [str(export_file)]
            
        except Exception as e:
            self.logger.error(f"Failed to export RAG results: {e}")
            return []
    
    def _export_dwsim_results(self, dwsim_results: List[Dict[str, Any]]) -> List[str]:
        """Export DWSIM results to JSON file."""
        try:
            # Ensure results subdirectory exists
            results_subdir = self.results_dir / "results"
            results_subdir.mkdir(parents=True, exist_ok=True)
            
            export_file = results_subdir / f"dwsim_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(export_file, 'w') as f:
                json.dump(dwsim_results, f, indent=2)
            
            self.logger.info(f"DWSIM results exported to: {export_file}")
            return [str(export_file)]
            
        except Exception as e:
            self.logger.error(f"Failed to export DWSIM results: {e}")
            return []

    def _get_dwsim_pipeline_with_compat(self):
        """Get DWSIM pipeline with get_results method compatibility."""
        if not hasattr(self, '_dwsim_pipeline') or self._dwsim_pipeline is None:
            return None
            
        # Create a wrapper that provides get_results() method compatibility
        class DWSIMPipelineWrapper:
            def __init__(self, pipeline):
                self._pipeline = pipeline
            
            def __getattr__(self, name):
                return getattr(self._pipeline, name)
            
            def get_results(self):
                """Alias for get_simulation_results() for notebook compatibility."""
                return self._pipeline.get_simulation_results()
            
            def get_simulation_results(self):
                return self._pipeline.get_simulation_results()
        
        return DWSIMPipelineWrapper(self._dwsim_pipeline)

    def quick_test(self) -> Dict[str, Any]:
        """Quick test to verify basic functionality."""
        try:
            # Count CSV and JSON files in configs and results directories
            config_dir = Path("configs")
            results_main_dir = self.results_dir
            results_subdir = self.results_dir / "results"
            
            # Get CSV files from configs
            config_csv_files = list(config_dir.glob("*.csv")) if config_dir.exists() else []
            
            # Get JSON files from results directories
            json_files = []
            if results_main_dir.exists():
                json_files.extend(list(results_main_dir.glob("*.json")))
            if results_subdir.exists():
                json_files.extend(list(results_subdir.glob("*.json")))
            
            # Combine all files
            all_files = config_csv_files + json_files
            
            # Create file info list with details
            files_info = []
            for file_path in all_files:
                try:
                    file_size = file_path.stat().st_size
                    file_type = "Config CSV" if file_path.suffix == ".csv" else "Pipeline JSON"
                    files_info.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_size,
                        "type": file_type,
                        "location": file_path.parent.name
                    })
                except Exception:
                    # If we can't get file stats, include it anyway
                    files_info.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": 0,
                        "type": "Unknown",
                        "location": file_path.parent.name if file_path.parent else "unknown"
                    })
            
            return {
                "status": "success",
                "csv_files_count": len(config_csv_files),
                "json_files_count": len(json_files),
                "total_files_count": len(all_files),
                "csv_files": [f for f in files_info if f["type"] == "Config CSV"],
                "json_files": [f for f in files_info if f["type"] == "Pipeline JSON"],
                "all_files": files_info,
                "results_dir": str(self.results_dir),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error", 
                "error": str(e),
                "csv_files_count": 0,
                "json_files_count": 0,
                "total_files_count": 0,
                "csv_files": [],
                "json_files": [],
                "all_files": [],
                "results_dir": str(self.results_dir)
            }

    def view_results_summary(self):
        """Display a comprehensive results summary."""
        try:
            print("📊 PyNucleus System Summary")
            print("-" * 30)
            
            # Check actual result directories and files
            main_results_dir = self.results_dir
            results_subdir = self.results_dir / "results"
            
            # Count different types of files
            json_files_main = list(main_results_dir.glob("*.json")) if main_results_dir.exists() else []
            json_files_sub = list(results_subdir.glob("*.json")) if results_subdir.exists() else []
            csv_files = list(Path("configs").glob("*.csv")) if Path("configs").exists() else []
            
            print(f"📁 Main Results Dir ({main_results_dir.name}): {len(json_files_main)} JSON files")
            print(f"📁 Results Subdir ({results_subdir.name if results_subdir.exists() else 'N/A'}): {len(json_files_sub)} JSON files")
            print(f"📁 Config Files: {len(csv_files)} CSV files")
            
            # Check pipeline status
            print("\n🔧 Pipeline Status:")
            print(f"   • RAG Pipeline: {'Initialized' if self.rag_pipeline else 'Available'}")
            print(f"   • DWSIM Pipeline: {'Initialized' if self.dwsim_pipeline else 'Available'}")
            print(f"   • Results Directory: {self.results_dir}")
            
            # Show recent activity
            all_json_files = json_files_main + json_files_sub
            if all_json_files:
                # Sort by modification time
                files_with_time = []
                for file_path in all_json_files:
                    try:
                        mtime = file_path.stat().st_mtime
                        files_with_time.append((file_path.name, mtime))
                    except:
                        pass
                
                if files_with_time:
                    files_with_time.sort(key=lambda x: x[1], reverse=True)
                    print(f"\n📋 Recent Generated Files:")
                    for file_name, _ in files_with_time[:5]:  # Show top 5 recent files
                        print(f"   • {file_name}")
            else:
                print(f"\n📋 No pipeline result files found. Run pipeline to generate results.")
                
        except Exception as e:
            print(f"❌ Error displaying summary: {e}")
            print("💡 Try running the pipeline first to generate results")
    
    def clean_all_results(self):
        """Clean up all previous pipeline results."""
        try:
            # Clear DWSIM results
            if hasattr(self, '_dwsim_pipeline') and self._dwsim_pipeline:
                self._dwsim_pipeline.clear_results()
            
            # Clear result files
            if self.results_dir.exists():
                for file_path in self.results_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()
                        
            self.logger.info("All pipeline results cleared")
            print("✅ All previous results have been cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clean results: {e}")
            print(f"❌ Error cleaning results: {e}")


def run_full_pipeline(settings: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run the full PyNucleus pipeline with given settings.
    
    Args:
        settings: Configuration dictionary
        output_dir: Output directory path
        
    Returns:
        Dictionary with pipeline results
    """
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline
        pipeline = PipelineUtils(str(output_dir))
        
        # Mock pipeline execution for now
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "output_dir": str(output_dir),
            "settings": settings,
            "summary": "Pipeline completed successfully (mock implementation)"
        }
        
        # Save results
        results_file = output_dir / "pipeline_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        return results
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 