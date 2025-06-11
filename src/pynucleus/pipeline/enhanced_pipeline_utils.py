"""
Enhanced Pipeline Utils with DWSIM-RAG Integration

This module extends the basic pipeline functionality with:
- Configurable DWSIM simulations (JSON/CSV input)
- DWSIM-RAG integration for enhanced analysis
- LLM-ready output generation
- Advanced result interpretation
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union

# Add project root to path
sys.path.append(os.path.abspath('.'))

try:
    from pynucleus.pipeline.pipeline_utils import PipelineUtils
    from pynucleus.integration.config_manager import ConfigManager
    from pynucleus.integration.dwsim_rag_integrator import DWSIMRAGIntegrator
    # from pynucleus.integration.llm_output_generator import LLMOutputGenerator
except ImportError as e:
    print(f"Import warning: {e}")
    # Define minimal classes for fallback
    class ConfigManager:
        def __init__(self, *args, **kwargs): pass
        def create_template_json(self, *args, **kwargs): return "template.json"
        def create_template_csv(self, *args, **kwargs): return "template.csv"
        def load_from_json(self, *args, **kwargs): return []
        def load_from_csv(self, *args, **kwargs): return []
    
    class DWSIMRAGIntegrator:
        def __init__(self, *args, **kwargs): pass
        def integrate_simulation_results(self, *args, **kwargs): return []
        def export_integrated_results(self, *args, **kwargs): return "results.json"


class EnhancedPipelineUtils:
    """Enhanced Pipeline with DWSIM-RAG integration and configurable simulations."""
    
    def __init__(self, results_dir: str = "results", config_dir: str = "simulation_input_config"):
        """Initialize enhanced pipeline."""
        self.results_dir = Path(results_dir)
        self.config_dir = Path(config_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)
        
        # Initialize components
        try:
            self.base_pipeline = PipelineUtils(results_dir)
        except:
            print("⚠️ Base pipeline not available - using minimal mode")
            self.base_pipeline = None
            
        self.config_manager = ConfigManager(config_dir)
        self.dwsim_rag_integrator = None
        self.llm_generator = None
        
        # Results storage
        self.enhanced_results = {}
        
        print("🚀 Enhanced Pipeline initialized!")
        print(f"   📁 Results directory: {self.results_dir}")
        print(f"   ⚙️ Configuration directory: {self.config_dir}")
    
    def setup_configuration_templates(self) -> Dict[str, str]:
        """Create configuration templates for easy DWSIM setup."""
        print("📋 Setting up configuration templates...")
        
        templates = {}
        
        # Create JSON template
        json_template = self.config_manager.create_template_json("dwsim_simulations_template.json")
        templates['json'] = json_template
        
        # Create CSV template  
        csv_template = self.config_manager.create_template_csv("dwsim_simulations_template.csv")
        templates['csv'] = csv_template
        
        print("✅ Configuration templates created!")
        print(f"   📄 JSON template: {json_template}")
        print(f"   📊 CSV template: {csv_template}")
        print()
        print("💡 Usage Instructions:")
        print("   1. Edit the template files with your simulation parameters")
        print("   2. Use load_simulations_from_config() to load your configurations")
        print("   3. Run enhanced_pipeline() to execute with RAG integration")
        
        return templates
    
    def load_simulations_from_config(self, config_file: Union[str, Path], 
                                   file_type: str = 'auto') -> List[Dict]:
        """Load simulation configurations from JSON or CSV file."""
        
        config_path = Path(config_file)
        
        if not config_path.exists():
            print(f"❌ Configuration file not found: {config_path}")
            return []
        
        # Auto-detect file type
        if file_type == 'auto':
            file_type = config_path.suffix.lower()
        
        print(f"📥 Loading simulations from {file_type} file: {config_path}")
        
        try:
            if file_type in ['.json', 'json']:
                simulations = self.config_manager.load_from_json(config_path)
            elif file_type in ['.csv', 'csv']:
                simulations = self.config_manager.load_from_csv(config_path)
            else:
                print(f"❌ Unsupported file type: {file_type}")
                return []
            
            print(f"✅ Loaded {len(simulations)} simulation configurations")
            return simulations
            
        except Exception as e:
            print(f"❌ Error loading configurations: {e}")
            return []
    
    def run_enhanced_pipeline(self, simulation_configs: Optional[List[Dict]] = None,
                            config_file: Optional[str] = None,
                            include_rag_analysis: bool = True,
                            generate_llm_output: bool = True) -> Dict:
        """Run the complete enhanced pipeline with DWSIM-RAG integration."""
        
        start_time = datetime.now()
        print("🚀 Starting Enhanced PyNucleus Pipeline...")
        print(f"   🕐 Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {
            'pipeline_type': 'enhanced',
            'start_time': start_time.isoformat(),
            'rag_data': [],
            'dwsim_data': [],
            'integrated_data': [],
            'llm_output_file': None,
            'exported_files': [],
            'statistics': {},
            'errors': []
        }
        
        try:
            # Step 1: Load simulation configurations
            if config_file:
                simulation_configs = self.load_simulations_from_config(config_file)
            
            if not simulation_configs:
                print("📋 No simulation configurations provided, using defaults...")
                simulation_configs = self._get_default_simulations()
            
            # Step 2: Run RAG Pipeline (if base pipeline available)
            if self.base_pipeline and include_rag_analysis:
                print("\n📚 Running RAG Pipeline...")
                rag_results = self.base_pipeline.run_rag_only()
                if rag_results:
                    results['rag_data'] = rag_results['rag_data']
                    print(f"✅ RAG completed: {len(results['rag_data'])} queries processed")
            
            # Step 3: Run DWSIM Simulations with custom configurations
            print(f"\n🔬 Running {len(simulation_configs)} custom DWSIM simulations...")
            dwsim_results = self._run_custom_dwsim_simulations(simulation_configs)
            results['dwsim_data'] = dwsim_results
            print(f"✅ DWSIM completed: {len(dwsim_results)} simulations processed")
            
            # Step 4: DWSIM-RAG Integration
            if include_rag_analysis and self.base_pipeline:
                print("\n🔗 Performing DWSIM-RAG Integration...")
                self.dwsim_rag_integrator = DWSIMRAGIntegrator(
                    rag_pipeline=self.base_pipeline.rag_pipeline,
                    results_dir=self.results_dir
                )
                
                integrated_results = self.dwsim_rag_integrator.integrate_simulation_results(
                    dwsim_results, perform_rag_analysis=True
                )
                results['integrated_data'] = integrated_results
                
                # Export integrated results
                integrated_file = self.dwsim_rag_integrator.export_integrated_results()
                results['exported_files'].append(integrated_file)
                print(f"✅ Integration completed: {len(integrated_results)} enhanced results")
            
            # Step 5: Generate LLM-ready output
            if generate_llm_output and results['integrated_data']:
                print("\n📝 Generating LLM-ready output...")
                llm_output = self._generate_llm_output(results['integrated_data'])
                results['llm_output_file'] = llm_output
                results['exported_files'].append(llm_output)
                print(f"✅ LLM output generated: {llm_output}")
            
            # Step 6: Export all results
            print("\n💾 Exporting comprehensive results...")
            comprehensive_export = self._export_comprehensive_results(results)
            results['exported_files'].append(comprehensive_export)
            
            # Calculate final statistics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results.update({
                'end_time': end_time.isoformat(),
                'duration': duration,
                'success': True,
                'statistics': self._calculate_enhanced_statistics(results)
            })
            
            print(f"\n🎉 Enhanced pipeline completed successfully!")
            print(f"   ⏱️ Total duration: {duration:.1f} seconds")
            print(f"   📊 Results: {len(results['dwsim_data'])} simulations, {len(results['rag_data'])} RAG queries")
            print(f"   📁 Files exported: {len(results['exported_files'])}")
            
        except Exception as e:
            print(f"\n❌ Pipeline error: {str(e)}")
            results['errors'].append(str(e))
            results['success'] = False
        
        self.enhanced_results = results
        return results
    
    def _get_default_simulations(self) -> List[Dict]:
        """Get default simulation configurations."""
        return [
            {
                'name': 'enhanced_ethanol_distillation',
                'type': 'distillation',
                'components': ['water', 'ethanol'],
                'description': 'Enhanced ethanol-water separation with optimized parameters',
                'parameters': {
                    'temperature': 78.4,
                    'pressure': 101325,
                    'flow_rate': 1000
                }
            },
            {
                'name': 'enhanced_methane_reforming',
                'type': 'reactor',
                'components': ['methane', 'steam', 'hydrogen'],
                'description': 'Steam methane reforming with improved catalyst',
                'parameters': {
                    'temperature': 900,
                    'pressure': 2500000,
                    'flow_rate': 500
                }
            }
        ]
    
    def _run_custom_dwsim_simulations(self, simulation_configs: List[Dict]) -> List[Dict]:
        """Run DWSIM simulations with custom configurations."""
        
        if self.base_pipeline and hasattr(self.base_pipeline, 'dwsim_pipeline'):
            # Use base pipeline DWSIM with custom configs
            dwsim_pipeline = self.base_pipeline.dwsim_pipeline
            
            # Convert configs to DWSIM format
            dwsim_cases = []
            for config in simulation_configs:
                dwsim_case = {
                    'name': config.get('name', 'unknown'),
                    'type': config.get('type', 'reactor'),
                    'components': config.get('components', []),
                    'description': config.get('description', ''),
                }
                dwsim_cases.append(dwsim_case)
            
            # Run simulations
            return dwsim_pipeline.run_simulations(dwsim_cases)
        else:
            # Generate mock results for demonstration
            print("⚠️ Using mock DWSIM results for demonstration")
            return self._generate_mock_dwsim_results(simulation_configs)
    
    def _generate_mock_dwsim_results(self, simulation_configs: List[Dict]) -> List[Dict]:
        """Generate mock DWSIM results for demonstration."""
        
        mock_results = []
        for config in simulation_configs:
            result = {
                'case_name': config.get('name', 'unknown'),
                'simulation_type': config.get('type', 'reactor'),
                'components': ', '.join(config.get('components', [])),
                'description': config.get('description', ''),
                'success': True,
                'duration_seconds': 0.001,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'result_summary': {
                    'simulation_id': f"enhanced_{config.get('name', 'unknown')}",
                    'success': True,
                    'results': {
                        'conversion': 0.85,
                        'selectivity': 0.92,
                        'yield': 0.78
                    },
                    'parameters_used': config.get('parameters', {})
                }
            }
            mock_results.append(result)
        
        return mock_results
    
    def _generate_llm_output(self, integrated_results: List[Dict]) -> str:
        """Generate LLM-ready text output."""
        
        # Simple text generation since LLMOutputGenerator might not be available
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_simulation_summary_{timestamp}.txt"
        output_file = self.results_dir / filename
        
        # Generate summary text
        summary_text = self._create_text_summary(integrated_results)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        return str(output_file)
    
    def _create_text_summary(self, integrated_results: List[Dict]) -> str:
        """Create a comprehensive text summary."""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        summary_parts = [
            "# PyNucleus Enhanced Pipeline - Simulation Analysis Report",
            f"Generated: {timestamp}",
            f"Total Simulations: {len(integrated_results)}",
            "",
            "## Executive Summary",
            "",
            "This report presents the results of an enhanced PyNucleus pipeline analysis",
            "that integrates DWSIM chemical process simulations with RAG knowledge base",
            "capabilities for comprehensive process understanding.",
            "",
            "## Simulation Results Summary",
            ""
        ]
        
        for i, result in enumerate(integrated_results, 1):
            sim = result['original_simulation']
            summary_parts.extend([
                f"### Simulation {i}: {sim.get('case_name', 'Unknown')}",
                f"- Process Type: {sim.get('simulation_type', 'Unknown')}",
                f"- Components: {sim.get('components', 'Not specified')}",
                f"- Status: {'Success' if sim.get('success', True) else 'Failed'}",
                f"- Performance: {result['performance_metrics'].get('overall_performance', 'Unknown')}",
                ""
            ])
            
            # Add recommendations
            if result.get('recommendations'):
                summary_parts.append("Recommendations:")
                for rec in result['recommendations']:
                    summary_parts.append(f"- {rec}")
                summary_parts.append("")
        
        summary_parts.extend([
            "## Conclusion",
            "",
            "The enhanced pipeline successfully demonstrates the integration of",
            "chemical process simulation with knowledge base analysis, providing",
            "comprehensive insights for process optimization and decision making.",
            "",
            f"Report generated by PyNucleus Enhanced Pipeline on {timestamp}"
        ])
        
        return "\n".join(summary_parts)
    
    def _export_comprehensive_results(self, results: Dict) -> str:
        """Export comprehensive results to JSON file."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"enhanced_pipeline_results_{timestamp}.json"
        output_file = self.results_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"✅ Comprehensive results exported: {output_file}")
        return str(output_file)
    
    def _calculate_enhanced_statistics(self, results: Dict) -> Dict:
        """Calculate enhanced pipeline statistics."""
        
        return {
            'total_simulations': len(results['dwsim_data']),
            'successful_simulations': sum(1 for r in results['dwsim_data'] if r.get('success', True)),
            'rag_queries_processed': len(results['rag_data']),
            'integrated_results': len(results['integrated_data']),
            'files_exported': len(results['exported_files']),
            'pipeline_success': results.get('success', False),
            'duration_seconds': results.get('duration', 0)
        }
    
    def create_custom_simulation_config(self, name: str, process_type: str, 
                                      components: List[str], description: str = "",
                                      parameters: Dict = None) -> Dict:
        """Create a custom simulation configuration."""
        
        config = {
            'name': name,
            'type': process_type,
            'components': components,
            'description': description,
            'parameters': parameters or {},
            'timestamp': datetime.now().isoformat(),
            'source': 'enhanced_pipeline'
        }
        
        return config
    
    def save_custom_config(self, configurations: List[Dict], 
                          filename: str = None, format: str = 'json') -> str:
        """Save custom configurations to file."""
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"custom_simulations_{timestamp}.{format}"
        
        output_file = self.config_dir / filename
        
        if format == 'json':
            with open(output_file, 'w') as f:
                json.dump({'simulations': configurations}, f, indent=2)
        elif format == 'csv':
            # Implement CSV export if needed
            print("⚠️ CSV export not implemented yet")
            return str(output_file)
        
        print(f"✅ Custom configuration saved: {output_file}")
        return str(output_file)
    
    def get_enhanced_status(self) -> Dict:
        """Get comprehensive status of enhanced pipeline."""
        
        status = {
            'pipeline_type': 'enhanced',
            'timestamp': datetime.now().isoformat(),
            'components': {
                'base_pipeline': self.base_pipeline is not None,
                'config_manager': self.config_manager is not None,
                'dwsim_rag_integrator': self.dwsim_rag_integrator is not None,
                'llm_generator': self.llm_generator is not None
            },
            'directories': {
                'results': str(self.results_dir),
                'config': str(self.config_dir)
            },
            'last_run_results': self.enhanced_results.get('statistics', {}) if self.enhanced_results else {}
        }
        
        return status
    
    def print_enhanced_status(self):
        """Print enhanced pipeline status."""
        
        status = self.get_enhanced_status()
        
        print("\n🚀 Enhanced PyNucleus Pipeline Status")
        print(f"   📅 Status time: {status['timestamp']}")
        print(f"   📁 Results directory: {status['directories']['results']}")
        print(f"   ⚙️ Config directory: {status['directories']['config']}")
        print()
        print("📦 Components Status:")
        for component, available in status['components'].items():
            status_icon = "✅" if available else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
        
        if status['last_run_results']:
            print("\n📊 Last Run Statistics:")
            for key, value in status['last_run_results'].items():
                print(f"   • {key.replace('_', ' ').title()}: {value}")
    
    def demo_enhanced_pipeline(self) -> Dict:
        """Run a demonstration of the enhanced pipeline."""
        
        print("🎯 Running Enhanced Pipeline Demonstration...")
        
        # Create demo configurations
        demo_configs = [
            self.create_custom_simulation_config(
                name="demo_distillation",
                process_type="distillation",
                components=["water", "ethanol"],
                description="Demo ethanol-water separation",
                parameters={"temperature": 78.4, "pressure": 101325}
            ),
            self.create_custom_simulation_config(
                name="demo_reactor",
                process_type="reactor", 
                components=["methane", "oxygen"],
                description="Demo methane combustion",
                parameters={"temperature": 1000, "pressure": 101325}
            )
        ]
        
        # Save demo config
        config_file = self.save_custom_config(demo_configs, "demo_simulations.json")
        
        # Run enhanced pipeline
        results = self.run_enhanced_pipeline(
            simulation_configs=demo_configs,
            include_rag_analysis=True,
            generate_llm_output=True
        )
        
        print("\n🎉 Demo completed! Check the results directory for outputs.")
        return results 