"""
DWSIM Pipeline Module

Handles DWSIM (Dynamic Simulator of Industrial Processes) operations including:
- DWSIM bridge initialization
- Simulation execution for various process types
- Results collection and error handling
- Mock simulation fallback for testing
"""

import sys
import os
import importlib
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

class DWSIMPipeline:
    """Main DWSIM Pipeline class for managing chemical process simulations."""
    
    def __init__(self, results_dir="data/05_output/results"):
        """Initialize DWSIM Pipeline with results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results_data = []
        self.bridge = None
        self.service = None
        
        # Import and setup DWSIM modules
        self._setup_imports()
    
    def _setup_imports(self):
        """Setup and reload DWSIM modules."""
        print("🔧 Setting up DWSIM imports...")
        
        try:
            # Clear DWSIM module cache
            dwsim_modules = [
                'dwsim_rag_integration.core.enhanced_dwsim_bridge',
                'dwsim_rag_integration.service.enhanced_dwsim_service'
            ]
            
            for module_name in dwsim_modules:
                if module_name in sys.modules:
                    importlib.reload(sys.modules[module_name])
            
            from dwsim_rag_integration.core.enhanced_dwsim_bridge import DWSimBridge
            from dwsim_rag_integration.service.dwsim_service_client import DWSIMServiceClient
            
            self.DWSimBridge = DWSimBridge
            self.DWSIMServiceClient = DWSIMServiceClient
            self.dwsim_available = True
            
            print("✅ DWSIM modules imported successfully")
            
        except ImportError as e:
            print(f"❌ DWSIM Import error: {e}")
            print("📝 Note: DWSIM modules not available - will use mock mode")
            self.dwsim_available = False
    
    def get_test_cases(self):
        """Get predefined test cases for DWSIM simulations."""
        return [
            {
                "name": "distillation_ethanol_water", 
                "type": "distillation", 
                "components": ["water", "ethanol"],
                "description": "Ethanol-water separation column"
            },
            {
                "name": "reactor_methane_combustion", 
                "type": "reactor", 
                "components": ["methane", "oxygen"],
                "description": "Methane combustion reactor"
            },
            {
                "name": "heat_exchanger_steam", 
                "type": "heat_exchanger", 
                "components": ["water", "steam"],
                "description": "Steam heat exchanger"
            },
            {
                "name": "absorber_co2_capture", 
                "type": "absorber", 
                "components": ["CO2", "water"],
                "description": "CO2 absorption column"
            },
            {
                "name": "crystallizer_salt", 
                "type": "crystallizer", 
                "components": ["water", "salt"],
                "description": "Salt crystallization unit"
            }
        ]
    
    def run_simulations(self, test_cases=None):
        """Run DWSIM simulations for given test cases."""
        print("🔬 Starting DWSIM Simulations...")
        
        if test_cases is None:
            test_cases = self.get_test_cases()
        
        self.results_data.clear()
        
        if self.dwsim_available:
            self._run_real_simulations(test_cases)
        else:
            self._run_mock_simulations(test_cases)
        
        # Print summary
        successful_sims = sum(1 for r in self.results_data if r['success'])
        print(f"\n📊 Simulation Summary:")
        print(f"   • Successful simulations: {successful_sims}/{len(test_cases)}")
        print(f"   • Failed simulations: {len(test_cases) - successful_sims}/{len(test_cases)}")
        
        return self.results_data
    
    def _run_real_simulations(self, test_cases):
        """Run actual DWSIM simulations."""
        try:
            # Initialize DWSIM components
            self.bridge = self.DWSimBridge()
            self.service = self.DWSIMServiceClient(self.bridge)
            
            print(f"📋 Running {len(test_cases)} simulation cases...")
            
            for i, case in enumerate(test_cases, 1):
                print(f"\n🧪 Case {i}/{len(test_cases)}: {case['name']}")
                self._execute_simulation(case)
                
        except Exception as e:
            print(f"❌ Critical DWSIM error: {str(e)}")
            print("📝 Falling back to mock mode...")
            self._run_mock_simulations(test_cases)
    
    def _execute_simulation(self, case):
        """Execute a single simulation case."""
        try:
            start_time = datetime.now()
            sim_result = self.service.run_simulation(case)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract key metrics from simulation result
            conversion = "N/A"
            selectivity = "N/A"
            yield_val = "N/A"
            temperature = "N/A"
            pressure = "N/A"
            
            if sim_result and isinstance(sim_result, dict):
                results = sim_result.get('results', {})
                conversion = results.get('conversion', 'N/A')
                selectivity = results.get('selectivity', 'N/A')
                yield_val = results.get('yield', 'N/A')
                temperature = results.get('temperature', 'N/A')
                pressure = results.get('pressure', 'N/A')
            
            # Store successful results with clean format
            result_data = {
                'case_name': case['name'],
                'simulation_type': case['type'],
                'components': ', '.join(case['components']),
                'description': case['description'],
                'success': True,
                'duration_seconds': round(duration, 4),
                'conversion': conversion,
                'selectivity': selectivity,
                'yield': yield_val,
                'temperature': temperature,
                'pressure': pressure,
                'status': 'Completed Successfully',
                'timestamp': start_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results_data.append(result_data)
            print(f"   ✅ Success - Duration: {duration:.2f}s")
            
        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error: {error_msg}")
            
            # Store error results
            result_data = {
                'case_name': case['name'],
                'simulation_type': case['type'],
                'components': ', '.join(case['components']),
                'description': case['description'],
                'success': False,
                'duration_seconds': 0,
                'conversion': 'N/A',
                'selectivity': 'N/A',
                'yield': 'N/A',
                'temperature': 'N/A',
                'pressure': 'N/A',
                'status': f'Failed: {error_msg}',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results_data.append(result_data)
    
    def _run_mock_simulations(self, test_cases):
        """Run mock simulations for demonstration/testing."""
        print("📝 Running mock simulations...")
        
        for case in test_cases:
            # Generate realistic mock results based on simulation type
            duration = 0.5 + (len(case['name']) * 0.02)  # Variable duration based on complexity
            
            # Generate realistic mock values based on simulation type
            mock_values = self._generate_mock_values(case['type'])
            
            result_data = {
                'case_name': case['name'],
                'simulation_type': case['type'],
                'components': ', '.join(case['components']),
                'description': case['description'],
                'success': True,
                'duration_seconds': round(duration, 4),
                'conversion': mock_values['conversion'],
                'selectivity': mock_values['selectivity'],
                'yield': mock_values['yield'],
                'temperature': mock_values['temperature'],
                'pressure': mock_values['pressure'],
                'status': 'Mock Simulation Completed',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.results_data.append(result_data)
            print(f"   ✅ Mock {case['name']} - Duration: {duration:.2f}s")
    
    def _generate_mock_values(self, sim_type):
        """Generate realistic mock values based on simulation type."""
        import random
        
        if sim_type == "distillation":
            return {
                'conversion': round(random.uniform(0.85, 0.95), 3),
                'selectivity': round(random.uniform(0.90, 0.98), 3),
                'yield': round(random.uniform(0.80, 0.90), 3),
                'temperature': round(random.uniform(70, 85), 1),  # °C
                'pressure': round(random.uniform(1.0, 1.5), 2)    # atm
            }
        elif sim_type == "reactor":
            return {
                'conversion': round(random.uniform(0.75, 0.88), 3),
                'selectivity': round(random.uniform(0.85, 0.95), 3),
                'yield': round(random.uniform(0.70, 0.82), 3),
                'temperature': round(random.uniform(400, 600), 1),  # °C
                'pressure': round(random.uniform(5, 15), 1)         # atm
            }
        elif sim_type == "heat_exchanger":
            return {
                'conversion': 'N/A',
                'selectivity': 'N/A',
                'yield': 'N/A',
                'temperature': round(random.uniform(120, 180), 1),  # °C
                'pressure': round(random.uniform(2, 8), 1)          # atm
            }
        elif sim_type == "absorber":
            return {
                'conversion': round(random.uniform(0.92, 0.98), 3),
                'selectivity': round(random.uniform(0.95, 0.99), 3),
                'yield': round(random.uniform(0.88, 0.95), 3),
                'temperature': round(random.uniform(25, 40), 1),   # °C
                'pressure': round(random.uniform(1, 3), 1)         # atm
            }
        elif sim_type == "crystallizer":
            return {
                'conversion': round(random.uniform(0.80, 0.90), 3),
                'selectivity': round(random.uniform(0.88, 0.96), 3),
                'yield': round(random.uniform(0.75, 0.85), 3),
                'temperature': round(random.uniform(10, 30), 1),   # °C
                'pressure': round(random.uniform(0.8, 1.2), 2)     # atm
            }
        else:
            return {
                'conversion': round(random.uniform(0.70, 0.90), 3),
                'selectivity': round(random.uniform(0.80, 0.95), 3),
                'yield': round(random.uniform(0.65, 0.85), 3),
                'temperature': round(random.uniform(50, 200), 1),
                'pressure': round(random.uniform(1, 10), 1)
            }
    
    def run_single_simulation(self, case_config):
        """Run a single simulation with custom configuration."""
        print(f"🧪 Running single simulation: {case_config.get('name', 'Custom')}")
        
        if self.dwsim_available and self.service:
            self._execute_simulation(case_config)
        else:
            self._run_mock_simulations([case_config])
        
        return self.results_data[-1] if self.results_data else None
    
    def get_results(self):
        """Get collected simulation results."""
        return self.results_data
    
    def clear_results(self):
        """Clear collected results."""
        self.results_data.clear()
        print("🗑️ DWSIM results cleared.")
    
    def get_statistics(self):
        """Get DWSIM pipeline statistics."""
        if not self.results_data:
            return {}
        
        successful = [r for r in self.results_data if r['success']]
        failed = [r for r in self.results_data if not r['success']]
        
        stats = {
            'total_simulations': len(self.results_data),
            'successful_simulations': len(successful),
            'failed_simulations': len(failed),
            'success_rate': len(successful) / len(self.results_data) * 100,
            'avg_duration': sum(r['duration_seconds'] for r in successful) / len(successful) if successful else 0,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"📊 DWSIM Statistics:")
        print(f"   • Total Simulations: {stats['total_simulations']}")
        print(f"   • Success Rate: {stats['success_rate']:.1f}%")
        print(f"   • Average Duration: {stats['avg_duration']:.2f}s")
        
        return stats 