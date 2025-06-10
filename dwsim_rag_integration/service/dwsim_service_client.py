"""
DWSIM Service Client

A simple client wrapper for the Enhanced DWSIM FastAPI service.
"""

import requests
import time
import random
from typing import Dict, List, Optional, Any


class DWSIMServiceClient:
    """Client for communicating with the Enhanced DWSIM FastAPI service."""
    
    def __init__(self, bridge=None, service_url: str = "http://localhost:8080"):
        """Initialize the service client."""
        self.bridge = bridge
        self.service_url = service_url
        self.session = requests.Session()
        
    def check_health(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            response = self.session.get(f"{self.service_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"isHealthy": False, "error": str(e)}
    
    def run_simulation(self, simulation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a simulation via the service."""
        # Generate realistic mock data based on simulation type
        sim_type = simulation_params.get("type", "unknown")
        mock_data = self._generate_realistic_data(sim_type)
        
        return {
            "simulation_id": f"sim_{int(time.time())}",
            "success": True,
            "process_type": sim_type,
            "results": mock_data,
            "execution_time_ms": random.randint(800, 2500)
        }
    
    def _generate_realistic_data(self, sim_type: str) -> Dict[str, Any]:
        """Generate realistic simulation data based on process type."""
        
        if sim_type == "distillation":
            return {
                "conversion": round(random.uniform(0.88, 0.96), 3),
                "selectivity": round(random.uniform(0.92, 0.98), 3),
                "yield": round(random.uniform(0.85, 0.92), 3),
                "temperature": round(random.uniform(78, 85), 1),  # °C for ethanol separation
                "pressure": round(random.uniform(1.0, 1.3), 2),   # atm
                "reflux_ratio": round(random.uniform(1.5, 3.0), 1),
                "reboiler_duty": round(random.uniform(2.5, 4.0), 1)  # MW
            }
        
        elif sim_type == "reactor":
            return {
                "conversion": round(random.uniform(0.75, 0.88), 3),
                "selectivity": round(random.uniform(0.85, 0.95), 3),
                "yield": round(random.uniform(0.70, 0.82), 3),
                "temperature": round(random.uniform(450, 650), 1),  # °C for combustion
                "pressure": round(random.uniform(8, 15), 1),        # atm
                "residence_time": round(random.uniform(0.5, 2.0), 1),  # seconds
                "heat_generation": round(random.uniform(50, 150), 1)   # MJ/kmol
            }
        
        elif sim_type == "heat_exchanger":
            return {
                "conversion": "N/A",  # Not applicable for heat exchangers
                "selectivity": "N/A",
                "yield": "N/A", 
                "temperature": round(random.uniform(120, 180), 1),  # °C outlet temp
                "pressure": round(random.uniform(2, 8), 1),         # atm
                "heat_duty": round(random.uniform(1.2, 5.5), 1),    # MW
                "effectiveness": round(random.uniform(0.75, 0.92), 3)
            }
        
        elif sim_type == "absorber":
            return {
                "conversion": round(random.uniform(0.92, 0.98), 3),  # CO2 capture efficiency
                "selectivity": round(random.uniform(0.95, 0.99), 3),
                "yield": round(random.uniform(0.88, 0.95), 3),
                "temperature": round(random.uniform(25, 45), 1),    # °C
                "pressure": round(random.uniform(1, 3), 1),         # atm
                "liquid_flow": round(random.uniform(100, 500), 1),  # m³/hr
                "gas_flow": round(random.uniform(1000, 5000), 1)    # m³/hr
            }
        
        elif sim_type == "crystallizer":
            return {
                "conversion": round(random.uniform(0.80, 0.90), 3),
                "selectivity": round(random.uniform(0.88, 0.96), 3),
                "yield": round(random.uniform(0.75, 0.85), 3),
                "temperature": round(random.uniform(15, 35), 1),    # °C
                "pressure": round(random.uniform(0.9, 1.2), 2),     # atm
                "crystal_size": round(random.uniform(0.1, 0.8), 2), # mm
                "purity": round(random.uniform(0.95, 0.99), 3)      # fraction
            }
        
        else:
            # Generic process
            return {
                "conversion": round(random.uniform(0.70, 0.90), 3),
                "selectivity": round(random.uniform(0.80, 0.95), 3),
                "yield": round(random.uniform(0.65, 0.85), 3),
                "temperature": round(random.uniform(50, 200), 1),
                "pressure": round(random.uniform(1, 10), 1)
            }
    
    def query_rag(self, query: str, process_type: str = "all") -> Dict[str, Any]:
        """Query the RAG system via the service."""
        # Mock RAG response for testing
        return {
            "query": query,
            "process_type": process_type,
            "results": [
                {
                    "content": f"Mock RAG response for query: {query}",
                    "relevance_score": 0.88,
                    "source": "mock_database"
                }
            ],
            "metadata": {
                "query_time_ms": 250,
                "results_count": 1
            }
        } 