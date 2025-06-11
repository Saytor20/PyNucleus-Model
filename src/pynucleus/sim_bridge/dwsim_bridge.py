"""
Enhanced DWSIM Bridge for PyNucleus-Model

Provides advanced chemical engineering simulation capabilities with:
- Material and energy balance calculations
- Economic optimization and sensitivity analysis
- RAG integration for intelligent modular plant design
- Real-time process optimization
"""

import os
import json
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
import numpy as np
from dataclasses import dataclass, asdict

# Check if running in Docker environment
IS_DOCKER = os.path.exists("/.dockerenv") or "DOCKER_CONTAINER" in os.environ

# Base URL for DWSIM service
BASE_URL = "http://dwsim-service:8080" if IS_DOCKER else "http://localhost:8080"


@dataclass
class ProcessConditions:
    """Process stream conditions."""
    mass_flow: float  # kg/hr
    temperature: float  # K
    pressure: float  # kPa
    composition: Dict[str, float]  # Component mole fractions


@dataclass
class EconomicParameters:
    """Economic analysis parameters."""
    product_price: float  # $/tonne
    raw_material_cost: float  # $/tonne
    utility_cost: float  # $/kWh
    operating_hours: float  # hours/year
    discount_rate: float  # fraction


@dataclass
class OptimizationResult:
    """Process optimization results."""
    optimal_parameters: Dict[str, float]
    objective_value: float
    economic_analysis: Dict[str, Any]
    design_recommendations: List[str]
    sensitivity_analysis: Optional[Dict[str, Any]] = None


@dataclass
class SimulationResult:
    """Enhanced simulation results."""
    simulation_id: str
    success: bool
    process_type: str
    feed_analysis: Dict[str, Any]
    product_streams: Dict[str, Any]
    energy_analysis: Dict[str, Any]
    economic_analysis: Dict[str, Any]
    modular_design: Dict[str, Any]
    execution_time_ms: int
    csv_path: Optional[str] = None
    sensitivity_results: Optional[Dict[str, Any]] = None
    rag_metadata: Optional[Dict[str, Any]] = None


class DWSimBridge:
    """Enhanced DWSIM Bridge with RAG integration and advanced analytics."""
    
    def __init__(self, service_url: str = None):
        """Initialize the enhanced DWSIM bridge."""
        self.service_url = service_url or BASE_URL
        self.session = requests.Session()
        self.simulation_cache: Dict[str, SimulationResult] = {}
        self.optimization_cache: Dict[str, OptimizationResult] = {}
        
    def check_health(self) -> Dict[str, Any]:
        """Check DWSIM service health with enhanced capabilities."""
        try:
            # Check basic health
            response = self.session.get(f"{self.service_url}/health", timeout=10)
            basic_health = response.json()
            
            # Check enhanced API health
            api_response = self.session.get(f"{self.service_url}/api/simulation/health", timeout=10)
            api_health = api_response.json()
            
            return {
                "isHealthy": basic_health.get("status") == "healthy" and api_health.get("isHealthy", False),
                "basicService": basic_health,
                "enhancedFeatures": api_health.get("enhancedFeatures", {}),
                "serviceUrl": self.service_url,
                "capabilities": [
                    "Chemical Engineering Calculations",
                    "Material & Energy Balances", 
                    "Economic Optimization",
                    "Sensitivity Analysis",
                    "RAG Integration",
                    "Modular Plant Design"
                ]
            }
        except Exception as e:
            return {
                "isHealthy": False,
                "error": str(e),
                "serviceUrl": self.service_url
            }
    
    def run_chemical_simulation(
        self,
        process_type: str = "methanol_synthesis",
        feed_conditions: ProcessConditions = None,
        economic_params: EconomicParameters = None,
        equipment_specs: Dict[str, Any] = None,
        run_sensitivity: bool = False,
        optimization_mode: str = "standard"
    ) -> SimulationResult:
        """Run enhanced chemical engineering simulation."""
        
        # Create simulation file
        sim_file_content = self._create_simulation_file(
            process_type, feed_conditions, economic_params, equipment_specs
        )
        
        try:
            # Prepare multipart form data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.dwsim', delete=False) as f:
                f.write(sim_file_content)
                temp_file_path = f.name
            
            with open(temp_file_path, 'rb') as f:
                files = {
                    'DwsimFile': ('simulation.dwsim', f, 'text/plain')
                }
                data = {
                    'ExportCsv': 'true',
                    'OptimizationMode': optimization_mode,
                    'SensitivityAnalysis': str(run_sensitivity).lower()
                }
                
                response = self.session.post(
                    f"{self.service_url}/api/simulation/run",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            # Clean up temp file
            os.unlink(temp_file_path)
            
            if response.status_code == 200:
                result_data = response.json()
                
                # Parse result into dataclass
                simulation_result = SimulationResult(
                    simulation_id=result_data.get("simulationId"),
                    success=result_data.get("success", False),
                    process_type=result_data.get("processType", process_type),
                    feed_analysis=result_data.get("results", {}).get("feed_analysis", {}),
                    product_streams=result_data.get("results", {}).get("product_streams", {}),
                    energy_analysis=result_data.get("results", {}).get("energy_analysis", {}),
                    economic_analysis=result_data.get("results", {}).get("economic_analysis", {}),
                    modular_design=result_data.get("results", {}).get("modular_design", {}),
                    execution_time_ms=result_data.get("executionTimeMs", 0),
                    csv_path=result_data.get("csvDataPath"),
                    sensitivity_results=result_data.get("sensitivityResults"),
                    rag_metadata=result_data.get("ragMetadata", {})
                )
                
                # Cache the result
                self.simulation_cache[simulation_result.simulation_id] = simulation_result
                
                return simulation_result
            else:
                raise Exception(f"Simulation failed: {response.text}")
                
        except Exception as e:
            raise Exception(f"Enhanced simulation error: {str(e)}")
    
    def optimize_process(
        self,
        process_type: str = "methanol_synthesis",
        objective: str = "profit_maximization",
        constraints: List[str] = None
    ) -> OptimizationResult:
        """Run process optimization for modular plant design."""
        
        if constraints is None:
            constraints = ["safety", "environmental"]
        
        try:
            params = {
                "process_type": process_type,
                "objective": objective,
                "constraints": ",".join(constraints)
            }
            
            response = self.session.get(
                f"{self.service_url}/api/simulation/optimize",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                result_data = response.json()
                
                optimization_result = OptimizationResult(
                    optimal_parameters=result_data.get("optimalParameters", {}),
                    objective_value=result_data.get("objectiveValue", 0.0),
                    economic_analysis=result_data.get("economicAnalysis", {}),
                    design_recommendations=result_data.get("designRecommendations", [])
                )
                
                # Cache the result
                optimization_id = result_data.get("optimizationId")
                if optimization_id:
                    self.optimization_cache[optimization_id] = optimization_result
                
                return optimization_result
            else:
                raise Exception(f"Optimization failed: {response.text}")
                
        except Exception as e:
            raise Exception(f"Process optimization error: {str(e)}")
    
    def query_rag_system(
        self,
        query: str,
        process_type: str = "all",
        run_simulation: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG system for process design insights."""
        
        try:
            params = {
                "query": query,
                "process_type": process_type
            }
            
            response = self.session.get(
                f"{self.service_url}/api/simulation/rag/query",
                params=params,
                timeout=45
            )
            
            if response.status_code == 200:
                rag_response = response.json()
                
                # If simulation was triggered, cache the results
                if rag_response.get("requiresSimulation") and rag_response.get("simulationResults"):
                    sim_results = rag_response["simulationResults"]
                    
                    # Create pseudo simulation result for caching
                    simulation_result = SimulationResult(
                        simulation_id=f"rag_{int(time.time())}",
                        success=True,
                        process_type=process_type,
                        feed_analysis=sim_results.get("feed_analysis", {}),
                        product_streams=sim_results.get("product_streams", {}),
                        energy_analysis=sim_results.get("energy_analysis", {}),
                        economic_analysis=sim_results.get("economic_analysis", {}),
                        modular_design=sim_results.get("modular_design", {}),
                        execution_time_ms=0,
                        rag_metadata={"query": query, "triggered_by_rag": True}
                    )
                    
                    self.simulation_cache[simulation_result.simulation_id] = simulation_result
                
                return rag_response
            else:
                raise Exception(f"RAG query failed: {response.text}")
                
            except Exception as e:
            raise Exception(f"RAG system error: {str(e)}")
    
    def run_comprehensive_analysis(
        self,
        process_type: str = "methanol_synthesis",
        feed_conditions: ProcessConditions = None,
        economic_params: EconomicParameters = None,
        optimization_objectives: List[str] = None
    ) -> Dict[str, Any]:
        """Run comprehensive process analysis including simulation, optimization, and RAG insights."""
        
        if optimization_objectives is None:
            optimization_objectives = ["profit_maximization", "energy_efficiency"]
        
        comprehensive_results = {
            "process_type": process_type,
            "timestamp": time.time(),
            "base_simulation": None,
            "sensitivity_analysis": None,
            "optimizations": {},
            "rag_insights": {},
            "recommendations": []
        }
        
        try:
            # 1. Run base simulation with sensitivity analysis
            base_simulation = self.run_chemical_simulation(
                process_type=process_type,
                feed_conditions=feed_conditions,
                economic_params=economic_params,
                run_sensitivity=True,
                optimization_mode="comprehensive"
            )
            comprehensive_results["base_simulation"] = asdict(base_simulation)
            comprehensive_results["sensitivity_analysis"] = base_simulation.sensitivity_results
            
            # 2. Run optimizations for different objectives
            for objective in optimization_objectives:
                try:
                    optimization_result = self.optimize_process(
                        process_type=process_type,
                        objective=objective,
                        constraints=["safety", "environmental", "economic"]
                    )
                    comprehensive_results["optimizations"][objective] = asdict(optimization_result)
                except Exception as e:
                    comprehensive_results["optimizations"][objective] = {"error": str(e)}
            
            # 3. Query RAG system for design insights
            rag_queries = [
                f"What are the optimal operating conditions for {process_type}?",
                f"How can I improve the profitability of a {process_type} plant?",
                f"What are the key design considerations for modular {process_type} plants?",
                "What sensitivity analysis should I consider for process optimization?"
            ]
            
            for query in rag_queries:
                try:
                    rag_response = self.query_rag_system(query, process_type, run_simulation=False)
                    comprehensive_results["rag_insights"][query] = rag_response
                except Exception as e:
                    comprehensive_results["rag_insights"][query] = {"error": str(e)}
            
            # 4. Generate comprehensive recommendations
            comprehensive_results["recommendations"] = self._generate_comprehensive_recommendations(
                base_simulation, comprehensive_results["optimizations"], comprehensive_results["rag_insights"]
            )
            
            return comprehensive_results

        except Exception as e:
            comprehensive_results["error"] = str(e)
            return comprehensive_results
    
    def calculate_profitability_threshold(
        self,
        process_type: str = "methanol_synthesis",
        market_conditions: Dict[str, float] = None,
        capacity_range: Tuple[float, float] = (5000, 50000)  # tonnes/year
    ) -> Dict[str, Any]:
        """Calculate minimum flow rates and conditions for profitability."""
        
        if market_conditions is None:
            market_conditions = {
                "product_price": 400.0,  # $/tonne
                "raw_material_cost": 200.0,  # $/tonne
                "energy_cost": 0.12,  # $/kWh
                "labor_cost": 50000.0,  # $/year
                "maintenance_factor": 0.05  # fraction of CAPEX
            }
        
        profitability_analysis = {
            "process_type": process_type,
            "market_conditions": market_conditions,
            "capacity_analysis": [],
            "breakeven_analysis": {},
            "sensitivity_to_market": {},
            "recommendations": []
        }
        
        try:
            # Analyze different plant capacities
            min_capacity, max_capacity = capacity_range
            capacity_steps = np.linspace(min_capacity, max_capacity, 10)
            
            for capacity in capacity_steps:
                # Calculate equivalent mass flow (kg/hr)
                operating_hours = 8400  # hours/year
                mass_flow = capacity * 1000 / operating_hours  # kg/hr
                
                # Create feed conditions for this capacity
                feed_conditions = ProcessConditions(
                    mass_flow=mass_flow,
                    temperature=523.15,  # K
                    pressure=5000.0,  # kPa
                    composition={"CO": 0.25, "H2": 0.65, "CO2": 0.08, "N2": 0.02}
                )
                
                # Create economic parameters
                economic_params = EconomicParameters(
                    product_price=market_conditions["product_price"],
                    raw_material_cost=market_conditions["raw_material_cost"],
                    utility_cost=market_conditions["energy_cost"],
                    operating_hours=operating_hours,
                    discount_rate=0.10
                )
                
                # Run simulation
                try:
                    simulation = self.run_chemical_simulation(
                        process_type=process_type,
                        feed_conditions=feed_conditions,
                        economic_params=economic_params,
                        optimization_mode="economic"
                    )
                    
                    # Extract economic metrics
                    econ_analysis = simulation.economic_analysis
                    
                    capacity_analysis = {
                        "capacity_tonnes_year": capacity,
                        "mass_flow_kg_hr": mass_flow,
                        "annual_production": econ_analysis.get("annual_production", 0),
                        "annual_revenue": econ_analysis.get("annual_revenue", 0),
                        "annual_cost": econ_analysis.get("annual_operating_cost", 0),
                        "annual_profit": econ_analysis.get("annual_profit", 0),
                        "profit_margin": econ_analysis.get("profit_margin", 0),
                        "production_cost": econ_analysis.get("production_cost", 0),
                        "payback_period": econ_analysis.get("payback_period", 0),
                        "npv_10_years": econ_analysis.get("npv_10_years", 0),
                        "is_profitable": econ_analysis.get("annual_profit", 0) > 0
                    }
                    
                    profitability_analysis["capacity_analysis"].append(capacity_analysis)

        except Exception as e:
                    profitability_analysis["capacity_analysis"].append({
                        "capacity_tonnes_year": capacity,
                        "error": str(e)
                    })
            
            # Find breakeven point
            profitable_cases = [
                case for case in profitability_analysis["capacity_analysis"] 
                if case.get("is_profitable", False)
            ]
            
            if profitable_cases:
                min_profitable = min(profitable_cases, key=lambda x: x["capacity_tonnes_year"])
                profitability_analysis["breakeven_analysis"] = {
                    "minimum_profitable_capacity": min_profitable["capacity_tonnes_year"],
                    "minimum_flow_rate": min_profitable["mass_flow_kg_hr"],
                    "breakeven_production_cost": min_profitable["production_cost"],
                    "minimum_profit_margin": min_profitable["profit_margin"]
                }
            else:
                profitability_analysis["breakeven_analysis"] = {
                    "no_profitable_capacity_found": True,
                    "market_conditions_unfavorable": True
                }
            
            # Market sensitivity analysis
            price_variations = [-20, -10, 0, 10, 20]  # % changes
            for price_change in price_variations:
                adjusted_price = market_conditions["product_price"] * (1 + price_change/100)
                
                # Use mid-range capacity for sensitivity
                mid_capacity = (min_capacity + max_capacity) / 2
                mid_flow = mid_capacity * 1000 / 8400
                
                feed_conditions = ProcessConditions(
                    mass_flow=mid_flow,
                    temperature=523.15,
                    pressure=5000.0,
                    composition={"CO": 0.25, "H2": 0.65, "CO2": 0.08, "N2": 0.02}
                )
                
                economic_params = EconomicParameters(
                    product_price=adjusted_price,
                    raw_material_cost=market_conditions["raw_material_cost"],
                    utility_cost=market_conditions["energy_cost"],
                    operating_hours=8400,
                    discount_rate=0.10
            )

        try:
                    simulation = self.run_chemical_simulation(
                        process_type=process_type,
                        feed_conditions=feed_conditions,
                        economic_params=economic_params,
                        optimization_mode="economic"
                    )
                    
                    profitability_analysis["sensitivity_to_market"][f"price_{price_change}pct"] = {
                        "product_price": adjusted_price,
                        "annual_profit": simulation.economic_analysis.get("annual_profit", 0),
                        "profit_margin": simulation.economic_analysis.get("profit_margin", 0)
                    }
                except Exception as e:
                    profitability_analysis["sensitivity_to_market"][f"price_{price_change}pct"] = {
                        "error": str(e)
                    }
            
            # Generate recommendations
            profitability_analysis["recommendations"] = self._generate_profitability_recommendations(
                profitability_analysis
            )
            
            return profitability_analysis

        except Exception as e:
            profitability_analysis["error"] = str(e)
            return profitability_analysis
    
    def get_simulation_results(self, simulation_id: str) -> Optional[SimulationResult]:
        """Get cached simulation results."""
        return self.simulation_cache.get(simulation_id)
    
    def get_optimization_results(self, optimization_id: str) -> Optional[OptimizationResult]:
        """Get cached optimization results."""
        return self.optimization_cache.get(optimization_id)
    
    def export_results_csv(self, simulation_id: str, output_path: str = None) -> str:
        """Export simulation results to CSV file."""
        try:
            response = self.session.get(
                f"{self.service_url}/api/simulation/results/{simulation_id}/csv",
                timeout=30
            )
            
            if response.status_code == 200:
                if output_path is None:
                    output_path = f"simulation_{simulation_id}_results.csv"
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                return output_path
            else:
                raise Exception(f"Failed to export CSV: {response.text}")
                
        except Exception as e:
            raise Exception(f"CSV export error: {str(e)}")
    
    def cleanup_simulation(self, simulation_id: str) -> bool:
        """Clean up simulation files and cache."""
        try:
            # Remove from service
            response = self.session.delete(
                f"{self.service_url}/api/simulation/results/{simulation_id}",
                timeout=10
            )
            
            # Remove from local cache
            if simulation_id in self.simulation_cache:
                del self.simulation_cache[simulation_id]
            
            return response.status_code == 200

        except Exception as e:
            print(f"Cleanup error: {str(e)}")
            return False

    def _create_simulation_file(
        self,
        process_type: str,
        feed_conditions: ProcessConditions = None,
        economic_params: EconomicParameters = None,
        equipment_specs: Dict[str, Any] = None
    ) -> str:
        """Create enhanced simulation file content."""
        
        # Default feed conditions
        if feed_conditions is None:
            feed_conditions = ProcessConditions(
                mass_flow=1000.0,
                temperature=523.15,
                pressure=5000.0,
                composition={"CO": 0.25, "H2": 0.65, "CO2": 0.08, "N2": 0.02}
            )
        
        # Default economic parameters
        if economic_params is None:
            economic_params = EconomicParameters(
                product_price=400.0,
                raw_material_cost=200.0,
                utility_cost=0.12,
                operating_hours=8400.0,
                discount_rate=0.10
            )
        
        # Generate simulation file content
        sim_content = f"""# Enhanced DWSIM Process Simulation
# Generated by PyNucleus-Model Enhanced DWSIM Bridge
# Process Type: {process_type}

simulation_name: "Enhanced {process_type.replace('_', ' ').title()} Process"
created_by: "PyNucleus-Model Enhanced System"
version: "2.0"
process_type: "{process_type}"

# Feed Stream Specifications
feed_streams:
  - name: "primary_feed"
    components:"""
        
        for component, fraction in feed_conditions.composition.items():
            sim_content += f"""
      - name: "{component}"
        mole_fraction: {fraction}  # mol/mol"""
        
        sim_content += f"""
    conditions:
      mass_flow: {feed_conditions.mass_flow}     # kg/hr
      temperature: {feed_conditions.temperature}   # K
      pressure: {feed_conditions.pressure}       # kPa

# Economic Parameters
economics:
  operating_hours: {economic_params.operating_hours}   # hours/year
  product_prices:
    primary_product: {economic_params.product_price}    # $/tonne
  raw_material_costs:
    feedstock: {economic_params.raw_material_cost}      # $/tonne
  utility_costs:
    electricity: {economic_params.utility_cost}         # $/kWh

# Process Equipment (Default)
equipment:
  - name: "reactor_R101"
    type: "catalytic_reactor"
    specifications:
      volume: 5.0           # m³
      temperature: {feed_conditions.temperature}   # K
      pressure: {feed_conditions.pressure}         # kPa
      
  - name: "separator_V101"
    type: "flash_separator"
    specifications:
      temperature: 313.15   # K
      pressure: 300.0       # kPa
      efficiency: 0.95      # dimensionless

# Sensitivity Analysis Parameters
sensitivity_analysis:
  variables:
    - name: "reactor_temperature"
      range: [{feed_conditions.temperature - 30}, {feed_conditions.temperature + 30}]  # K
      steps: 5
    - name: "reactor_pressure"
      range: [{feed_conditions.pressure * 0.8}, {feed_conditions.pressure * 1.2}]  # kPa
      steps: 5
    - name: "feed_flow"
      range: [{feed_conditions.mass_flow * 0.8}, {feed_conditions.mass_flow * 1.2}]  # kg/hr
      steps: 5

# RAG Integration Metadata
rag_metadata:
  process_keywords: ["{process_type}", "simulation", "optimization", "modular"]
  generated_by: "enhanced_dwsim_bridge"
  timestamp: {time.time()}
"""
        
        return sim_content
    
    def _generate_comprehensive_recommendations(
        self,
        base_simulation: SimulationResult,
        optimizations: Dict[str, Any],
        rag_insights: Dict[str, Any]
    ) -> List[str]:
        """Generate comprehensive process recommendations."""
        
        recommendations = []
        
        # Base simulation insights
        if base_simulation.success:
            econ = base_simulation.economic_analysis
            energy = base_simulation.energy_analysis
            
            if econ.get("profit_margin", 0) < 0.15:
                recommendations.append(
                    f"⚠️  Profit margin ({econ.get('profit_margin', 0):.1%}) below target 15%. "
                    "Consider process optimization or market analysis."
                )
            
            if energy.get("energy_efficiency", 0) < 50:
                recommendations.append(
                    f"🔋 Energy efficiency ({energy.get('energy_efficiency', 0):.1f} kg/kW) can be improved. "
                    "Implement heat integration or process intensification."
                )
        
        # Optimization insights
        for objective, opt_result in optimizations.items():
            if not opt_result.get("error"):
                recommendations.extend([
                    f"✅ {objective.replace('_', ' ').title()}: {rec}"
                    for rec in opt_result.get("design_recommendations", [])
                ])
        
        # RAG insights
        for query, response in rag_insights.items():
            if not response.get("error") and response.get("design_recommendations"):
                recommendations.extend([
                    f"💡 RAG Insight: {rec}"
                    for rec in response["design_recommendations"]
                ])
        
        # Add general modular design recommendations
        recommendations.extend([
            "🏗️  Design for modular assembly with ISO container compatibility",
            "🔄 Implement recycle streams to improve conversion and reduce waste",
            "📊 Monitor key performance indicators: conversion, selectivity, energy efficiency",
            "💰 Conduct regular economic updates based on market fluctuations"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _generate_profitability_recommendations(
        self, profitability_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate profitability-focused recommendations."""
        
        recommendations = []
        
        breakeven = profitability_analysis.get("breakeven_analysis", {})
        
        if breakeven.get("no_profitable_capacity_found"):
            recommendations.extend([
                "❌ No profitable capacity found under current market conditions",
                "📈 Consider waiting for better market conditions or technology improvements",
                "🔍 Investigate alternative process routes or co-product opportunities",
                "💡 Consider government incentives or carbon credit opportunities"
            ])
        else:
            min_capacity = breakeven.get("minimum_profitable_capacity", 0)
            min_flow = breakeven.get("minimum_flow_rate", 0)
            
            recommendations.extend([
                f"✅ Minimum profitable capacity: {min_capacity:.0f} tonnes/year",
                f"⚡ Minimum feed flow rate: {min_flow:.0f} kg/hr",
                f"💰 Target production cost: ${breakeven.get('breakeven_production_cost', 0):.0f}/tonne",
                "📊 Monitor market prices closely for optimization timing"
            ])
        
        # Market sensitivity insights
        sensitivity = profitability_analysis.get("sensitivity_to_market", {})
        if sensitivity:
            price_impacts = [
                (k, v.get("profit_margin", 0)) for k, v in sensitivity.items()
                if "error" not in v
            ]
            
            if price_impacts:
                max_margin = max(price_impacts, key=lambda x: x[1])
                recommendations.append(
                    f"📈 Best case profit margin: {max_margin[1]:.1%} "
                    f"(at {max_margin[0].replace('price_', '').replace('pct', '% price change')})"
                )
        
        return recommendations


# Convenience functions for external use
def create_default_feed_conditions(process_type: str = "methanol_synthesis") -> ProcessConditions:
    """Create default feed conditions for common processes."""
    
    if process_type == "methanol_synthesis":
        return ProcessConditions(
            mass_flow=1000.0,
            temperature=523.15,  # 250°C
            pressure=5000.0,     # 50 bar
            composition={"CO": 0.25, "H2": 0.65, "CO2": 0.08, "N2": 0.02}
        )
    elif process_type == "ammonia_synthesis":
        return ProcessConditions(
            mass_flow=1200.0,
            temperature=673.15,  # 400°C
            pressure=15000.0,    # 150 bar
            composition={"N2": 0.25, "H2": 0.75}
        )
    else:
        # Generic process
        return ProcessConditions(
            mass_flow=1000.0,
            temperature=473.15,  # 200°C
            pressure=2000.0,     # 20 bar
            composition={"component_a": 0.7, "component_b": 0.3}
        )


def create_default_economic_params(market_scenario: str = "base_case") -> EconomicParameters:
    """Create default economic parameters for different market scenarios."""
    
    scenarios = {
        "base_case": EconomicParameters(
            product_price=400.0,
            raw_material_cost=200.0,
            utility_cost=0.12,
            operating_hours=8400.0,
            discount_rate=0.10
        ),
        "optimistic": EconomicParameters(
            product_price=500.0,
            raw_material_cost=180.0,
            utility_cost=0.10,
            operating_hours=8760.0,
            discount_rate=0.08
        ),
        "conservative": EconomicParameters(
            product_price=350.0,
            raw_material_cost=220.0,
            utility_cost=0.15,
            operating_hours=8000.0,
            discount_rate=0.12
        )
    }
    
    return scenarios.get(market_scenario, scenarios["base_case"])


# Example usage and testing
if __name__ == "__main__":
    # Initialize enhanced bridge
    bridge = DWSimBridge()
    
    # Check health
    health = bridge.check_health()
    print("Enhanced DWSIM Bridge Health:", health)
    
    if health.get("isHealthy"):
        # Run comprehensive analysis
        feed_conditions = create_default_feed_conditions("methanol_synthesis")
        economic_params = create_default_economic_params("base_case")
        
        comprehensive_results = bridge.run_comprehensive_analysis(
            process_type="methanol_synthesis",
            feed_conditions=feed_conditions,
            economic_params=economic_params,
            optimization_objectives=["profit_maximization", "energy_efficiency"]
        )
        
        print("\nComprehensive Analysis Results:")
        print(f"Process Type: {comprehensive_results['process_type']}")
        print(f"Recommendations: {len(comprehensive_results['recommendations'])}")
        
        for i, rec in enumerate(comprehensive_results['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
    else:
        print("Enhanced DWSIM service not available for testing")
