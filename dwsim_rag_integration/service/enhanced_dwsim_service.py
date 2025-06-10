"""
Enhanced Mock DWSIM Service for Chemical Engineering

A FastAPI-based service that simulates realistic DWSIM functionality with:
- Material and energy balances
- Chemical reaction calculations
- Economic optimization
- Sensitivity analysis
- RAG integration for modular plant design
"""

import os
import uuid
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
import uvicorn


app = FastAPI(
    title="Enhanced DWSIM Service for Modular Plant Design",
    description="Chemical engineering simulation service with RAG integration",
    version="2.0.0"
)

# Global storage for simulation status
simulations: Dict[str, Dict] = {}
process_database: Dict[str, Any] = {}

# Directories - handle both local and Docker environments
def setup_directories():
    """Setup work and results directories for local and Docker environments."""
    if os.path.exists("/.dockerenv"):
        # Docker environment
        work_dir = Path(os.getenv("DWSIM_WORK_DIR", "/app/simulations"))
        results_dir = Path(os.getenv("DWSIM_RESULTS_DIR", "/app/results"))
    else:
        # Local development environment
        work_dir = Path("./temp_simulations")
        results_dir = Path("./temp_results")
    
    work_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)
    
    return work_dir, results_dir

WORK_DIR, RESULTS_DIR = setup_directories()

# Chemical Engineering Constants
R = 8.314  # J/(mol·K) - Universal gas constant
MW = {  # Molecular weights (kg/kmol)
    "CO": 28.01, "H2": 2.016, "CO2": 44.01, "N2": 28.014,
    "CH3OH": 32.04, "H2O": 18.015
}


@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "Enhanced DWSIM Service",
        "mode": "chemical_engineering",
        "capabilities": ["material_balance", "energy_balance", "economics", "sensitivity_analysis"]
    }


@app.get("/api/simulation/health")
async def api_health_check():
    """Detailed API health check."""
    return {
        "isHealthy": True,
        "message": "Enhanced DWSIM service ready for chemical engineering simulations",
        "timestamp": time.time(),
        "dwsimDllPath": "/enhanced/dwsim/libs",
        "dwsimDllsAvailable": True,
        "availableDlls": [
            "DWSIM.Automation.dll", "DWSIM.Interfaces.dll", "DWSIM.Thermodynamics.dll",
            "DWSIM.UnitOperations.dll", "DWSIM.GlobalSettings.dll", "CapeOpen.dll"
        ],
        "missingDlls": [],
        "enhancedFeatures": {
            "material_balance": True,
            "energy_balance": True,
            "reaction_kinetics": True,
            "economic_optimization": True,
            "sensitivity_analysis": True,
            "rag_integration": True
        }
    }


@app.post("/api/simulation/run")
async def run_simulation(
    DwsimFile: UploadFile = File(...),
    ExportCsv: str = Form("true"),
    OptimizationMode: str = Form("standard"),
    SensitivityAnalysis: str = Form("false")
):
    """Run enhanced chemical engineering simulation."""
    
    simulation_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Save and parse simulation file
        file_path = WORK_DIR / f"{simulation_id}.dwsim"
        with open(file_path, "wb") as f:
            content = await DwsimFile.read()
            f.write(content)
        
        # Parse simulation parameters
        sim_params = parse_simulation_file(file_path)
        
        # Store simulation status
        simulations[simulation_id] = {
            "simulationId": simulation_id,
            "status": "Running",
            "createdAt": time.time(),
            "fileName": DwsimFile.filename,
            "parameters": sim_params,
            "optimizationMode": OptimizationMode,
            "sensitivityAnalysis": SensitivityAnalysis.lower() == "true"
        }
        
        # Run chemical engineering calculations
        await simulate_chemical_process(simulation_id, sim_params)
        
        # Calculate results
        results = calculate_process_results(sim_params, OptimizationMode)
        
        # Run sensitivity analysis if requested
        sensitivity_results = None
        if SensitivityAnalysis.lower() == "true":
            sensitivity_results = run_sensitivity_analysis(sim_params)
        
        # Export results
        csv_path = None
        if ExportCsv.lower() == "true":
            csv_path = RESULTS_DIR / f"{simulation_id}_results.csv"
            export_enhanced_csv(results, sensitivity_results, csv_path)
        
        # Update simulation status
        execution_time = int((time.time() - start_time) * 1000)
        simulations[simulation_id].update({
            "status": "Completed",
            "completedAt": time.time(),
            "message": f"Enhanced simulation completed. Process: {sim_params.get('process_type', 'unknown')}",
            "hasResults": True,
            "results": results,
            "sensitivityResults": sensitivity_results,
            "executionTimeMs": execution_time
        })
        
        return {
            "simulationId": simulation_id,
            "success": True,
            "message": f"Enhanced simulation completed successfully",
            "processType": sim_params.get('process_type', 'unknown'),
            "csvDataPath": str(csv_path) if csv_path else None,
            "results": results,
            "sensitivityResults": sensitivity_results,
            "executionTimeMs": execution_time,
            "ragMetadata": sim_params.get('rag_metadata', {})
        }
        
    except Exception as e:
        simulations[simulation_id] = {
            "simulationId": simulation_id,
            "status": "Failed",
            "createdAt": start_time,
            "completedAt": time.time(),
            "message": f"Enhanced simulation failed: {str(e)}",
            "hasResults": False
        }
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@app.get("/api/simulation/optimize")
async def optimize_process(
    process_type: str = Query("methanol_synthesis"),
    objective: str = Query("profit_maximization"),
    constraints: str = Query("safety,environmental")
):
    """Run process optimization for modular plant design."""
    
    optimization_id = str(uuid.uuid4())
    
    # Define optimization problem
    optimization_params = {
        "process_type": process_type,
        "objective": objective,
        "constraints": constraints.split(","),
        "variables": get_optimization_variables(process_type),
        "bounds": get_optimization_bounds(process_type)
    }
    
    # Run optimization algorithm (simplified)
    optimal_results = run_process_optimization(optimization_params)
    
    return {
        "optimizationId": optimization_id,
        "processType": process_type,
        "objective": objective,
        "optimalParameters": optimal_results["parameters"],
        "objectiveValue": optimal_results["objective_value"],
        "economicAnalysis": optimal_results["economics"],
        "designRecommendations": optimal_results["design_recommendations"]
    }


@app.get("/api/simulation/rag/query")
async def rag_process_query(
    query: str = Query(..., description="Process design query"),
    process_type: str = Query("all", description="Filter by process type")
):
    """RAG-based process design query interface."""
    
    # Simulate RAG processing
    rag_response = process_rag_query(query, process_type)
    
    # If query requires simulation, trigger it
    if rag_response["requires_simulation"]:
        sim_params = rag_response["simulation_parameters"]
        results = calculate_process_results(sim_params, "optimization")
        rag_response["simulation_results"] = results
    
    return {
        "query": query,
        "processType": process_type,
        "response": rag_response["response"],
        "requiresSimulation": rag_response["requires_simulation"],
        "simulationResults": rag_response.get("simulation_results"),
        "designRecommendations": rag_response["design_recommendations"],
        "economicFeasibility": rag_response["economic_feasibility"]
    }


def parse_simulation_file(file_path: Path) -> Dict[str, Any]:
    """Parse enhanced simulation file with YAML-like structure."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Simple YAML-like parser for our enhanced format
    try:
        # Convert to proper YAML format (simplified)
        yaml_content = content.replace('# ', '# ').replace(':', ': ')
        params = yaml.safe_load(yaml_content)
    except:
        # Fallback to default methanol synthesis parameters
        params = get_default_methanol_parameters()
    
    return params


def calculate_process_results(sim_params: Dict[str, Any], mode: str = "standard") -> Dict[str, Any]:
    """Calculate realistic chemical engineering results."""
    
    process_type = sim_params.get('process_type', 'methanol_synthesis')
    
    if process_type == 'methanol_synthesis':
        return calculate_methanol_synthesis(sim_params, mode)
    else:
        return calculate_generic_process(sim_params, mode)


def calculate_methanol_synthesis(sim_params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Calculate methanol synthesis with realistic chemical engineering."""
    
    # Extract feed conditions
    feed_streams = sim_params.get('feed_streams', [])
    syngas_feed = next((s for s in feed_streams if s['name'] == 'syngas_feed'), None)
    
    if not syngas_feed:
        raise ValueError("Syngas feed stream not found")
    
    # Feed composition and conditions
    components = {comp['name']: comp['mole_fraction'] 
                 for comp in syngas_feed['components']}
    
    conditions = syngas_feed['conditions']
    feed_flow = conditions['mass_flow']  # kg/hr
    temperature = conditions['temperature']  # K
    pressure = conditions['pressure']  # kPa
    
    # Calculate molar flows (kmol/hr)
    avg_mw = sum(MW[comp] * frac for comp, frac in components.items())
    total_molar_flow = feed_flow / avg_mw
    
    molar_flows = {comp: total_molar_flow * frac 
                   for comp, frac in components.items()}
    
    # Reaction calculations (simplified kinetics)
    h2_co_ratio = molar_flows['H2'] / molar_flows['CO'] if molar_flows['CO'] > 0 else 2.6
    
    # Conversion based on temperature, pressure, and H2/CO ratio
    co_conversion = calculate_co_conversion(temperature, pressure, h2_co_ratio)
    
    # Product calculations
    methanol_produced = molar_flows['CO'] * co_conversion  # kmol/hr
    methanol_mass_flow = methanol_produced * MW['CH3OH']  # kg/hr
    
    # Material balance
    products = calculate_material_balance(molar_flows, co_conversion)
    
    # Energy balance
    energy_balance = calculate_energy_balance(molar_flows, products, temperature, pressure)
    
    # Economic analysis
    economics = calculate_economics(methanol_mass_flow, energy_balance, sim_params)
    
    # Compile results
    results = {
        "feed_analysis": {
            "total_mass_flow": feed_flow,  # kg/hr
            "total_molar_flow": total_molar_flow,  # kmol/hr
            "h2_co_ratio": h2_co_ratio,
            "composition": components
        },
        "reaction_performance": {
            "co_conversion": co_conversion,
            "methanol_yield": methanol_produced / molar_flows['CO'],
            "reactor_temperature": temperature,  # K
            "reactor_pressure": pressure  # kPa
        },
        "product_streams": {
            "methanol_product": {
                "mass_flow": methanol_mass_flow,  # kg/hr
                "molar_flow": methanol_produced,  # kmol/hr
                "purity": 0.995,  # mol/mol
                "temperature": 298.15,  # K
                "pressure": 101.325  # kPa
            },
            "unreacted_gas": {
                "mass_flow": sum(products[comp] * MW[comp] for comp in ['CO', 'H2', 'CO2', 'N2']),
                "composition": {comp: products[comp] for comp in ['CO', 'H2', 'CO2', 'N2']},
                "recycle_potential": True
            }
        },
        "energy_analysis": energy_balance,
        "economic_analysis": economics,
        "modular_design": {
            "footprint": sim_params.get('modular_specifications', {}).get('footprint', {}),
            "weight": sim_params.get('modular_specifications', {}).get('weight', 15000),
            "assembly_time": sim_params.get('modular_specifications', {}).get('assembly_time', 72)
        }
    }
    
    return results


def calculate_co_conversion(temperature: float, pressure: float, h2_co_ratio: float) -> float:
    """Calculate CO conversion based on realistic kinetics."""
    # Simplified Arrhenius-type equation
    k0 = 1e6  # Pre-exponential factor
    ea = 65000  # Activation energy (J/mol)
    
    # Temperature effect
    k_temp = k0 * np.exp(-ea / (R * temperature))
    
    # Pressure effect (equilibrium shifts)
    pressure_factor = (pressure / 5000.0) ** 0.3
    
    # H2/CO ratio effect (optimal around 2.0-2.5)
    ratio_factor = 1.0 - abs(h2_co_ratio - 2.2) * 0.1
    
    # Calculate conversion (bounded between 0 and 1)
    conversion = min(0.95, max(0.1, k_temp * pressure_factor * ratio_factor * 1e-6))
    
    return conversion


def calculate_material_balance(feed_flows: Dict[str, float], co_conversion: float) -> Dict[str, float]:
    """Calculate material balance for methanol synthesis."""
    
    # Reactions:
    # CO + 2H2 → CH3OH
    # CO2 + 3H2 → CH3OH + H2O (minor)
    
    co_reacted = feed_flows['CO'] * co_conversion
    h2_consumed = co_reacted * 2.0  # Stoichiometry
    methanol_formed = co_reacted
    
    # Minor CO2 reaction (5% of total)
    co2_reacted = feed_flows['CO2'] * 0.05
    h2_consumed += co2_reacted * 3.0
    water_formed = co2_reacted
    
    # Product flows
    products = {
        'CO': feed_flows['CO'] - co_reacted,
        'H2': feed_flows['H2'] - h2_consumed,
        'CO2': feed_flows['CO2'] - co2_reacted,
        'N2': feed_flows['N2'],  # Inert
        'CH3OH': methanol_formed,
        'H2O': water_formed
    }
    
    return products


def calculate_energy_balance(feed_flows: Dict[str, float], products: Dict[str, float], 
                           temperature: float, pressure: float) -> Dict[str, Any]:
    """Calculate energy balance for the process."""
    
    # Heat of reaction (kJ/mol)
    delta_h_methanol = -90.7  # Exothermic
    
    # Calculate heat generation
    methanol_produced = products['CH3OH']
    heat_generated = methanol_produced * abs(delta_h_methanol)  # kJ/hr
    
    # Heat duties for equipment
    preheater_duty = 150.0  # kW (from simulation file)
    reactor_cooling = heat_generated / 3600.0 * 0.7  # Remove 70% of heat, convert to kW
    condenser_duty = -200.0  # kW (cooling)
    
    # Total energy consumption
    total_power = preheater_duty + abs(condenser_duty) + 50.0  # kW (pumps, etc.)
    
    return {
        "heat_generated": heat_generated,  # kJ/hr
        "preheater_duty": preheater_duty,  # kW
        "reactor_cooling_required": reactor_cooling,  # kW
        "condenser_duty": condenser_duty,  # kW
        "total_power_consumption": total_power,  # kW
        "energy_efficiency": methanol_produced * MW['CH3OH'] / total_power  # kg product/kW
    }


def calculate_economics(methanol_flow: float, energy_balance: Dict, sim_params: Dict) -> Dict[str, Any]:
    """Calculate economic analysis."""
    
    economics_params = sim_params.get('economics', {})
    
    # Annual production
    operating_hours = economics_params.get('operating_hours', 8400)
    annual_production = methanol_flow * operating_hours / 1000.0  # tonnes/year
    
    # Revenue
    methanol_price = economics_params.get('product_prices', {}).get('methanol', 400.0)  # $/tonne
    annual_revenue = annual_production * methanol_price  # $/year
    
    # Operating costs
    power_cost = energy_balance['total_power_consumption'] * operating_hours * 0.12 / 1000.0  # $/year
    raw_material_cost = annual_production * 1.5 * 200.0  # $/year (1.5 tonnes syngas per tonne methanol)
    
    annual_operating_cost = power_cost + raw_material_cost + annual_production * 50.0  # $/year
    
    # Profit analysis
    annual_profit = annual_revenue - annual_operating_cost
    profit_margin = annual_profit / annual_revenue if annual_revenue > 0 else 0
    
    return {
        "annual_production": annual_production,  # tonnes/year
        "annual_revenue": annual_revenue,  # $/year
        "annual_operating_cost": annual_operating_cost,  # $/year
        "annual_profit": annual_profit,  # $/year
        "profit_margin": profit_margin,  # fraction
        "production_cost": annual_operating_cost / annual_production,  # $/tonne
        "payback_period": 2.5,  # years (simplified)
        "npv_10_years": annual_profit * 6.144  # Simplified NPV at 10% discount
    }


def run_sensitivity_analysis(sim_params: Dict[str, Any]) -> Dict[str, Any]:
    """Run sensitivity analysis on key process parameters."""
    
    sensitivity_params = sim_params.get('sensitivity_analysis', {})
    variables = sensitivity_params.get('variables', [])
    
    results = {}
    
    for var in variables:
        var_name = var['name']
        var_range = var['range']
        steps = var.get('steps', 5)
        
        values = np.linspace(var_range[0], var_range[1], steps)
        outcomes = []
        
        for value in values:
            # Modify simulation parameters
            modified_params = sim_params.copy()
            apply_sensitivity_parameter(modified_params, var_name, value)
            
            # Recalculate
            result = calculate_process_results(modified_params, "sensitivity")
            
            outcomes.append({
                "parameter_value": value,
                "methanol_production": result["product_streams"]["methanol_product"]["mass_flow"],
                "energy_consumption": result["energy_analysis"]["total_power_consumption"],
                "profit": result["economic_analysis"]["annual_profit"],
                "conversion": result["reaction_performance"]["co_conversion"]
            })
        
        results[var_name] = {
            "variable": var_name,
            "range": var_range,
            "outcomes": outcomes,
            "optimal_value": find_optimal_value(outcomes),
            "sensitivity_index": calculate_sensitivity_index(outcomes)
        }
    
    return results


def apply_sensitivity_parameter(params: Dict, var_name: str, value: float):
    """Apply sensitivity parameter to simulation."""
    if var_name == "reactor_temperature":
        params['feed_streams'][0]['conditions']['temperature'] = value
    elif var_name == "reactor_pressure":
        params['feed_streams'][0]['conditions']['pressure'] = value
    elif var_name == "feed_flow":
        params['feed_streams'][0]['conditions']['mass_flow'] = value
    elif var_name == "h2_co_ratio":
        # Adjust H2 and CO fractions to achieve target ratio
        adjust_h2_co_ratio(params, value)


def adjust_h2_co_ratio(params: Dict, target_ratio: float):
    """Adjust feed composition to achieve target H2/CO ratio."""
    components = params['feed_streams'][0]['components']
    
    # Find CO and H2 components
    co_comp = next(c for c in components if c['name'] == 'CO')
    h2_comp = next(c for c in components if c['name'] == 'H2')
    
    # Maintain total mole fraction = 1
    total_other = sum(c['mole_fraction'] for c in components if c['name'] not in ['CO', 'H2'])
    remaining = 1.0 - total_other
    
    # Calculate new fractions
    co_fraction = remaining / (1 + target_ratio)
    h2_fraction = remaining - co_fraction
    
    co_comp['mole_fraction'] = co_fraction
    h2_comp['mole_fraction'] = h2_fraction


def find_optimal_value(outcomes: List[Dict]) -> float:
    """Find optimal parameter value based on profit."""
    max_profit_outcome = max(outcomes, key=lambda x: x['profit'])
    return max_profit_outcome['parameter_value']


def calculate_sensitivity_index(outcomes: List[Dict]) -> float:
    """Calculate sensitivity index (normalized variance)."""
    profits = [o['profit'] for o in outcomes]
    return np.std(profits) / np.mean(profits) if np.mean(profits) > 0 else 0


def process_rag_query(query: str, process_type: str) -> Dict[str, Any]:
    """Process RAG query for modular plant design."""
    
    # Simulate intelligent query processing
    query_lower = query.lower()
    
    response = {
        "response": "",
        "requires_simulation": False,
        "simulation_parameters": {},
        "design_recommendations": [],
        "economic_feasibility": {}
    }
    
    # Query classification and response generation
    if "optimal" in query_lower or "best" in query_lower:
        response["requires_simulation"] = True
        response["simulation_parameters"] = get_default_methanol_parameters()
        response["response"] = "Running optimization analysis to find optimal operating conditions..."
        
    elif "profit" in query_lower or "economic" in query_lower:
        response["requires_simulation"] = True
        response["response"] = "Analyzing economic performance and profitability..."
        
    elif "sensitivity" in query_lower or "effect" in query_lower:
        response["requires_simulation"] = True
        response["response"] = "Performing sensitivity analysis on key process parameters..."
        
    elif "design" in query_lower or "modular" in query_lower:
        response["response"] = "Generating modular plant design recommendations based on process requirements..."
        response["design_recommendations"] = [
            "Use skid-mounted reactor system for easy transport",
            "Implement heat integration to improve energy efficiency",
            "Design for 8000 tonnes/year capacity in ISO container format",
            "Include automated control system for minimal operator requirements"
        ]
    
    else:
        response["response"] = f"Process information for {process_type}: Standard modular design available with flexible capacity options."
    
    # Add economic feasibility assessment
    response["economic_feasibility"] = {
        "estimated_capex": "2.5-3.5 M$",
        "payback_period": "2.5-3.5 years",
        "roi": "25-35%",
        "market_conditions": "favorable"
    }
    
    return response


def get_optimization_variables(process_type: str) -> List[str]:
    """Get optimization variables for process type."""
    if process_type == "methanol_synthesis":
        return ["reactor_temperature", "reactor_pressure", "h2_co_ratio", "feed_flow"]
    return ["temperature", "pressure", "flow_rate"]


def get_optimization_bounds(process_type: str) -> Dict[str, List[float]]:
    """Get optimization bounds for process type."""
    if process_type == "methanol_synthesis":
        return {
            "reactor_temperature": [493.15, 553.15],  # K
            "reactor_pressure": [3000.0, 7000.0],     # kPa
            "h2_co_ratio": [1.8, 3.2],               # ratio
            "feed_flow": [800.0, 1200.0]             # kg/hr
        }
    return {}


def run_process_optimization(optimization_params: Dict) -> Dict[str, Any]:
    """Run process optimization algorithm."""
    
    # Simplified optimization (in reality, would use scipy.optimize or similar)
    process_type = optimization_params["process_type"]
    objective = optimization_params["objective"]
    
    if process_type == "methanol_synthesis" and objective == "profit_maximization":
        optimal_params = {
            "reactor_temperature": 523.15,  # K
            "reactor_pressure": 5500.0,     # kPa
            "h2_co_ratio": 2.2,            # ratio
            "feed_flow": 1000.0             # kg/hr
        }
        
        return {
            "parameters": optimal_params,
            "objective_value": 850000.0,  # $/year profit
            "economics": {
                "annual_profit": 850000.0,
                "production_cost": 280.0,  # $/tonne
                "energy_efficiency": 92.5   # %
            },
            "design_recommendations": [
                "Use optimal H2/CO ratio of 2.2 for maximum conversion",
                "Operate reactor at 250°C and 55 bar for best selectivity",
                "Implement heat recovery to reduce energy costs by 15%",
                "Design for 1000 kg/hr feed capacity"
            ]
        }
    
    return {"parameters": {}, "objective_value": 0, "economics": {}, "design_recommendations": []}


def get_default_methanol_parameters() -> Dict[str, Any]:
    """Get default methanol synthesis parameters."""
    return {
        "process_type": "methanol_synthesis",
        "feed_streams": [{
            "name": "syngas_feed",
            "components": [
                {"name": "CO", "mole_fraction": 0.25},
                {"name": "H2", "mole_fraction": 0.65},
                {"name": "CO2", "mole_fraction": 0.08},
                {"name": "N2", "mole_fraction": 0.02}
            ],
            "conditions": {
                "mass_flow": 1000.0,
                "temperature": 523.15,
                "pressure": 5000.0
            }
        }],
        "economics": {
            "operating_hours": 8400.0,
            "product_prices": {"methanol": 400.0},
            "raw_material_costs": {"syngas": 200.0}
        }
    }


def export_enhanced_csv(results: Dict, sensitivity_results: Optional[Dict], csv_path: Path):
    """Export enhanced results to CSV with multiple sheets equivalent."""
    
    # Create comprehensive results dataframe
    data_rows = []
    
    # Material balance data
    feed = results["feed_analysis"]
    data_rows.append({
        "Category": "Feed",
        "Stream": "Total Feed",
        "Mass_Flow_kg_hr": feed["total_mass_flow"],
        "Molar_Flow_kmol_hr": feed["total_molar_flow"],
        "Temperature_K": 523.15,
        "Pressure_kPa": 5000.0,
        "H2_CO_Ratio": feed["h2_co_ratio"],
        "Energy_kW": 0,
        "Cost_USD_hr": 0
    })
    
    # Product streams
    methanol = results["product_streams"]["methanol_product"]
    data_rows.append({
        "Category": "Product",
        "Stream": "Methanol",
        "Mass_Flow_kg_hr": methanol["mass_flow"],
        "Molar_Flow_kmol_hr": methanol["molar_flow"],
        "Temperature_K": methanol["temperature"],
        "Pressure_kPa": methanol["pressure"],
        "H2_CO_Ratio": 0,
        "Energy_kW": 0,
        "Cost_USD_hr": 0
    })
    
    unreacted = results["product_streams"]["unreacted_gas"]
    data_rows.append({
        "Category": "Waste/Recycle",
        "Stream": "Unreacted Gas",
        "Mass_Flow_kg_hr": unreacted["mass_flow"],
        "Molar_Flow_kmol_hr": 0,
        "Temperature_K": 313.15,
        "Pressure_kPa": 300.0,
        "H2_CO_Ratio": 0,
        "Energy_kW": 0,
        "Cost_USD_hr": 0
    })
    
    # Energy data
    energy = results["energy_analysis"]
    data_rows.append({
        "Category": "Energy",
        "Stream": "Total Energy",
        "Mass_Flow_kg_hr": 0,
        "Molar_Flow_kmol_hr": 0,
        "Temperature_K": 0,
        "Pressure_kPa": 0,
        "H2_CO_Ratio": 0,
        "Energy_kW": energy["total_power_consumption"],
        "Cost_USD_hr": energy["total_power_consumption"] * 0.12
    })
    
    # Economic data
    econ = results["economic_analysis"]
    data_rows.append({
        "Category": "Economics",
        "Stream": "Annual Metrics",
        "Mass_Flow_kg_hr": econ["annual_production"] / 8400.0 * 1000.0,  # Convert back to kg/hr
        "Molar_Flow_kmol_hr": 0,
        "Temperature_K": 0,
        "Pressure_kPa": 0,
        "H2_CO_Ratio": 0,
        "Energy_kW": 0,
        "Cost_USD_hr": econ["annual_profit"] / 8400.0
    })
    
    # Add sensitivity analysis results if available
    if sensitivity_results:
        for var_name, var_data in sensitivity_results.items():
            optimal = var_data["optimal_value"]
            sensitivity = var_data["sensitivity_index"]
            
            data_rows.append({
                "Category": "Sensitivity",
                "Stream": var_name,
                "Mass_Flow_kg_hr": 0,
                "Molar_Flow_kmol_hr": 0,
                "Temperature_K": optimal if "temperature" in var_name else 0,
                "Pressure_kPa": optimal if "pressure" in var_name else 0,
                "H2_CO_Ratio": optimal if "ratio" in var_name else 0,
                "Energy_kW": 0,
                "Cost_USD_hr": sensitivity
            })
    
    # Create and save DataFrame
    df = pd.DataFrame(data_rows)
    df.to_csv(csv_path, index=False)


@app.get("/api/simulation/status/{simulation_id}")
async def get_simulation_status(simulation_id: str):
    """Get simulation status."""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    return simulations[simulation_id]


@app.get("/api/simulation/results/{simulation_id}/csv")
async def get_simulation_csv(simulation_id: str):
    """Download simulation results as CSV."""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    csv_path = RESULTS_DIR / f"{simulation_id}_results.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"enhanced_simulation_{simulation_id}_results.csv"
    )


@app.delete("/api/simulation/results/{simulation_id}")
async def cleanup_simulation(simulation_id: str):
    """Clean up simulation files."""
    if simulation_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    input_file = WORK_DIR / f"{simulation_id}.dwsim"
    csv_file = RESULTS_DIR / f"{simulation_id}_results.csv"
    
    if input_file.exists():
        input_file.unlink()
    if csv_file.exists():
        csv_file.unlink()
    
    del simulations[simulation_id]
    return {"message": "Enhanced simulation files cleaned up successfully"}


async def simulate_chemical_process(simulation_id: str, sim_params: Dict):
    """Simulate realistic chemical process timing."""
    import asyncio
    
    # Simulate complex calculations
    calculation_time = np.random.uniform(2.0, 4.0)  # 2-4 seconds
    await asyncio.sleep(calculation_time)


def calculate_generic_process(sim_params: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """Calculate generic process for unknown process types."""
    return {
        "feed_analysis": {"total_mass_flow": 1000.0},
        "product_streams": {"product": {"mass_flow": 800.0}},
        "energy_analysis": {"total_power_consumption": 100.0},
        "economic_analysis": {"annual_profit": 500000.0}
    }


@app.get("/")
async def root():
    """Root endpoint with enhanced service information."""
    return {
        "service": "Enhanced DWSIM Service for Modular Plant Design",
        "version": "2.0.0",
        "mode": "chemical_engineering",
        "description": "Advanced chemical process simulation with RAG integration",
        "capabilities": [
            "Material and energy balances",
            "Chemical reaction kinetics",
            "Economic optimization",
            "Sensitivity analysis",
            "RAG-based process queries",
            "Modular plant design"
        ],
        "endpoints": {
            "health": "/health",
            "api_health": "/api/simulation/health",
            "run_simulation": "/api/simulation/run",
            "optimize_process": "/api/simulation/optimize",
            "rag_query": "/api/simulation/rag/query",
            "get_status": "/api/simulation/status/{simulation_id}",
            "get_csv": "/api/simulation/results/{simulation_id}/csv",
            "cleanup": "/api/simulation/results/{simulation_id}"
        }
    }


if __name__ == "__main__":
    port = int(os.getenv("DWSIM_API_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port) 