"""
Configuration Manager for DWSIM Simulations

Handles:
- JSON/CSV configuration loading with Pydantic validation
- Dynamic simulation parameter adjustment
- Template generation for easy configuration
- Type-safe validation of simulation parameters
"""

import json
import sys
import os
import csv
import logging
from pathlib import Path

# Try to import pandas
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Some features may be limited.")

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Try to import jsonschema for validation
try:
    import jsonschema
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    print("Warning: jsonschema not available. Schema validation disabled.")

# Try to import pydantic for validation
try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("Warning: pydantic not available. Advanced validation disabled.")

# Handle settings import with fallback
try:
    from .settings import AppSettings, load_settings, SimulationSettings, FeedComponent, FeedConditions, OperatingConditions
except ImportError:
    # Create fallback classes if settings import fails
    class AppSettings:
        def __init__(self, simulations=None):
            self.simulations = simulations or []
        
        def model_dump_json(self, indent=4):
            return json.dumps({"simulations": [s.__dict__ if hasattr(s, '__dict__') else s for s in self.simulations]}, indent=indent)
    
    class SimulationSettings:
        def __init__(self, simulation_name="", feed=None, operating=None):
            self.simulation_name = simulation_name
            self.feed = feed
            self.operating = operating
    
    class FeedComponent:
        def __init__(self, name="", mole_fraction=0.0, mass_flow_kgh=0.0):
            self.name = name
            self.mole_fraction = mole_fraction
            self.mass_flow_kgh = mass_flow_kgh
    
    class FeedConditions:
        def __init__(self, temperature_c=25.0, pressure_kpa=101.325, total_flow_kmol_h=100.0, components=None):
            self.temperature_c = temperature_c
            self.pressure_kpa = pressure_kpa
            self.total_flow_kmol_h = total_flow_kmol_h
            self.components = components or []
    
    class OperatingConditions:
        def __init__(self, reflux_ratio=2.0, residence_time_min=30.0):
            self.reflux_ratio = reflux_ratio
            self.residence_time_min = residence_time_min
    
    def load_settings(filepath):
        return AppSettings()

# Create fallback functions for pandas operations
class FallbackDataFrame:
    def __init__(self, data):
        self.data = data if isinstance(data, list) else [data]
    
    def to_csv(self, path, index=False):
        with open(path, 'w', newline='') as f:
            if self.data:
                keys = self.data[0].keys() if isinstance(self.data[0], dict) else []
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.data)
                
    @property
    def columns(self):
        if self.data and isinstance(self.data[0], dict):
            return list(self.data[0].keys())
        return []

# Mock pandas functions
def read_csv(filepath):
    """Fallback CSV reader"""
    data = []
    try:
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except Exception as e:
        print(f"Error reading CSV: {e}")
    return FallbackDataFrame(data)

def notna(value):
    """Check if value is not NaN/None"""
    return value is not None and str(value).strip() != ''

# Create a mock pandas module
class MockPandas:
    DataFrame = FallbackDataFrame
    read_csv = staticmethod(read_csv)
    notna = staticmethod(notna)

pd = MockPandas()

from typing import Dict, List, Union, Any, Optional
from datetime import datetime

class ConfigManager:
    """Manages DWSIM simulation configurations using Pydantic models."""
    
    def __init__(self, config_dir: str | Path = "configs"):
        """Initialize ConfigManager with specified directory."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # Create only the essential bulk-ready templates for 1000+ modular plants
        # These templates include all functionality of the basic templates plus comprehensive bulk support
        self.create_bulk_ready_json_template(verbose=False)
        self.create_bulk_ready_csv_template(verbose=False)

    def load(self, file_name: str) -> AppSettings:
        """Load settings using Pydantic validation."""
        return load_settings(self.config_dir / file_name)

    def create_template_json(self, file_name: str = "template.json", verbose: bool = False) -> Path:
        """Create a JSON template using Pydantic models."""
        filepath = self.config_dir / file_name
        
        # Check if file already exists
        if filepath.exists():
            if verbose:
                print(f"⏭️ Template already exists, skipping: {filepath}")
            return filepath
        
        # Create template with sample simulation
        sample_simulation = SimulationSettings(
            simulation_name="ethanol_distillation_example",
            feed=FeedConditions(
                temperature_c=78.4,
                pressure_kpa=101.325,
                total_flow_kmol_h=100.0,
                components=[
                    FeedComponent(name="water", mole_fraction=0.4, mass_flow_kgh=720.0),
                    FeedComponent(name="ethanol", mole_fraction=0.6, mass_flow_kgh=1380.0)
                ]
            ),
            operating=OperatingConditions(
                reflux_ratio=2.5,
                residence_time_min=30.0
            )
        )
        
        template = AppSettings(simulations=[sample_simulation])
        template_json = template.model_dump_json(indent=4)
        filepath.write_text(template_json)
        
        if verbose:
            print(f"✅ Pydantic template created: {filepath}")
        
        return filepath
    
    def create_template_csv(self, filename: str, verbose: bool = False) -> str:
        """Create a CSV template for simulation configuration. Skip if file already exists."""
        
        filepath = self.config_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            if verbose:
                print(f"⏭️ Template already exists, skipping: {filepath}")
            return str(filepath)
        
        template_data = [
            {
                'name': 'ethanol_distillation_example',
                'type': 'distillation',
                'components': 'water,ethanol', 
                'description': 'Ethanol-water separation column',
                'temperature': 78.4,
                'pressure': 101325,
                'flow_rate': 1000,
                'reflux_ratio': 2.5,
                'expected_conversion': 0.95,
                'expected_selectivity': 0.98,
                'expected_yield': 0.93
            },
            {
                'name': 'methane_reforming_example',
                'type': 'reactor', 
                'components': 'methane,steam,hydrogen,carbon_monoxide',
                'description': 'Steam methane reforming reactor',
                'temperature': 850,
                'pressure': 2500000,
                'flow_rate': 500,
                'catalyst_loading': 100,
                'expected_conversion': 0.85,
                'expected_selectivity': 0.92,
                'expected_yield': 0.78
            }
        ]
        
        df = pd.DataFrame(template_data)
        df.to_csv(filepath, index=False)
        
        if verbose:
            print(f"✅ Template created: {filepath}")
        
        return str(filepath)
    
    def create_bulk_ready_json_template(self, filename: str = "bulk_modular_plants_template.json", verbose: bool = False) -> str:
        """Create a JSON template optimized for bulk input of 1000+ modular plants."""
        
        filepath = self.config_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            if verbose:
                print(f"⏭️ Bulk template already exists, skipping: {filepath}")
            return str(filepath)
        
        # Create comprehensive template for bulk operations
        bulk_template = {
            "metadata": {
                "template_version": "2.0",
                "description": "Bulk template for 1000+ modular plants",
                "created_date": datetime.now().isoformat(),
                "supports_bulk_operations": True,
                "max_simulations": "unlimited",
                "copy_paste_ready": True
            },
            "simulation_types": {
                "distillation": ["ethanol_water", "methanol_water", "benzene_toluene", "acetone_water"],
                "reactor": ["steam_reforming", "cracking", "polymerization", "hydrogenation"],
                "heat_exchanger": ["shell_tube", "plate", "air_cooled", "reboiler"],
                "absorber": ["co2_capture", "acid_gas_removal", "dehydration"],
                "crystallizer": ["salt_crystallization", "sugar_crystallization", "pharmaceutical"]
            },
            "simulations": [
                # Modular Plant Type 1: Small-Scale Ethanol Plant
                {
                    "plant_id": "MP001",
                    "name": "small_ethanol_plant_africa",
                    "type": "distillation",
                    "location": "West Africa",
                    "capacity": "small",
                    "components": ["water", "ethanol", "methanol"],
                    "description": "Small-scale modular ethanol plant for rural deployment",
                    "parameters": {
                        "temperature": 78.4,
                        "pressure": 101325,
                        "flow_rate": 500,
                        "feed_composition": {"ethanol": 0.12, "water": 0.87, "methanol": 0.01},
                        "reflux_ratio": 2.5,
                        "number_of_stages": 12,
                        "feed_temperature": 25.0,
                        "feed_pressure": 101325,
                        "reboiler_duty": 1500.0,
                        "condenser_duty": 1200.0
                    },
                    "expected_outputs": {
                        "conversion": 0.95,
                        "selectivity": 0.98,
                        "yield": 0.93,
                        "purity": 0.999,
                        "recovery_rate": 0.94
                    },
                    "economic_data": {
                        "capex": 250000,
                        "opex_annual": 75000,
                        "revenue_annual": 180000,
                        "payback_period": 3.2
                    }
                },
                # Modular Plant Type 2: Medium-Scale Methane Reformer
                {
                    "plant_id": "MP002", 
                    "name": "medium_hydrogen_plant_asia",
                    "type": "reactor",
                    "location": "Southeast Asia",
                    "capacity": "medium",
                    "components": ["methane", "steam", "hydrogen", "carbon_monoxide", "carbon_dioxide"],
                    "description": "Medium-scale modular hydrogen production via steam reforming",
                    "parameters": {
                        "temperature": 850,
                        "pressure": 2500000,
                        "flow_rate": 1000,
                        "feed_composition": {"methane": 0.75, "steam": 0.25},
                        "residence_time": 3.0,
                        "catalyst_loading": 150,
                        "steam_to_carbon_ratio": 3.0,
                        "feed_temperature": 400.0,
                        "feed_pressure": 2500000,
                        "reactor_volume": 50.0
                    },
                    "expected_outputs": {
                        "conversion": 0.88,
                        "selectivity": 0.92,
                        "yield": 0.81,
                        "hydrogen_purity": 0.95,
                        "recovery_rate": 0.85
                    },
                    "economic_data": {
                        "capex": 1500000,
                        "opex_annual": 400000,
                        "revenue_annual": 850000,
                        "payback_period": 2.8
                    }
                },
                # Template for Copy-Paste: Add more plants below
                {
                    "plant_id": "MP003",
                    "name": "COPY_PASTE_TEMPLATE_HERE",
                    "type": "SELECT: distillation|reactor|heat_exchanger|absorber|crystallizer",
                    "location": "YOUR_LOCATION",
                    "capacity": "SELECT: small|medium|large",
                    "components": ["component1", "component2", "component3"],
                    "description": "Description of your modular plant",
                    "parameters": {
                        "temperature": 0.0,
                        "pressure": 101325,
                        "flow_rate": 1000,
                        "feed_composition": {"comp1": 0.5, "comp2": 0.5},
                        "reflux_ratio": 2.5,
                        "number_of_stages": 15,
                        "residence_time": 2.0,
                        "catalyst_loading": 100,
                        "feed_temperature": 25.0,
                        "feed_pressure": 101325
                    },
                    "expected_outputs": {
                        "conversion": 0.90,
                        "selectivity": 0.95,
                        "yield": 0.85,
                        "purity": 0.99,
                        "recovery_rate": 0.90
                    },
                    "economic_data": {
                        "capex": 500000,
                        "opex_annual": 150000,
                        "revenue_annual": 300000,
                        "payback_period": 3.5
                    }
                }
            ],
            "bulk_operation_instructions": {
                "how_to_add_1000_plants": [
                    "1. Copy the template simulation (MP003) above",
                    "2. Paste it below and modify the parameters",
                    "3. Increment plant_id (MP004, MP005, etc.)",
                    "4. Update name, location, and parameters",
                    "5. Repeat for each modular plant"
                ],
                "supported_parameters": [
                    "temperature", "pressure", "flow_rate", "feed_composition",
                    "reflux_ratio", "number_of_stages", "residence_time",
                    "catalyst_loading", "feed_temperature", "feed_pressure",
                    "reboiler_duty", "condenser_duty", "reactor_volume",
                    "steam_to_carbon_ratio"
                ],
                "performance_note": "PyNucleus can handle 1000+ simulations efficiently"
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(bulk_template, f, indent=2)
        
        if verbose:
            print(f"✅ Bulk-ready JSON template created: {filepath}")
            print(f"    📊 Supports unlimited modular plants")
            print(f"    📋 Copy-paste ready format")
            print(f"    🏭 Includes economic data fields")
        
        return str(filepath)
    
    def create_bulk_ready_csv_template(self, filename: str = "bulk_modular_plants_template.csv", verbose: bool = False) -> str:
        """Create a CSV template optimized for bulk input of 1000+ modular plants with comprehensive columns."""
        
        filepath = self.config_dir / filename
        
        # Check if file already exists
        if filepath.exists():
            if verbose:
                print(f"⏭️ Bulk CSV template already exists, skipping: {filepath}")
            return str(filepath)
        
        # Create comprehensive CSV template for bulk operations
        bulk_csv_data = [
            {
                # Plant Identification
                'plant_id': 'MP001',
                'name': 'small_ethanol_plant_africa',
                'type': 'distillation',
                'location': 'West Africa',
                'capacity': 'small',
                'components': 'water,ethanol,methanol',
                'description': 'Small-scale modular ethanol plant for rural deployment',
                
                # Process Parameters
                'temperature': 78.4,
                'pressure': 101325,
                'flow_rate': 500,
                'feed_temperature': 25.0,
                'feed_pressure': 101325,
                'reflux_ratio': 2.5,
                'number_of_stages': 12,
                'residence_time': '',
                'catalyst_loading': '',
                'reboiler_duty': 1500.0,
                'condenser_duty': 1200.0,
                'reactor_volume': '',
                'steam_to_carbon_ratio': '',
                
                # Feed Composition (as percentages)
                'comp1_fraction': 0.12,  # ethanol
                'comp2_fraction': 0.87,  # water
                'comp3_fraction': 0.01,  # methanol
                'comp4_fraction': '',
                'comp5_fraction': '',
                
                # Expected Performance
                'expected_conversion': 0.95,
                'expected_selectivity': 0.98,
                'expected_yield': 0.93,
                'expected_purity': 0.999,
                'expected_recovery': 0.94,
                
                # Economic Data
                'capex': 250000,
                'opex_annual': 75000,
                'revenue_annual': 180000,
                'payback_period': 3.2
            },
            {
                # Plant Identification
                'plant_id': 'MP002',
                'name': 'medium_hydrogen_plant_asia',
                'type': 'reactor',
                'location': 'Southeast Asia',
                'capacity': 'medium',
                'components': 'methane,steam,hydrogen,carbon_monoxide,carbon_dioxide',
                'description': 'Medium-scale modular hydrogen production via steam reforming',
                
                # Process Parameters
                'temperature': 850,
                'pressure': 2500000,
                'flow_rate': 1000,
                'feed_temperature': 400.0,
                'feed_pressure': 2500000,
                'reflux_ratio': '',
                'number_of_stages': '',
                'residence_time': 3.0,
                'catalyst_loading': 150,
                'reboiler_duty': '',
                'condenser_duty': '',
                'reactor_volume': 50.0,
                'steam_to_carbon_ratio': 3.0,
                
                # Feed Composition (as percentages)
                'comp1_fraction': 0.75,  # methane
                'comp2_fraction': 0.25,  # steam
                'comp3_fraction': '',
                'comp4_fraction': '',
                'comp5_fraction': '',
                
                # Expected Performance
                'expected_conversion': 0.88,
                'expected_selectivity': 0.92,
                'expected_yield': 0.81,
                'expected_purity': 0.95,
                'expected_recovery': 0.85,
                
                # Economic Data
                'capex': 1500000,
                'opex_annual': 400000,
                'revenue_annual': 850000,
                'payback_period': 2.8
            },
            {
                # Template Row for Copy-Paste
                'plant_id': 'MP003_TEMPLATE',
                'name': 'COPY_THIS_ROW_FOR_BULK_INPUT',
                'type': 'distillation_OR_reactor_OR_heat_exchanger_OR_absorber_OR_crystallizer',
                'location': 'YOUR_LOCATION',
                'capacity': 'small_OR_medium_OR_large',
                'components': 'comp1,comp2,comp3,comp4,comp5',
                'description': 'Description of your modular plant',
                
                # Process Parameters (fill as needed)
                'temperature': 100.0,
                'pressure': 101325,
                'flow_rate': 1000,
                'feed_temperature': 25.0,
                'feed_pressure': 101325,
                'reflux_ratio': 2.5,
                'number_of_stages': 15,
                'residence_time': 2.0,
                'catalyst_loading': 100,
                'reboiler_duty': 1000,
                'condenser_duty': 800,
                'reactor_volume': 25,
                'steam_to_carbon_ratio': 2.5,
                
                # Feed Composition (sum should equal 1.0)
                'comp1_fraction': 0.4,
                'comp2_fraction': 0.3,
                'comp3_fraction': 0.2,
                'comp4_fraction': 0.1,
                'comp5_fraction': '',
                
                # Expected Performance
                'expected_conversion': 0.90,
                'expected_selectivity': 0.95,
                'expected_yield': 0.85,
                'expected_purity': 0.99,
                'expected_recovery': 0.90,
                
                # Economic Data
                'capex': 500000,
                'opex_annual': 150000,
                'revenue_annual': 300000,
                'payback_period': 3.5
            }
        ]
        
        df = pd.DataFrame(bulk_csv_data)
        df.to_csv(filepath, index=False)
        
        if verbose:
            print(f"✅ Bulk-ready CSV template created: {filepath}")
            print(f"    📊 Ready for 1000+ plants via copy-paste")
            print(f"    📋 Comprehensive parameter columns")
            print(f"    🏭 Economic data included")
            print(f"    💡 Template row for easy duplication")
        
        return str(filepath)
    
    def load_from_json(self, filepath: Union[str, Path]) -> List[Dict]:
        """Load simulation configurations from JSON file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Extract simulations array
        if 'simulations' in data:
            simulations = data['simulations']
        elif isinstance(data, list):
            simulations = data
        else:
            simulations = [data]
        
        # Validate each simulation
        validated_sims = []
        for i, sim in enumerate(simulations):
            try:
                jsonschema.validate(sim, self.schema)
                validated_sims.append(self._standardize_simulation(sim))
                print(f"✅ Simulation {i+1} validated: {sim['name']}")
            except jsonschema.ValidationError as e:
                print(f"❌ Validation error in simulation {i+1}: {e.message}")
                # Include with warning
                validated_sims.append(self._standardize_simulation(sim))
        
        return validated_sims
    
    def load_from_csv(self, filepath: Union[str, Path]) -> List[Dict]:
        """Load simulation configurations from CSV file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        simulations = []
        
        for _, row in df.iterrows():
            # Skip template rows
            if str(row.get('plant_id', '')).endswith('_TEMPLATE') or 'COPY_THIS_ROW' in str(row.get('name', '')):
                continue
                
            # Convert CSV row to simulation dict
            sim = {
                'plant_id': row.get('plant_id', f"plant_{len(simulations)+1}"),
                'name': row['name'],
                'type': row['type'],
                'location': row.get('location', ''),
                'capacity': row.get('capacity', ''),
                'components': row['components'].split(',') if isinstance(row['components'], str) else [],
                'description': row.get('description', ''),
                'parameters': {}
            }
            
            # Extract process parameters from columns
            param_columns = [
                'temperature', 'pressure', 'flow_rate', 'reflux_ratio', 
                'number_of_stages', 'residence_time', 'catalyst_loading',
                'feed_temperature', 'feed_pressure', 'reboiler_duty',
                'condenser_duty', 'reactor_volume', 'steam_to_carbon_ratio'
            ]
            
            for col in param_columns:
                if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != '':
                    sim['parameters'][col] = float(row[col])
            
            # Extract feed composition from comp1_fraction to comp5_fraction
            feed_composition = {}
            components = sim['components']
            for i in range(1, 6):  # comp1_fraction to comp5_fraction
                col = f'comp{i}_fraction'
                if col in df.columns and pd.notna(row[col]) and str(row[col]).strip() != '':
                    if i <= len(components):
                        feed_composition[components[i-1]] = float(row[col])
            
            if feed_composition:
                sim['parameters']['feed_composition'] = feed_composition
            
            # Extract expected outputs
            expected_columns = [
                'expected_conversion', 'expected_selectivity', 'expected_yield', 
                'expected_purity', 'expected_recovery'
            ]
            
            expected_outputs = {}
            for col in expected_columns:
                if col in df.columns and pd.notna(row[col]):
                    key = col.replace('expected_', '')
                    if key == 'recovery':
                        key = 'recovery_rate'
                    expected_outputs[key] = float(row[col])
            
            if expected_outputs:
                sim['expected_outputs'] = expected_outputs
            
            # Extract economic data
            economic_columns = ['capex', 'opex_annual', 'revenue_annual', 'payback_period']
            economic_data = {}
            for col in economic_columns:
                if col in df.columns and pd.notna(row[col]):
                    economic_data[col] = float(row[col])
            
            if economic_data:
                sim['economic_data'] = economic_data
            
            simulations.append(self._standardize_simulation(sim))
        
        print(f"✅ Loaded {len(simulations)} simulations from CSV")
        if len(simulations) >= 100:
            print(f"    🏭 Bulk operation detected: {len(simulations)} modular plants loaded")
        
        return simulations
    
    def _standardize_simulation(self, sim: Dict) -> Dict:
        """Standardize simulation configuration format."""
        standardized = {
            'name': sim.get('name', 'unnamed_simulation'),
            'type': sim.get('type', 'reactor'),
            'components': sim.get('components', []),
            'description': sim.get('description', ''),
            'parameters': sim.get('parameters', {}),
            'expected_outputs': sim.get('expected_outputs', {}),
            'timestamp': datetime.now().isoformat(),
            'source': 'config_manager'
        }
        
        return standardized
    
    def save_configuration(self, simulations: List[Dict], filepath: Union[str, Path], format: str = 'json'):
        """Save simulation configurations to file."""
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            data = {'simulations': simulations, 'created': datetime.now().isoformat()}
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format.lower() == 'csv':
            # Flatten configurations for CSV
            flattened = []
            for sim in simulations:
                row = {
                    'name': sim['name'],
                    'type': sim['type'],
                    'components': ','.join(sim['components']),
                    'description': sim['description']
                }
                
                # Add parameters as columns
                for key, value in sim.get('parameters', {}).items():
                    row[key] = value
                
                # Add expected outputs
                for key, value in sim.get('expected_outputs', {}).items():
                    row[f'expected_{key}'] = value
                
                flattened.append(row)
            
            df = pd.DataFrame(flattened)
            df.to_csv(filepath, index=False)
        
        print(f"✅ Configuration saved: {filepath}")
    
    def validate_simulation(self, simulation: Dict) -> bool:
        """Validate a single simulation configuration."""
        try:
            jsonschema.validate(simulation, self.schema)
            return True
        except jsonschema.ValidationError as e:
            print(f"❌ Validation error: {e.message}")
            return False
    
    def get_available_templates(self) -> List[str]:
        """Get list of available configuration templates."""
        templates = []
        for file in self.config_dir.glob("*template*"):
            templates.append(str(file))
        return templates
    
    def merge_configurations(self, *config_files) -> List[Dict]:
        """Merge multiple configuration files."""
        all_simulations = []
        
        for config_file in config_files:
            filepath = Path(config_file)
            if filepath.suffix.lower() == '.json':
                sims = self.load_from_json(filepath)
            elif filepath.suffix.lower() == '.csv':
                sims = self.load_from_csv(filepath)
            else:
                print(f"⚠️ Unsupported file format: {filepath}")
                continue
            
            all_simulations.extend(sims)
        
        print(f"✅ Merged {len(all_simulations)} simulations from {len(config_files)} files")
        return all_simulations
    
    def validate_config(self, config_data: Dict[str, Any], schema_file: str = None) -> Dict[str, Any]:
        """
        Validate configuration data against a schema.
        
        Args:
            config_data: Configuration data to validate
            schema_file: Optional schema file name
            
        Returns:
            Dict with validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not JSONSCHEMA_AVAILABLE:
            validation_result["warnings"].append("jsonschema not available - validation skipped")
            return validation_result
        
        try:
            if schema_file:
                schema_path = self.config_dir / schema_file
                if schema_path.exists():
                    with open(schema_path, 'r') as f:
                        schema = json.load(f)
                    
                    # Validate against schema
                    jsonschema.validate(config_data, schema)
                    validation_result["valid"] = True
                else:
                    validation_result["warnings"].append(f"Schema file not found: {schema_file}")
            
            # Basic validation checks
            if "simulation_name" not in config_data:
                validation_result["errors"].append("Missing required field: simulation_name")
                validation_result["valid"] = False
            
            if "feed_streams" not in config_data:
                validation_result["errors"].append("Missing required field: feed_streams") 
                validation_result["valid"] = False
                
        except Exception as e:
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation error: {str(e)}")
        
        return validation_result 