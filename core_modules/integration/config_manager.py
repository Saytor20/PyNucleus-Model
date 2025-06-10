"""
Configuration Manager for DWSIM Simulations

Handles:
- JSON/CSV configuration loading
- Dynamic simulation parameter adjustment
- Template generation for easy configuration
- Validation of simulation parameters
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Any
import jsonschema
from datetime import datetime


class ConfigManager:
    """Manages DWSIM simulation configurations from various input formats."""
    
    def __init__(self, config_dir: str = "config"):
        """Initialize ConfigManager with configuration directory."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.schema = self._get_simulation_schema()
        
    def _get_simulation_schema(self) -> Dict:
        """Define JSON schema for simulation validation."""
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "type": {"type": "string", "enum": ["distillation", "reactor", "heat_exchanger", "absorber", "crystallizer", "separator", "mixer"]},
                "components": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "description": {"type": "string"},
                "parameters": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number"},
                        "pressure": {"type": "number"},
                        "flow_rate": {"type": "number"},
                        "feed_composition": {"type": "object"},
                        "operating_conditions": {"type": "object"}
                    }
                },
                "expected_outputs": {
                    "type": "object",
                    "properties": {
                        "conversion": {"type": "number", "minimum": 0, "maximum": 1},
                        "selectivity": {"type": "number", "minimum": 0, "maximum": 1},
                        "yield": {"type": "number", "minimum": 0, "maximum": 1},
                        "purity": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            },
            "required": ["name", "type", "components", "description"]
        }
    
    def create_template_json(self, filename: str, verbose: bool = False) -> str:
        """Create a JSON template for simulation configuration."""
        
        template = {
            "simulations": [
                {
                    "name": "ethanol_distillation_example",
                    "type": "distillation", 
                    "components": ["water", "ethanol"],
                    "description": "Ethanol-water separation column",
                    "parameters": {
                        "temperature": 78.4,
                        "pressure": 101325,
                        "flow_rate": 1000,
                        "reflux_ratio": 2.5
                    },
                    "expected_outputs": {
                        "conversion": 0.95,
                        "selectivity": 0.98,
                        "yield": 0.93
                    }
                },
                {
                    "name": "methane_reforming_example", 
                    "type": "reactor",
                    "components": ["methane", "steam", "hydrogen", "carbon_monoxide"],
                    "description": "Steam methane reforming reactor",
                    "parameters": {
                        "temperature": 850,
                        "pressure": 2500000,
                        "flow_rate": 500,
                        "catalyst_loading": 100
                    },
                    "expected_outputs": {
                        "conversion": 0.85,
                        "selectivity": 0.92,
                        "yield": 0.78
                    }
                }
            ]
        }
        
        filepath = self.config_dir / filename
        with open(filepath, 'w') as f:
            json.dump(template, f, indent=2)
        
        if verbose:
            print(f"✅ Template created: {filepath}")
        
        return str(filepath)
    
    def create_template_csv(self, filename: str, verbose: bool = False) -> str:
        """Create a CSV template for simulation configuration."""
        
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
        
        filepath = self.config_dir / filename
        df = pd.DataFrame(template_data)
        df.to_csv(filepath, index=False)
        
        if verbose:
            print(f"✅ Template created: {filepath}")
        
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
            # Convert CSV row to simulation dict
            sim = {
                'name': row['name'],
                'type': row['type'],
                'components': row['components'].split(',') if isinstance(row['components'], str) else [],
                'description': row.get('description', ''),
                'parameters': {}
            }
            
            # Extract parameters from columns
            param_columns = ['temperature', 'pressure', 'flow_rate', 'reflux_ratio', 
                           'number_of_stages', 'residence_time', 'catalyst_loading']
            
            for col in param_columns:
                if col in df.columns and pd.notna(row[col]):
                    sim['parameters'][col] = float(row[col])
            
            # Extract expected outputs
            expected_columns = ['expected_conversion', 'expected_selectivity', 'expected_yield']
            if any(col in df.columns for col in expected_columns):
                sim['expected_outputs'] = {}
                for col in expected_columns:
                    if col in df.columns and pd.notna(row[col]):
                        key = col.replace('expected_', '')
                        sim['expected_outputs'][key] = float(row[col])
            
            simulations.append(self._standardize_simulation(sim))
        
        print(f"✅ Loaded {len(simulations)} simulations from CSV")
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