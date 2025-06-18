"""
Configuration manager for PyNucleus system.
"""

import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Union, List
from datetime import datetime

class ConfigManager:
    """Manage configuration files for PyNucleus pipeline."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def create_template_json(self, filename: str, verbose: bool = False) -> str:
        """
        Create a JSON configuration template.
        
        Args:
            filename: Name of the template file
            verbose: Whether to print verbose output
            
        Returns:
            Path to created template file
        """
        template_config = {
            "simulations": [
                {
                    "case_name": "template_case_1",
                    "temperature": 350.0,
                    "pressure": 2.5,
                    "feed_rate": 100.0,
                    "catalyst_type": "Pt/Al2O3",
                    "process_type": "distillation"
                },
                {
                    "case_name": "template_case_2", 
                    "temperature": 375.0,
                    "pressure": 3.0,
                    "feed_rate": 120.0,
                    "catalyst_type": "Pd/C",
                    "process_type": "reaction"
                }
            ],
            "metadata": {
                "created_by": "ConfigManager",
                "created_at": datetime.now().isoformat(),
                "version": "1.0",
                "description": "Configuration template for PyNucleus simulations"
            }
        }
        
        output_path = self.save(template_config, filename)
        
        if verbose:
            self.logger.info(f"JSON template created: {output_path}")
        
        return str(output_path)
    
    def create_template_csv(self, filename: str, verbose: bool = False) -> str:
        """
        Create a CSV configuration template.
        
        Args:
            filename: Name of the template file
            verbose: Whether to print verbose output
            
        Returns:
            Path to created template file
        """
        template_config = {
            "simulations": [
                {
                    "case_name": "template_case_1",
                    "temperature": 350.0,
                    "pressure": 2.5,
                    "feed_rate": 100.0,
                    "catalyst_type": "Pt/Al2O3",
                    "process_type": "distillation"
                },
                {
                    "case_name": "template_case_2", 
                    "temperature": 375.0,
                    "pressure": 3.0,
                    "feed_rate": 120.0,
                    "catalyst_type": "Pd/C",
                    "process_type": "reaction"
                },
                {
                    "case_name": "template_case_3", 
                    "temperature": 400.0,
                    "pressure": 4.0,
                    "feed_rate": 150.0,
                    "catalyst_type": "Ni/SiO2",
                    "process_type": "hydrogenation"
                }
            ]
        }
        
        output_path = self.save(template_config, filename)
        
        if verbose:
            self.logger.info(f"CSV template created: {output_path}")
        
        return str(output_path)
    
    def load(self, filename: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or CSV file.
        
        Args:
            filename: Name of the configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            if filename.endswith('.json'):
                return self._load_json(config_path)
            elif filename.endswith('.csv'):
                return self._load_csv(config_path)
            else:
                raise ValueError(f"Unsupported configuration file format: {filename}")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration {filename}: {e}")
            raise
    
    def save(self, config: Dict[str, Any], filename: str) -> Path:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            filename: Name of the output file
            
        Returns:
            Path to saved file
        """
        output_path = self.config_dir / filename
        
        try:
            if filename.endswith('.json'):
                self._save_json(config, output_path)
            elif filename.endswith('.csv'):
                self._save_csv(config, output_path)
            else:
                raise ValueError(f"Unsupported configuration file format: {filename}")
                
            self.logger.info(f"Configuration saved: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration {filename}: {e}")
            raise
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load JSON configuration file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_csv(self, file_path: Path) -> Dict[str, Any]:
        """Load CSV configuration file."""
        config = {"simulations": []}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                config["simulations"].append(dict(row))
                
        return config
    
    def _save_json(self, config: Dict[str, Any], file_path: Path):
        """Save configuration as JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, config: Dict[str, Any], file_path: Path):
        """Save configuration as CSV."""
        simulations = config.get("simulations", [])
        
        if not simulations:
            self.logger.warning("No simulations data to save as CSV")
            return
            
        fieldnames = list(simulations[0].keys())
        
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(simulations)
    
    def list_configs(self) -> List[str]:
        """List available configuration files."""
        configs = []
        for pattern in ['*.json', '*.csv']:
            configs.extend([f.name for f in self.config_dir.glob(pattern)])
        return sorted(configs) 