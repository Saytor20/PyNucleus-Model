"""
Plant Builder module for handling modular plant templates and configurations.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from ..utils.logging_config import get_logger
from ..data.mock_data_manager import get_mock_data_manager

logger = get_logger(__name__)

class PlantBuilder:
    """Handles plant template loading and configuration building."""
    
    def __init__(self):
        """Initialize the PlantBuilder with template data."""
        self.mock_data_manager = get_mock_data_manager()
        logger.info("PlantBuilder initialized with mock data manager")
    
    def get_available_templates(self) -> List[Dict[str, Any]]:
        """Get list of available plant templates."""
        return self.mock_data_manager.get_all_plant_templates()
    
    def get_template(self, template_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific template by ID."""
        return self.mock_data_manager.get_plant_template(template_id)
    
    def validate_parameters(self, template: Dict[str, Any], custom_parameters: Dict[str, Any]) -> None:
        """Validate custom parameters against template constraints."""
        # Validate feedstock
        if custom_parameters.get("feedstock") not in template.get("feedstock_options", []):
            raise ValueError(f"Invalid feedstock. Must be one of: {template['feedstock_options']}")
        
        # Validate production capacity
        capacity = custom_parameters.get("production_capacity")
        valid_ranges = template.get("valid_ranges", {})
        if "production_capacity" in valid_ranges:
            min_cap = valid_ranges["production_capacity"]["min"]
            max_cap = valid_ranges["production_capacity"]["max"]
            if not min_cap <= capacity <= max_cap:
                raise ValueError(f"Production capacity must be between {min_cap} and {max_cap} tons/year")
        
        # Validate operating hours
        hours = custom_parameters.get("operating_hours")
        if "operating_hours" in valid_ranges:
            min_hours = valid_ranges["operating_hours"]["min"]
            max_hours = valid_ranges["operating_hours"]["max"]
            if not min_hours <= hours <= max_hours:
                raise ValueError(f"Operating hours must be between {min_hours} and {max_hours} hours/year")
    
    def calculate_adjusted_costs(self, template: Dict[str, Any], custom_parameters: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adjusted costs based on custom parameters."""
        default_params = template["default_parameters"]
        location_factors = template.get("location_factors", {})
        
        # Support both 'production_capacity' and 'production_capacity_tpd'
        default_capacity = default_params.get("production_capacity_tpd")
        if default_capacity is None:
            default_capacity = default_params.get("production_capacity")
        custom_capacity = custom_parameters.get("production_capacity_tpd")
        if custom_capacity is None:
            custom_capacity = custom_parameters.get("production_capacity")
        
        # Get location factor
        location = custom_parameters.get("plant_location", "Texas, USA")
        location_factor = location_factors.get(location, 1.0)
        
        # Calculate capacity scaling factor
        if default_capacity is None or custom_capacity is None:
            scale_factor = 1.0
            capacity_ratio = 1.0
        else:
            capacity_ratio = custom_capacity / default_capacity
            scale_factor = capacity_ratio ** 0.6
        
        # Calculate adjusted costs
        adjusted_capital_cost = default_params["capital_cost"] * scale_factor * location_factor
        adjusted_operating_cost = default_params["operating_cost"] * capacity_ratio * location_factor
        
        return {
            "capital_cost": adjusted_capital_cost,
            "operating_cost": adjusted_operating_cost,
            "product_price": default_params["product_price"]  # Price typically doesn't scale
        }
    
    def build_plant(self, template_id: int, custom_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Build a plant configuration based on template and custom parameters."""
        # Get template
        template = self.get_template(template_id)
        if not template:
            raise ValueError(f"Template with ID {template_id} not found")
        
        # Support both 'production_capacity' and 'production_capacity_tpd' in parameters
        default_params = template["default_parameters"].copy()
        if "production_capacity_tpd" in default_params:
            default_params["production_capacity"] = default_params["production_capacity_tpd"]
        if "production_capacity_tpd" in custom_parameters:
            custom_parameters["production_capacity"] = custom_parameters["production_capacity_tpd"]
        
        # Validate parameters
        self.validate_parameters(template, custom_parameters)
        
        # Calculate adjusted costs
        adjusted_costs = self.calculate_adjusted_costs(template, custom_parameters)
        
        # Build plant configuration
        plant_config = {
            "template_info": {
                "id": template["id"],
                "name": template["name"],
                "description": template["description"],
                "technology": template["technology"]
            },
            "parameters": {
                **default_params,
                **custom_parameters
            },
            "financial_parameters": adjusted_costs,
            "validation": {
                "template_validated": True,
                "parameters_validated": True
            }
        }
        
        logger.info(f"Built plant configuration for template {template_id}")
        return plant_config 