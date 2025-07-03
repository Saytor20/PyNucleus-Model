"""
Enhanced Financial Analyzer for PyNucleus

Provides realistic financial analysis using pricing database and mock data.
Integrates with build function for accurate revenue and cost calculations.
"""

import json
import math
import functools
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from ..utils.logger import get_logger
from ..data.mock_data_manager import get_mock_data_manager

logger = get_logger(__name__)

class ReportSection(Enum):
    """Available report sections for customization."""
    REVENUE = "revenue"
    COSTS = "costs"
    PROFITABILITY = "profitability"
    RISK = "risk"
    SUSTAINABILITY = "sustainability"
    REGIONAL = "regional"
    MARKET_POSITIONING = "market_positioning"
    SENSITIVITY = "sensitivity"

@dataclass
class RiskThresholds:
    """Configurable risk thresholds for financial analysis."""
    revenue_low: float = 1e7  # $10M
    revenue_medium: float = 5e7  # $50M
    capital_high: float = 5e8  # $500M
    capital_medium: float = 2e8  # $200M
    operating_cost_high_ratio: float = 0.8
    operating_cost_medium_ratio: float = 0.6
    location_factor_high: float = 1.2
    product_price_low: float = 200
    product_price_medium: float = 500
    daily_production_small: float = 50
    daily_production_medium: float = 200
    political_stability_low: float = 0.5
    political_stability_medium: float = 0.7
    operating_hours_low: int = 7000

@dataclass
class SensitivityParameters:
    """Parameters for sensitivity analysis."""
    revenue_decrease: float = 0.10  # 10% decrease
    revenue_increase: float = 0.10  # 10% increase
    operating_cost_increase: float = 0.15  # 15% increase
    operating_cost_decrease: float = 0.10  # 10% decrease
    capital_cost_increase: float = 0.20  # 20% increase
    product_price_decrease: float = 0.15  # 15% decrease

@dataclass
class PlantConfiguration:
    """Validated plant configuration using dataclass for type safety."""
    template_id: int
    feedstock: str
    production_capacity: float  # tons/year
    plant_location: str
    operating_hours: int
    capital_cost: float
    operating_cost: float
    product_price: float
    template_info: Dict[str, Any] = field(default_factory=dict)
    location_adjustment: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for compatibility."""
        return {
            "parameters": {
                "feedstock": self.feedstock,
                "production_capacity": self.production_capacity,
                "plant_location": self.plant_location,
                "operating_hours": self.operating_hours
            },
            "financial_parameters": {
                "capital_cost": self.capital_cost,
                "operating_cost": self.operating_cost,
                "product_price": self.product_price
            },
            "template_info": self.template_info,
            "location_adjustment": self.location_adjustment
        }

class FinancialAnalyzer:
    """Enhanced financial analyzer with pricing database integration and advanced features."""
    
    def __init__(self, 
                 pricing_path: Optional[Path] = None, 
                 defaults: Optional[Dict] = None,
                 risk_thresholds: Optional[RiskThresholds] = None,
                 sensitivity_params: Optional[SensitivityParameters] = None):
        """
        Initialize the enhanced financial analyzer with dependency injection.
        
        Args:
            pricing_path: Optional path to pricing data file
            defaults: Optional default pricing data
            risk_thresholds: Optional custom risk thresholds
            sensitivity_params: Optional sensitivity analysis parameters
        """
        self.mock_data_manager = get_mock_data_manager()
        self.pricing_data = self._load_pricing_data(pricing_path, defaults)
        self.risk_thresholds = risk_thresholds or RiskThresholds()
        self.sensitivity_params = sensitivity_params or SensitivityParameters()
        
    @functools.lru_cache(maxsize=1)
    def _load_pricing_data(self, pricing_path: Optional[Path] = None, defaults: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Load product pricing data with caching and dependency injection.
        
        Args:
            pricing_path: Optional path to pricing data file
            defaults: Optional default pricing data
            
        Returns:
            Dictionary with pricing data
        """
        try:
            # Use provided defaults if available
            if defaults:
                logger.info(f"Using provided default pricing data with {len(defaults.get('prices', {}))} products")
                return defaults
            
            # Use provided path or default path
            if pricing_path is None:
                pricing_path = Path("data/product_prices.json")
            
            if pricing_path.exists():
                with open(pricing_path, 'r') as f:
                    data = json.load(f)
                logger.info(f"Loaded pricing data from {pricing_path} with {len(data.get('prices', {}))} products")
                return data
            else:
                logger.warning(f"Pricing data file not found at {pricing_path}, using defaults")
                return self._get_default_pricing()
        except Exception as e:
            logger.error(f"Error loading pricing data: {e}")
            return self._get_default_pricing()
    
    def _get_default_pricing(self) -> Dict[str, Any]:
        """Get default pricing structure."""
        return {
            "metadata": {
                "version": "1.0",
                "last_updated": datetime.now().isoformat(),
                "currency": "USD",
                "unit": "per_ton"
            },
            "prices": {
                "Methanol": 350,
                "Ammonia": 400,
                "Ethylene": 800,
                "Polyethylene": 1200,
                "Urea": 300,
                "Hydrogen": 1500,
                "Ethanol": 600,
                "Biodiesel": 900,
                "LNG": 450,
                "PET": 950
            },
            "market_indicators": {
                "crude_oil_brent": 85,
                "natural_gas_henry_hub": 3.5
            }
        }
    
    def get_product_price(self, product_name: str) -> float:
        """Get current price for a product."""
        prices = self.pricing_data.get("prices", {})
        
        # Try exact match first
        if product_name in prices:
            return prices[product_name]
        
        # Try partial matches
        for key, price in prices.items():
            if product_name.lower() in key.lower() or key.lower() in product_name.lower():
                return price
        
        # Default price if not found
        logger.warning(f"Product price not found for '{product_name}', using default $500/ton")
        return 500.0
    
    def validate_plant_config(self, plant_config: Union[Dict[str, Any], PlantConfiguration]) -> PlantConfiguration:
        """
        Validate and convert plant configuration to PlantConfiguration dataclass.
        
        Args:
            plant_config: Plant configuration as dict or PlantConfiguration
            
        Returns:
            Validated PlantConfiguration object
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        if isinstance(plant_config, PlantConfiguration):
            return plant_config
        
        if not isinstance(plant_config, dict):
            raise ValueError("Plant configuration must be a dictionary or PlantConfiguration object")
        
        # Extract parameters
        params = plant_config.get("parameters", {})
        financial_params = plant_config.get("financial_parameters", {})
        template_info = plant_config.get("template_info", {})
        location_adjustment = plant_config.get("location_adjustment", {})
        
        # Validate required fields
        required_params = ["feedstock", "production_capacity", "plant_location", "operating_hours"]
        required_financial = ["capital_cost", "operating_cost", "product_price"]
        
        missing_params = [p for p in required_params if p not in params]
        missing_financial = [p for p in required_financial if p not in financial_params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters: {missing_params}")
        if missing_financial:
            raise ValueError(f"Missing required financial parameters: {missing_financial}")
        
        # Validate data types and ranges
        if not isinstance(params["production_capacity"], (int, float)) or params["production_capacity"] <= 0:
            raise ValueError("Production capacity must be a positive number")
        
        if not isinstance(params["operating_hours"], int) or params["operating_hours"] <= 0:
            raise ValueError("Operating hours must be a positive integer")
        
        if not isinstance(financial_params["capital_cost"], (int, float)) or financial_params["capital_cost"] <= 0:
            raise ValueError("Capital cost must be a positive number")
        
        if not isinstance(financial_params["operating_cost"], (int, float)) or financial_params["operating_cost"] <= 0:
            raise ValueError("Operating cost must be a positive number")
        
        if not isinstance(financial_params["product_price"], (int, float)) or financial_params["product_price"] <= 0:
            raise ValueError("Product price must be a positive number")
        
        return PlantConfiguration(
            template_id=template_info.get("id", 0),
            feedstock=params["feedstock"],
            production_capacity=float(params["production_capacity"]),
            plant_location=params["plant_location"],
            operating_hours=int(params["operating_hours"]),
            capital_cost=float(financial_params["capital_cost"]),
            operating_cost=float(financial_params["operating_cost"]),
            product_price=float(financial_params["product_price"]),
            template_info=template_info,
            location_adjustment=location_adjustment
        )
    
    def calculate_capacity_scaling(self, base_capacity: float, target_capacity: float) -> float:
        """
        Calculate capacity scaling factor using 0.6 power law.
        
        Args:
            base_capacity: Base capacity (tons/day)
            target_capacity: Target capacity (tons/day)
            
        Returns:
            Scaling factor
        """
        if base_capacity <= 0 or target_capacity <= 0:
            return 1.0
        
        capacity_ratio = target_capacity / base_capacity
        return capacity_ratio ** 0.6
    
    def calculate_location_adjustment(self, location: str) -> float:
        """Calculate location-based cost adjustment."""
        location_factors = {
            "Texas, USA": 1.0,
            "Louisiana, USA": 1.05,
            "Alberta, Canada": 0.95,
            "Qatar": 0.90,
            "Australia": 1.10,
            "Europe": 1.15,
            "Asia": 0.85,
            "Middle East": 0.90,
            "Africa": 1.20,
            "South America": 1.15
        }
        
        # Try exact match first
        if location in location_factors:
            return location_factors[location]
        
        # Try partial matches
        for key, factor in location_factors.items():
            if location.lower() in key.lower() or key.lower() in location.lower():
                return factor
        
        return 1.0  # Default factor
    
    def calculate_annual_revenue(self, plant_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate annual revenue based on plant configuration.
        
        Args:
            plant_config: Plant configuration dictionary
            
        Returns:
            Dictionary with revenue calculations
        """
        try:
            # Extract parameters
            params = plant_config.get("parameters", {})
            template_info = plant_config.get("template_info", {})
            
            production_capacity = params.get("production_capacity", 0)
            operating_hours = params.get("operating_hours", 8000)
            plant_location = params.get("plant_location", "Texas, USA")
            
            # Get product name from template
            plant_name = template_info.get("name", "")
            product_name = self._extract_product_name(plant_name)
            
            # Get product price: prefer config, fallback to pricing data
            product_price = None
            if 'financial_parameters' in plant_config:
                product_price = plant_config['financial_parameters'].get('product_price', None)
            if not isinstance(product_price, (int, float)) or product_price <= 0:
                product_price = self.get_product_price(product_name)
                logger.warning(f"Falling back to pricing data for product price of '{product_name}': ${product_price}/ton")
            
            # Consistent daily and annual production (calendar day basis)
            annual_production = production_capacity  # tons/year
            daily_production = annual_production / 365  # tons/day (calendar)
            
            # Calculate revenue
            annual_revenue = annual_production * product_price
            daily_revenue = daily_production * product_price
            
            # Location adjustment: use the actual factor from plant config
            location_adjustment = plant_config.get("location_adjustment", {})
            location_factor = location_adjustment.get("factor", self.calculate_location_adjustment(plant_location))
            # (Revenue is not location-adjusted; OpEx is handled elsewhere)
            
            return {
                "daily_production_tons": daily_production,
                "annual_production_tons": annual_production,
                "product_name": product_name,
                "product_price_per_ton": product_price,
                "annual_revenue": annual_revenue,
                "daily_revenue": daily_revenue,
                "location_adjustment_factor": location_factor,
                "operating_hours_per_year": operating_hours
            }
        except Exception as e:
            logger.error(f"Error calculating annual revenue: {e}")
            return {
                "daily_production_tons": 0,
                "annual_production_tons": 0,
                "product_name": "Unknown",
                "product_price_per_ton": 0,
                "annual_revenue": 0,
                "daily_revenue": 0,
                "location_adjustment_factor": 1.0,
                "operating_hours_per_year": 8000
            }
    
    def _extract_product_name(self, plant_name: str) -> str:
        """Extract product name from plant name."""
        product_mapping = {
            "fertilizer": "Ammonia",
            "biofuel": "Methanol",
            "water treatment": "Clean Water",
            "cassava": "Cassava Products",
            "palm oil": "Palm Oil",
            "jatropha": "Biodiesel",
            "sugarcane": "Ethanol",
            "desalination": "Clean Water",
            "sodium hypochlorite": "Sodium Hypochlorite",
            "lubricant": "Lubricants",
            "caustic soda": "Caustic Soda",
            "soap": "Soap & Detergents",
            "paint": "Paint",
            "biopesticide": "Biopesticides",
            "plastic recycling": "PET Flakes",
            "phosphoric acid": "Phosphoric Acid",
            "hydrogen peroxide": "Hydrogen Peroxide",
            "gelatin": "Gelatin",
            "bitumen": "Bitumen Emulsion",
            "pyrethrum": "Pyrethrum Extract",
            "gas to liquids": "Synthetic Fuels",
            "cement": "Cement",
            "methanol": "Methanol",
            "ammonia": "Ammonia",
            "ethylene": "Ethylene",
            "polyethylene": "Polyethylene",
            "urea": "Urea",
            "hydrogen": "Hydrogen",
            "ethanol": "Ethanol",
            "biodiesel": "Biodiesel",
            "lng": "LNG",
            "pet": "PET",
            "propylene": "Propylene",
            "nitric acid": "Nitric Acid",
            "sulfuric acid": "Sulfuric Acid",
            "chlorine": "Chlorine",
            "sodium hydroxide": "Sodium Hydroxide"
        }
        
        plant_lower = plant_name.lower()
        for key, product in product_mapping.items():
            if key in plant_lower:
                return product
        
        return "Unknown"
    
    def calculate_financial_metrics(self, plant_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive financial metrics with African market context.
        
        Args:
            plant_config: Plant configuration dictionary
            
        Returns:
            Dictionary with comprehensive financial analysis
        """
        try:
            # Get revenue calculations
            revenue_data = self.calculate_annual_revenue(plant_config)
            
            # Get cost data
            financial_params = plant_config.get("financial_parameters", {})
            capital_cost = financial_params.get("capital_cost", 0)
            operating_cost = financial_params.get("operating_cost", 0)
            
            # Extract template information for enhanced analysis
            template_info = plant_config.get("template_info", {})
            params = plant_config.get("parameters", {})
            location_adjustment = plant_config.get("location_adjustment", {})
            
            # Operating cost is already location-adjusted from CLI, don't apply factor again
            # Just use the operating cost as provided in the plant config
            adjusted_operating_cost = operating_cost
            
            # Calculate key metrics
            annual_revenue = revenue_data["annual_revenue"]
            annual_profit = annual_revenue - adjusted_operating_cost
            
            # Calculate ratios
            profit_margin = (annual_profit / annual_revenue * 100) if annual_revenue > 0 else 0
            roi = (annual_profit / capital_cost * 100) if capital_cost > 0 else 0
            payback_period = capital_cost / annual_profit if annual_profit > 0 else float('inf')
            
            # Calculate per-unit metrics
            daily_production = revenue_data["daily_production_tons"]
            revenue_per_ton = revenue_data["product_price_per_ton"]
            operating_cost_per_ton = (adjusted_operating_cost / revenue_data["annual_production_tons"]) if revenue_data["annual_production_tons"] > 0 else 0
            profit_per_ton = revenue_per_ton - operating_cost_per_ton
            
            # Enhanced financial ratios
            capital_intensity = capital_cost / revenue_data["annual_production_tons"] if revenue_data["annual_production_tons"] > 0 else 0
            operating_margin = (annual_profit / annual_revenue * 100) if annual_revenue > 0 else 0
            asset_turnover = annual_revenue / capital_cost if capital_cost > 0 else 0
            
            # Risk assessment
            risk_factors = self._assess_financial_risks(plant_config, revenue_data)
            
            # Market positioning analysis
            market_position = self._analyze_market_position(plant_config, revenue_data, annual_profit)
            
            # Sustainability metrics
            sustainability_metrics = self._calculate_sustainability_metrics(plant_config, annual_profit)
            
            # Regional competitiveness analysis
            regional_analysis = self._analyze_regional_competitiveness(plant_config, revenue_data, operating_cost_per_ton)
            
            return {
                "revenue_analysis": revenue_data,
                "cost_analysis": {
                    "capital_cost": capital_cost,
                    "operating_cost": adjusted_operating_cost,
                    "operating_cost_per_ton": operating_cost_per_ton,
                    "capital_intensity_per_ton": capital_intensity
                },
                "profitability_metrics": {
                    "annual_revenue": annual_revenue,
                    "annual_profit": annual_profit,
                    "profit_margin_percent": profit_margin,
                    "roi_percent": roi,
                    "payback_period_years": payback_period,
                    "operating_margin_percent": operating_margin,
                    "asset_turnover": asset_turnover
                },
                "unit_economics": {
                    "revenue_per_ton": revenue_per_ton,
                    "operating_cost_per_ton": operating_cost_per_ton,
                    "profit_per_ton": profit_per_ton,
                    "daily_production_tons": daily_production,
                    "daily_revenue": daily_production * revenue_per_ton,
                    "daily_profit": daily_production * profit_per_ton
                },
                "market_positioning": market_position,
                "sustainability_metrics": sustainability_metrics,
                "regional_analysis": regional_analysis,
                "risk_assessment": risk_factors,
                "calculation_metadata": {
                    "calculation_timestamp": datetime.now().isoformat(),
                    "pricing_data_version": self.pricing_data.get("metadata", {}).get("version", "unknown"),
                    "location_adjustment_applied": revenue_data["location_adjustment_factor"],
                    "template_id": template_info.get("id"),
                    "technology": template_info.get("technology"),
                    "plant_location": params.get("plant_location")
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating financial metrics: {e}")
            return {
                "error": str(e),
                "calculation_timestamp": datetime.now().isoformat()
            }
    
    def _assess_financial_risks(self, plant_config: Dict[str, Any], revenue_data: Dict[str, float]) -> List[str]:
        """Assess comprehensive financial risks based on plant configuration and African market context."""
        risks = []
        params = plant_config.get("parameters", {})
        financial_params = plant_config.get("financial_parameters", {})
        template_info = plant_config.get("template_info", {})
        location_adjustment = plant_config.get("location_adjustment", {})
        annual_revenue = revenue_data["annual_revenue"]
        capital_cost = financial_params.get("capital_cost", 0)
        operating_cost = financial_params.get("operating_cost", 0)
        location_factor = location_adjustment.get("factor", 1.0)
        product_price = revenue_data["product_price_per_ton"]
        daily_production = revenue_data["daily_production_tons"]
        # --- Revenue and scale risks ---
        if annual_revenue < self.risk_thresholds.revenue_low:
            risks.append("Low revenue scale may limit profitability and financing options")
        elif annual_revenue < self.risk_thresholds.revenue_medium:
            risks.append("Moderate scale may face challenges with large competitors")
        # --- Cost structure risks ---
        if capital_cost > self.risk_thresholds.capital_high:
            risks.append("High capital cost increases financial risk and financing complexity")
        elif capital_cost > self.risk_thresholds.capital_medium:
            risks.append("Significant capital requirements may limit funding options")
        if operating_cost > annual_revenue * self.risk_thresholds.operating_cost_high_ratio:
            risks.append("High operating costs relative to revenue limit profit margins")
        elif operating_cost > annual_revenue * self.risk_thresholds.operating_cost_medium_ratio:
            risks.append("Moderate operating costs may be vulnerable to cost inflation")
        # --- Location-specific risks ---
        plant_location = params.get("plant_location", "")
        if location_factor > self.risk_thresholds.location_factor_high:
            risks.append(f"High location cost factor ({location_factor:.2f}x) increases project costs")
        # --- Market and product risks ---
        if product_price < self.risk_thresholds.product_price_low:
            risks.append("Low product price may be vulnerable to market fluctuations and competition")
        elif product_price < self.risk_thresholds.product_price_medium:
            risks.append("Moderate product price may face margin pressure in competitive markets")
        # --- Capacity and operational risks ---
        if daily_production < self.risk_thresholds.daily_production_small:
            risks.append("Small scale may limit economies of scale and operational efficiency")
        elif daily_production < self.risk_thresholds.daily_production_medium:
            risks.append("Moderate scale may face challenges with supply chain optimization")
        # --- Technology and process risks ---
        technology = template_info.get("technology", "")
        if "novel" in technology.lower() or "experimental" in technology.lower():
            risks.append("Novel technology may face operational and regulatory uncertainties")
        # --- Regional context risks ---
        regional_context = template_info.get("regional_context_africa", {})
        infrastructure_quality = regional_context.get("infrastructure_quality", "")
        political_stability = regional_context.get("political_stability_index", 0.5)
        regulatory_environment = regional_context.get("regulatory_environment", "")
        if infrastructure_quality == "Developing":
            risks.append("Developing infrastructure may increase operational costs and risks")
        elif infrastructure_quality == "Poor":
            risks.append("Poor infrastructure quality significantly increases project risks")
        if political_stability < self.risk_thresholds.political_stability_low:
            risks.append("Low political stability increases regulatory and operational risks")
        elif political_stability < self.risk_thresholds.political_stability_medium:
            risks.append("Moderate political stability may affect long-term project viability")
        if regulatory_environment == "Complex":
            risks.append("Complex regulatory environment may delay project implementation")
        elif regulatory_environment == "Uncertain":
            risks.append("Uncertain regulatory environment increases compliance risks")
        # --- Sustainability and environmental risks ---
        sustainability = template_info.get("sustainability_impact", {})
        co2_reduction = sustainability.get("co2_reduction_potential_tpy", 0)
        if co2_reduction < 0:  # Negative means CO2 emissions
            risks.append("CO2 emissions may face future regulatory restrictions and carbon pricing")
        # --- Feedstock risks ---
        feedstock = params.get("feedstock", "")
        if feedstock in ["Natural Gas", "Fossil Fuels"]:
            risks.append("Fossil fuel dependency may face future regulatory and market pressures")
        if feedstock in ["Brackish Water", "Seawater", "Groundwater"]:
            risks.append("Water feedstock may face scarcity, quality, or regulatory risks")
        # --- Operating hours risks ---
        operating_hours = params.get("operating_hours", 8000)
        if operating_hours < self.risk_thresholds.operating_hours_low:
            risks.append("Low operating hours may limit revenue potential and efficiency")
        # --- Currency and economic risks ---
        if plant_location in ["Nigeria", "Ghana", "Kenya", "Tanzania", "Ethiopia"]:
            risks.append("Local currency fluctuations may affect project economics")
        # --- Extreme ROI risks ---
        roi = 0
        try:
            roi = (annual_revenue - operating_cost) / capital_cost * 100 if capital_cost > 0 else 0
        except Exception:
            pass
        if roi > 100:
            risks.append("Extremely high ROI may indicate unrealistic assumptions or data errors")
        elif roi < 0:
            risks.append("Negative ROI indicates unprofitable project")
        return risks
    
    def calculate_sensitivity_analysis(self, plant_config: Union[Dict[str, Any], PlantConfiguration]) -> Dict[str, Any]:
        """
        Perform sensitivity analysis on key financial parameters.
        Uses calendar-day basis for daily production (tons/year ÷ 365 = tons/day).
        Location factor is only applied to OpEx.
        """
        try:
            validated_config = self.validate_plant_config(plant_config)
            base_metrics = self.calculate_financial_metrics(validated_config.to_dict())
            if "error" in base_metrics:
                return {"error": base_metrics["error"]}
            
            # Get the original (pre-location-adjusted) operating cost for sensitivity calculations
            original_operating_cost = validated_config.operating_cost
            base_operating_cost = base_metrics["cost_analysis"]["operating_cost"]  # Keep for reference
            base_capital_cost = base_metrics["cost_analysis"]["capital_cost"]
            base_annual_revenue = base_metrics["revenue_analysis"]["annual_revenue"]
            base_product_price = base_metrics["revenue_analysis"]["product_price_per_ton"]
            base_production_capacity = base_metrics["revenue_analysis"]["annual_production_tons"]
            base_location_factor = base_metrics["revenue_analysis"]["location_adjustment_factor"]
            sensitivity_scenarios = {}
            # Revenue sensitivity: recompute revenue, profit, and ROI using correct formula
            for scenario_name, factor in {
                "revenue_decrease": 1 - self.sensitivity_params.revenue_decrease,
                "revenue_increase": 1 + self.sensitivity_params.revenue_increase
            }.items():
                modified_config = validated_config.to_dict()
                # Adjust product price to simulate revenue change
                modified_config["financial_parameters"]["product_price"] = base_product_price * factor
                scenario_metrics = self.calculate_financial_metrics(modified_config)
                if "error" not in scenario_metrics:
                    # Recompute ROI using the correct formula
                    scenario_revenue = base_annual_revenue * factor
                    scenario_profit = scenario_revenue - base_operating_cost
                    scenario_roi = (scenario_profit / base_capital_cost * 100) if base_capital_cost > 0 else 0
                    sensitivity_scenarios[scenario_name] = {
                        "roi_percent": scenario_roi,
                        "profit_margin_percent": scenario_metrics["profitability_metrics"]["profit_margin_percent"],
                        "payback_period_years": scenario_metrics["profitability_metrics"]["payback_period_years"],
                        "annual_profit": scenario_profit
                    }
            # Operating cost sensitivity
            for scenario_name, factor in {
                "operating_cost_increase": 1 + self.sensitivity_params.operating_cost_increase,
                "operating_cost_decrease": 1 - self.sensitivity_params.operating_cost_decrease
            }.items():
                # Use the exact baseline values from the base case
                base_op_ex = base_metrics["cost_analysis"]["operating_cost"]
                base_revenue = base_metrics["revenue_analysis"]["annual_revenue"]
                base_cap_ex = base_metrics["cost_analysis"]["capital_cost"]
                
                # Apply the factor to the baseline OpEx
                new_op_ex = base_op_ex * factor
                
                # Calculate new profit and ROI using the exact formula
                new_profit = base_revenue - new_op_ex
                new_roi = (new_profit / base_cap_ex * 100) if base_cap_ex > 0 else 0
                
                sensitivity_scenarios[scenario_name] = {
                    "roi_percent": new_roi,
                    "profit_margin_percent": (new_profit / base_revenue * 100) if base_revenue > 0 else 0,
                    "payback_period_years": base_cap_ex / new_profit if new_profit > 0 else float('inf'),
                    "annual_profit": new_profit
                }
            # Capital cost sensitivity
            modified_config = validated_config.to_dict()
            modified_config["financial_parameters"]["capital_cost"] = base_capital_cost * (1 + self.sensitivity_params.capital_cost_increase)
            scenario_metrics = self.calculate_financial_metrics(modified_config)
            if "error" not in scenario_metrics:
                sensitivity_scenarios["capital_cost_increase"] = {
                    "roi_percent": scenario_metrics["profitability_metrics"]["roi_percent"],
                    "profit_margin_percent": scenario_metrics["profitability_metrics"]["profit_margin_percent"],
                    "payback_period_years": scenario_metrics["profitability_metrics"]["payback_period_years"],
                    "annual_profit": scenario_metrics["profitability_metrics"]["annual_profit"]
                }
            # Product price decrease
            modified_config = validated_config.to_dict()
            # Apply price decrease to product price
            modified_config["financial_parameters"]["product_price"] = base_product_price * (1 - self.sensitivity_params.product_price_decrease)
            scenario_metrics = self.calculate_financial_metrics(modified_config)
            if "error" not in scenario_metrics:
                # Recompute ROI using the correct formula with new price
                scenario_annual_revenue = scenario_metrics["revenue_analysis"]["annual_revenue"]
                scenario_profit = scenario_annual_revenue - base_operating_cost
                scenario_roi = (scenario_profit / base_capital_cost * 100) if base_capital_cost > 0 else 0
                sensitivity_scenarios["product_price_decrease"] = {
                    "roi_percent": scenario_roi,
                    "profit_margin_percent": scenario_metrics["profitability_metrics"]["profit_margin_percent"],
                    "payback_period_years": scenario_metrics["profitability_metrics"]["payback_period_years"],
                    "annual_profit": scenario_profit
                }
            return {
                "base_metrics": base_metrics["profitability_metrics"],
                "sensitivity_scenarios": sensitivity_scenarios,
                "risk_scores": {},
                "sensitivity_parameters": {
                    "revenue_decrease": self.sensitivity_params.revenue_decrease,
                    "revenue_increase": self.sensitivity_params.revenue_increase,
                    "operating_cost_increase": self.sensitivity_params.operating_cost_increase,
                    "operating_cost_decrease": self.sensitivity_params.operating_cost_decrease,
                    "capital_cost_increase": self.sensitivity_params.capital_cost_increase,
                    "product_price_decrease": self.sensitivity_params.product_price_decrease
                }
            }
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {"error": str(e)}
    
    def analyze_multiple_plants(self, plant_configs: List[Union[Dict[str, Any], PlantConfiguration]]) -> Dict[str, Any]:
        """
        Analyze multiple plant configurations and provide comparative analysis.
        
        Args:
            plant_configs: List of plant configurations
            
        Returns:
            Dictionary with comparative analysis results
        """
        try:
            results = []
            summary_data = []
            
            for i, config in enumerate(plant_configs):
                # Validate configuration
                validated_config = self.validate_plant_config(config)
                
                # Calculate metrics
                metrics = self.calculate_financial_metrics(validated_config.to_dict())
                
                if "error" not in metrics:
                    plant_result = {
                        "plant_id": i + 1,
                        "template_name": validated_config.template_info.get("name", f"Plant {i + 1}"),
                        "technology": validated_config.template_info.get("technology", "Unknown"),
                        "location": validated_config.plant_location,
                        "capacity_tpy": validated_config.production_capacity,
                        "metrics": metrics
                    }
                    results.append(plant_result)
                    
                    # Prepare summary data for DataFrame
                    summary_data.append({
                        "Plant ID": i + 1,
                        "Template": validated_config.template_info.get("name", f"Plant {i + 1}"),
                        "Technology": validated_config.template_info.get("technology", "Unknown"),
                        "Location": validated_config.plant_location,
                        "Capacity (tpy)": validated_config.production_capacity,
                        "Capital Cost ($)": metrics["cost_analysis"]["capital_cost"],
                        "Operating Cost ($/year)": metrics["cost_analysis"]["operating_cost"],
                        "Annual Revenue ($)": metrics["revenue_analysis"]["annual_revenue"],
                        "Annual Profit ($)": metrics["profitability_metrics"]["annual_profit"],
                        "ROI (%)": metrics["profitability_metrics"]["roi_percent"],
                        "Profit Margin (%)": metrics["profitability_metrics"]["profit_margin_percent"],
                        "Payback Period (years)": metrics["profitability_metrics"]["payback_period_years"],
                        "Risk Count": len(metrics["risk_assessment"]),
                        "Sustainability Score": metrics["sustainability_metrics"]["social_impact_score"]
                    })
            
            # Create comparative analysis
            if results:
                # Find best performers
                best_roi = max(results, key=lambda x: x["metrics"]["profitability_metrics"]["roi_percent"])
                best_margin = max(results, key=lambda x: x["metrics"]["profitability_metrics"]["profit_margin_percent"])
                best_payback = min(results, key=lambda x: x["metrics"]["profitability_metrics"]["payback_period_years"])
                lowest_risk = min(results, key=lambda x: len(x["metrics"]["risk_assessment"]))
                best_sustainability = max(results, key=lambda x: x["metrics"]["sustainability_metrics"]["social_impact_score"])
                
                comparative_analysis = {
                    "best_performers": {
                        "highest_roi": {
                            "plant_id": best_roi["plant_id"],
                            "template": best_roi["template_name"],
                            "roi_percent": best_roi["metrics"]["profitability_metrics"]["roi_percent"]
                        },
                        "highest_margin": {
                            "plant_id": best_margin["plant_id"],
                            "template": best_margin["template_name"],
                            "margin_percent": best_margin["metrics"]["profitability_metrics"]["profit_margin_percent"]
                        },
                        "fastest_payback": {
                            "plant_id": best_payback["plant_id"],
                            "template": best_payback["template_name"],
                            "payback_years": best_payback["metrics"]["profitability_metrics"]["payback_period_years"]
                        },
                        "lowest_risk": {
                            "plant_id": lowest_risk["plant_id"],
                            "template": lowest_risk["template_name"],
                            "risk_count": len(lowest_risk["metrics"]["risk_assessment"])
                        },
                        "best_sustainability": {
                            "plant_id": best_sustainability["plant_id"],
                            "template": best_sustainability["template_name"],
                            "sustainability_score": best_sustainability["metrics"]["sustainability_metrics"]["social_impact_score"]
                        }
                    },
                    "summary_statistics": {
                        "total_plants": len(results),
                        "average_roi": sum(r["metrics"]["profitability_metrics"]["roi_percent"] for r in results) / len(results),
                        "average_margin": sum(r["metrics"]["profitability_metrics"]["profit_margin_percent"] for r in results) / len(results),
                        "average_payback": sum(r["metrics"]["profitability_metrics"]["payback_period_years"] for r in results) / len(results),
                        "average_risk_count": sum(len(r["metrics"]["risk_assessment"]) for r in results) / len(results)
                    }
                }
            else:
                comparative_analysis = {}
            
            return {
                "individual_results": results,
                "comparative_analysis": comparative_analysis,
                "summary_data": summary_data,
                "pandas_available": PANDAS_AVAILABLE
            }
            
        except Exception as e:
            logger.error(f"Error in batch analysis: {e}")
            return {"error": str(e)}
    
    def generate_comparative_report(self, plant_configs: List[Union[Dict[str, Any], PlantConfiguration]], 
                                  output_format: str = "text") -> Union[str, Dict[str, Any]]:
        """
        Generate comparative report for multiple plants.
        
        Args:
            plant_configs: List of plant configurations
            output_format: Output format ("text", "json", "dataframe")
            
        Returns:
            Comparative report in specified format
        """
        try:
            analysis = self.analyze_multiple_plants(plant_configs)
            
            if "error" in analysis:
                return f"❌ Comparative analysis failed: {analysis['error']}"
            
            if output_format == "json":
                return analysis
            
            if output_format == "dataframe" and PANDAS_AVAILABLE:
                df = pd.DataFrame(analysis["summary_data"])
                return df
            
            # Generate text report
            report = f"""
📊 COMPARATIVE PLANT ANALYSIS REPORT
{'='*60}
Total Plants Analyzed: {len(analysis['individual_results'])}
{'='*60}

🏆 BEST PERFORMERS
"""
            
            best = analysis["comparative_analysis"]["best_performers"]
            report += f"""
• Highest ROI: {best['highest_roi']['template']} (Plant {best['highest_roi']['plant_id']}) - {best['highest_roi']['roi_percent']:.2f}%
• Highest Margin: {best['highest_margin']['template']} (Plant {best['highest_margin']['plant_id']}) - {best['highest_margin']['margin_percent']:.2f}%
• Fastest Payback: {best['fastest_payback']['template']} (Plant {best['fastest_payback']['plant_id']}) - {best['fastest_payback']['payback_years']:.2f} years
• Lowest Risk: {best['lowest_risk']['template']} (Plant {best['lowest_risk']['plant_id']}) - {best['lowest_risk']['risk_count']} risks
• Best Sustainability: {best['best_sustainability']['template']} (Plant {best['best_sustainability']['plant_id']}) - Score: {best['best_sustainability']['sustainability_score']}/100

📈 SUMMARY STATISTICS
• Average ROI: {analysis['comparative_analysis']['summary_statistics']['average_roi']:.2f}%
• Average Profit Margin: {analysis['comparative_analysis']['summary_statistics']['average_margin']:.2f}%
• Average Payback Period: {analysis['comparative_analysis']['summary_statistics']['average_payback']:.2f} years
• Average Risk Count: {analysis['comparative_analysis']['summary_statistics']['average_risk_count']:.2f}

📋 INDIVIDUAL PLANT RESULTS
"""
            
            for result in analysis["individual_results"]:
                metrics = result["metrics"]["profitability_metrics"]
                report += f"""
Plant {result['plant_id']}: {result['template_name']}
• Technology: {result['technology']}
• Location: {result['location']}
• Capacity: {result['capacity_tpy']:,.2f} tons/year
• ROI: {metrics['roi_percent']:.2f}%
• Profit Margin: {metrics['profit_margin_percent']:.2f}%
• Payback: {metrics['payback_period_years']:.2f} years
• Risks: {len(result['metrics']['risk_assessment'])}
"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating comparative report: {e}")
            return f"❌ Error generating comparative report: {str(e)}"
    
    def _analyze_market_position(self, plant_config: Dict[str, Any], revenue_data: Dict[str, float], annual_profit: float) -> Dict[str, Any]:
        """Analyze market positioning and competitive landscape."""
        template_info = plant_config.get("template_info", {})
        params = plant_config.get("parameters", {})
        
        annual_revenue = revenue_data["annual_revenue"]
        daily_production = revenue_data["daily_production_tons"]
        product_price = revenue_data["product_price_per_ton"]
        
        # Market scale analysis
        if annual_revenue > 100000000:
            market_scale = "Large"
            scale_advantage = "High"
        elif annual_revenue > 50000000:
            market_scale = "Medium"
            scale_advantage = "Moderate"
        else:
            market_scale = "Small"
            scale_advantage = "Low"
        
        # Technology positioning
        technology = template_info.get("technology", "")
        if any(word in technology.lower() for word in ["modular", "advanced", "novel"]):
            tech_position = "Innovative"
        elif any(word in technology.lower() for word in ["traditional", "conventional"]):
            tech_position = "Traditional"
        else:
            tech_position = "Standard"
        
        # Product positioning
        if product_price > 1000:
            product_position = "Premium"
        elif product_price > 500:
            product_position = "Mid-market"
        else:
            product_position = "Commodity"
        
        # Competitive advantages
        advantages = []
        if daily_production > 200:
            advantages.append("Economies of scale")
        if "modular" in technology.lower():
            advantages.append("Modular design flexibility")
        if template_info.get("sustainability_impact", {}).get("resource_circularity") == "High":
            advantages.append("Sustainable operations")
        
        return {
            "market_scale": market_scale,
            "scale_advantage": scale_advantage,
            "technology_position": tech_position,
            "product_position": product_position,
            "competitive_advantages": advantages,
            "annual_revenue_category": "Large" if annual_revenue > 100000000 else "Medium" if annual_revenue > 50000000 else "Small"
        }
    
    def _calculate_sustainability_metrics(self, plant_config: Dict[str, Any], annual_profit: float) -> Dict[str, Any]:
        """Calculate sustainability and social impact metrics."""
        template_info = plant_config.get("template_info", {})
        sustainability = template_info.get("sustainability_impact", {})
        params = plant_config.get("parameters", {})
        
        # Employment metrics
        employment = sustainability.get("community_employment", 0)
        employment_intensity = employment / (annual_profit / 1000000) if annual_profit > 0 else 0  # jobs per $1M profit
        
        # Environmental metrics
        co2_impact = sustainability.get("co2_reduction_potential_tpy", 0)
        circularity = sustainability.get("resource_circularity", "Unknown")
        
        # Social impact scoring
        social_impact_score = 0
        if employment > 50:
            social_impact_score += 25
        if circularity == "High":
            social_impact_score += 25
        if co2_impact > 0:
            social_impact_score += 25
        if "renewable" in params.get("feedstock", "").lower():
            social_impact_score += 25
        
        return {
            "employment_metrics": {
                "total_jobs": employment,
                "employment_intensity": employment_intensity,
                "job_creation_rating": "High" if employment > 100 else "Medium" if employment > 50 else "Low"
            },
            "environmental_metrics": {
                "co2_impact_tons_per_year": co2_impact,
                "resource_circularity": circularity,
                "environmental_rating": "Positive" if co2_impact > 0 else "Neutral" if co2_impact == 0 else "Negative"
            },
            "social_impact_score": social_impact_score,
            "sustainability_rating": "Excellent" if social_impact_score >= 75 else "Good" if social_impact_score >= 50 else "Moderate" if social_impact_score >= 25 else "Low"
        }
    
    def _analyze_regional_competitiveness(self, plant_config: Dict[str, Any], revenue_data: Dict[str, float], operating_cost_per_ton: float) -> Dict[str, Any]:
        """Analyze regional competitiveness and market positioning."""
        template_info = plant_config.get("template_info", {})
        params = plant_config.get("parameters", {})
        location_adjustment = plant_config.get("location_adjustment", {})
        
        plant_location = params.get("plant_location", "")
        regional_context = template_info.get("regional_context_africa", {})
        location_factor = location_adjustment.get("factor", 1.0)
        
        # Regional infrastructure assessment
        infrastructure_quality = regional_context.get("infrastructure_quality", "")
        if infrastructure_quality == "Good":
            infrastructure_score = 80
        elif infrastructure_quality == "Moderate":
            infrastructure_score = 60
        elif infrastructure_quality == "Developing":
            infrastructure_score = 40
        else:
            infrastructure_score = 20
        
        # Labor availability assessment
        labor_availability = regional_context.get("skilled_labor_availability", "")
        if labor_availability == "Good":
            labor_score = 80
        elif labor_availability == "Moderate":
            labor_score = 60
        else:
            labor_score = 40
        
        # Political stability assessment
        political_stability = regional_context.get("political_stability_index", 0.5)
        political_score = political_stability * 100
        
        # Cost competitiveness
        revenue_per_ton = revenue_data["product_price_per_ton"]
        cost_competitiveness = "High" if operating_cost_per_ton < revenue_per_ton * 0.5 else "Moderate" if operating_cost_per_ton < revenue_per_ton * 0.7 else "Low"
        
        # Regional advantages
        advantages = []
        if location_factor <= 1.1:
            advantages.append("Favorable cost structure")
        if political_stability > 0.7:
            advantages.append("Political stability")
        if infrastructure_quality in ["Good", "Moderate"]:
            advantages.append("Adequate infrastructure")
        
        # Regional challenges
        challenges = []
        if location_factor > 1.2:
            challenges.append("High cost environment")
        if political_stability < 0.6:
            challenges.append("Political uncertainty")
        if infrastructure_quality == "Developing":
            challenges.append("Infrastructure limitations")
        
        return {
            "regional_scores": {
                "infrastructure_score": infrastructure_score,
                "labor_score": labor_score,
                "political_score": political_score,
                "overall_regional_score": (infrastructure_score + labor_score + political_score) / 3
            },
            "cost_competitiveness": cost_competitiveness,
            "regional_advantages": advantages,
            "regional_challenges": challenges,
            "location_factor": location_factor,
            "regional_rating": "Excellent" if (infrastructure_score + labor_score + political_score) / 3 >= 75 else "Good" if (infrastructure_score + labor_score + political_score) / 3 >= 60 else "Moderate" if (infrastructure_score + labor_score + political_score) / 3 >= 40 else "Challenging"
        }
    
    def generate_financial_report(self, plant_config: Union[Dict[str, Any], PlantConfiguration], 
                                enabled_sections: Optional[List[ReportSection]] = None) -> str:
        """
        Generate a comprehensive human-readable financial report with customizable sections.
        
        Args:
            plant_config: Plant configuration dictionary or PlantConfiguration object
            enabled_sections: List of report sections to include (None = all sections)
            
        Returns:
            Formatted financial report string
        """
        try:
            # Validate plant configuration
            validated_config = self.validate_plant_config(plant_config)
            metrics = self.calculate_financial_metrics(validated_config.to_dict())
            
            if "error" in metrics:
                return f"❌ Financial analysis failed: {metrics['error']}"
            
            revenue = metrics["revenue_analysis"]
            costs = metrics["cost_analysis"]
            profitability = metrics["profitability_metrics"]
            unit_economics = metrics["unit_economics"]
            risks = metrics["risk_assessment"]
            
            # Extract template information
            template_info = validated_config.template_info
            params = validated_config.to_dict()["parameters"]
            location_adjustment = validated_config.location_adjustment
            
            # Get sustainability and regional context
            sustainability = template_info.get("sustainability_impact", {})
            regional_context = template_info.get("regional_context_africa", {})
            
            # Determine which sections to include
            if enabled_sections is None:
                enabled_sections = list(ReportSection)
            
            report = f"""💰 COMPREHENSIVE FINANCIAL ANALYSIS REPORT
{'='*60}
🌍 African Market Focus | {template_info.get('name', 'Unknown Plant')}
{'='*60}"""

            # Build report sections based on enabled sections
            if ReportSection.REVENUE in enabled_sections:
                report += f"""
📊 PRODUCTION & REVENUE ANALYSIS
• Plant Technology: {template_info.get('technology', 'Unknown')}
• Feedstock: {params.get('feedstock', 'Unknown')}
• Daily Production: {revenue['daily_production_tons']:,.2f} tons/day
• Annual Production: {revenue['annual_production_tons']:,.2f} tons/year
• Product: {revenue['product_name']}
• Market Price: ${revenue['product_price_per_ton']:,.2f}/ton
• Annual Revenue: ${revenue['annual_revenue']:,.2f}
• Daily Revenue: ${revenue['daily_revenue']:,.2f}
• Location Adjustment Factor: {revenue['location_adjustment_factor']:.2f}x"""

            if ReportSection.COSTS in enabled_sections:
                # Get the original (pre-location-adjusted) operating cost for sensitivity calculations
                original_operating_cost = validated_config.operating_cost
                base_operating_cost = costs['operating_cost']  # Keep for reference
                base_capital_cost = costs['capital_cost']
                base_annual_revenue = revenue['annual_revenue']
                base_product_price = revenue['product_price_per_ton']
                base_production_capacity = revenue['annual_production_tons']
                base_location_factor = revenue['location_adjustment_factor']
                
                # Get the location factor from the plant config
                location_factor = plant_config.get('location_adjustment', {}).get('factor', base_location_factor)
                
                # Determine if CapEx was actually location-adjusted
                cap_ex_adjusted = location_factor != 1.0
                cap_ex_label = f" (location-adjusted by {location_factor:.2f}x)" if cap_ex_adjusted else " (not location-adjusted)"
                op_ex_label = f" (location-adjusted by {location_factor:.2f}x)" if location_factor != 1.0 else ""
                
                report += f"""

💸 COSTS & INVESTMENT BREAKDOWN
• Capital Cost: ${costs['capital_cost']:,.2f}{cap_ex_label}
• Annual Operating Cost: ${costs['operating_cost']:,.2f}{op_ex_label}
• Operating Cost per Ton: ${costs['operating_cost_per_ton']:,.2f}
• Operating Hours: {params.get('operating_hours', 0):,} hours/year"""

            if ReportSection.PROFITABILITY in enabled_sections:
                report += f"""

📈 PROFITABILITY METRICS
• Annual Profit: ${profitability['annual_profit']:,.2f}
• Profit Margin: {profitability['profit_margin_percent']:.2f}%
• Return on Investment: {profitability['roi_percent']:.2f}%
• Payback Period: {"Never" if profitability['payback_period_years'] == float('inf') else f"{profitability['payback_period_years']:.2f} years"}

⚖️ UNIT ECONOMICS
• Revenue per Ton: ${unit_economics['revenue_per_ton']:,.2f}
• Operating Cost per Ton: ${unit_economics['operating_cost_per_ton']:,.2f}
• Profit per Ton: ${unit_economics['profit_per_ton']:,.2f}
• Daily Revenue: ${revenue['daily_revenue']:,.2f}"""

            if ReportSection.SUSTAINABILITY in enabled_sections:
                report += f"""

🌍 REGIONAL CONTEXT & SUSTAINABILITY
• Plant Location: {params.get('plant_location', 'Unknown')}
• Infrastructure Quality: {regional_context.get('infrastructure_quality', 'Unknown')}
• Labor Availability: {regional_context.get('skilled_labor_availability', 'Unknown')}
• Regulatory Environment: {regional_context.get('regulatory_environment', 'Unknown')}
• Political Stability Index: {regional_context.get('political_stability_index', 0):.2f}
• Community Employment: {sustainability.get('community_employment', 0)} jobs
• Resource Circularity: {sustainability.get('resource_circularity', 'Unknown')}
• CO₂ Impact: {sustainability.get('co2_reduction_potential_tpy', 0):+,.2f} tons/year"""

            if ReportSection.RISK in enabled_sections:
                report += f"""

⚠️ COMPREHENSIVE RISK ASSESSMENT"""
            
            if ReportSection.RISK in enabled_sections and risks:
                # Categorize risks
                financial_risks = []
                operational_risks = []
                market_risks = []
                regulatory_risks = []
                
                for risk in risks:
                    risk_lower = risk.lower()
                    if any(word in risk_lower for word in ['revenue', 'profit', 'cost', 'capital', 'roi', 'margin']):
                        financial_risks.append(risk)
                    elif any(word in risk_lower for word in ['infrastructure', 'operational', 'efficiency', 'scale']):
                        operational_risks.append(risk)
                    elif any(word in risk_lower for word in ['market', 'competition', 'price', 'currency']):
                        market_risks.append(risk)
                    elif any(word in risk_lower for word in ['regulatory', 'political', 'compliance', 'environmental']):
                        regulatory_risks.append(risk)
                    else:
                        financial_risks.append(risk)
                
                if financial_risks:
                    report += "💰 Financial Risks:\n"
                    for i, risk in enumerate(financial_risks, 1):
                        report += f"   {i}. {risk}\n"
                
                if operational_risks:
                    report += "\n⚙️ Operational Risks:\n"
                    for i, risk in enumerate(operational_risks, 1):
                        report += f"   {i}. {risk}\n"
                
                if market_risks:
                    report += "\n📈 Market Risks:\n"
                    for i, risk in enumerate(market_risks, 1):
                        report += f"   {i}. {risk}\n"
                
                if regulatory_risks:
                    report += "\n📋 Regulatory Risks:\n"
                    for i, risk in enumerate(regulatory_risks, 1):
                        report += f"   {i}. {risk}\n"
            else:
                report += "• No significant risks identified\n"
            
            if ReportSection.MARKET_POSITIONING in enabled_sections:
                report += f"""

🎯 MARKET POSITIONING ANALYSIS
• Revenue Scale: {'Large' if revenue['annual_revenue'] > 100000000 else 'Medium' if revenue['annual_revenue'] > 50000000 else 'Small'}
• Cost Competitiveness: {'High' if costs['operating_cost_per_ton'] < unit_economics['revenue_per_ton'] * 0.5 else 'Moderate' if costs['operating_cost_per_ton'] < unit_economics['revenue_per_ton'] * 0.7 else 'Low'}
• Profitability Profile: {'Excellent' if profitability['profit_margin_percent'] > 25 else 'Good' if profitability['profit_margin_percent'] > 15 else 'Moderate' if profitability['profit_margin_percent'] > 8 else 'Challenging'}
• Investment Attractiveness: {'High' if profitability['roi_percent'] > 20 else 'Good' if profitability['roi_percent'] > 15 else 'Moderate' if profitability['roi_percent'] > 10 else 'Low'}"""

            if ReportSection.SENSITIVITY in enabled_sections:
                sensitivity_analysis = self.calculate_sensitivity_analysis(validated_config)
                if "error" not in sensitivity_analysis:
                    scenarios = sensitivity_analysis['sensitivity_scenarios']
                    
                    # Helper function to safely get ROI value
                    def get_roi_value(scenario_name):
                        scenario = scenarios.get(scenario_name, {})
                        roi = scenario.get('roi_percent', 'N/A')
                        return f"{roi:.2f}%" if isinstance(roi, (int, float)) else roi
                    
                    report += f"""

📊 SENSITIVITY ANALYSIS
• Base ROI: {sensitivity_analysis['base_metrics']['roi_percent']:.2f}%
• Revenue -10%: {get_roi_value('revenue_decrease')}
• Revenue +10%: {get_roi_value('revenue_increase')}
• Operating Cost +15%: {get_roi_value('operating_cost_increase')}
• Product Price -15%: {get_roi_value('product_price_decrease')}
"""

            report += f"""

📋 ANALYSIS METADATA
• Calculation Date: {metrics['calculation_metadata']['calculation_timestamp']}
• Pricing Data Version: {metrics['calculation_metadata']['pricing_data_version']}
• Location Factor Applied: {plant_config.get('location_adjustment', {}).get('factor', metrics['calculation_metadata']['location_adjustment_applied']):.2f}
• Template Version: {template_info.get('id', 'Unknown')}"""
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating financial report: {e}")
            return f"❌ Error generating financial report: {str(e)}"
    
    def estimate_capacity_for_revenue(self, target_revenue: float, product_name: str, 
                                    location: str = "Texas, USA", operating_hours: int = 8000) -> Dict[str, Any]:
        """
        Estimate required capacity to achieve target revenue.
        
        Args:
            target_revenue: Target annual revenue in USD
            product_name: Product name
            location: Plant location
            operating_hours: Operating hours per year
            
        Returns:
            Dictionary with capacity estimates
        """
        try:
            # Get product price
            product_price = self.get_product_price(product_name)
            
            # Calculate location adjustment
            location_factor = self.calculate_location_adjustment(location)
            adjusted_price = product_price * location_factor
            
            # Calculate required annual production
            required_annual_production = target_revenue / adjusted_price
            
            # Calculate daily capacity
            daily_capacity = required_annual_production / (operating_hours / 24) / 365
            
            return {
                "target_revenue": target_revenue,
                "product_name": product_name,
                "product_price_per_ton": product_price,
                "location_adjustment_factor": location_factor,
                "adjusted_price_per_ton": adjusted_price,
                "required_annual_production_tons": required_annual_production,
                "required_daily_capacity_tons": daily_capacity,
                "operating_hours_per_year": operating_hours
            }
            
        except Exception as e:
            logger.error(f"Error estimating capacity: {e}")
            return {
                "error": str(e),
                "target_revenue": target_revenue,
                "product_name": product_name
            }
    
    def auto_populate_from_mock_data(self, template_id: int, **overrides) -> Dict[str, Any]:
        """
        Auto-populate plant configuration from mock data templates.
        
        Args:
            template_id: Template ID from mock data
            **overrides: Optional parameter overrides
            
        Returns:
            Complete plant configuration dictionary
        """
        try:
            # Get template from mock data
            templates = self.mock_data_manager.get_all_plant_templates()
            template = None
            
            for t in templates:
                if t.get("id") == template_id:
                    template = t
                    break
            
            if not template:
                raise ValueError(f"Template with ID {template_id} not found in mock data")
            
            # Extract base parameters
            params = template.get("parameters", {})
            financial_params = template.get("financial_parameters", {})
            
            # Apply overrides
            for key, value in overrides.items():
                if key in params:
                    params[key] = value
                elif key in financial_params:
                    financial_params[key] = value
                else:
                    # Try to guess where it belongs
                    if key in ["production_capacity", "feedstock", "plant_location", "operating_hours"]:
                        params[key] = value
                    elif key in ["capital_cost", "operating_cost", "product_price"]:
                        financial_params[key] = value
            
            # Ensure required fields are present
            required_params = {
                "feedstock": "Natural Gas",
                "production_capacity": 100000,  # 100k tons/year
                "plant_location": "Texas, USA",
                "operating_hours": 8000
            }
            
            required_financial = {
                "capital_cost": 50000000,  # $50M
                "operating_cost": 20000000,  # $20M/year
                "product_price": 500  # $500/ton
            }
            
            # Fill missing required parameters
            for key, default_value in required_params.items():
                if key not in params:
                    params[key] = default_value
            
            for key, default_value in required_financial.items():
                if key not in financial_params:
                    financial_params[key] = default_value
            
            return {
                "parameters": params,
                "financial_parameters": financial_params,
                "template_info": template,
                "location_adjustment": template.get("location_adjustment", {"factor": 1.0})
            }
            
        except Exception as e:
            logger.error(f"Error auto-populating from mock data: {e}")
            return {"error": str(e)}
    
    def analyze_all_mock_plants(self, **overrides) -> Dict[str, Any]:
        """
        Analyze all plants from mock data with optional parameter overrides.
        
        Args:
            **overrides: Optional parameter overrides to apply to all plants
            
        Returns:
            Comparative analysis of all mock plants
        """
        try:
            templates = self.mock_data_manager.get_all_plant_templates()
            plant_configs = []
            
            for template in templates:
                template_id = template.get("id")
                if template_id is not None:
                    config = self.auto_populate_from_mock_data(template_id, **overrides)
                    if "error" not in config:
                        plant_configs.append(config)
            
            if not plant_configs:
                return {"error": "No valid plant configurations found in mock data"}
            
            return self.analyze_multiple_plants(plant_configs)
            
        except Exception as e:
            logger.error(f"Error analyzing all mock plants: {e}")
            return {"error": str(e)}