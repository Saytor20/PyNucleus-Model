"""
Enhanced Financial Analyzer for PyNucleus

Provides realistic financial analysis using pricing database and mock data.
Integrates with build function for accurate revenue and cost calculations.
"""

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from ..utils.logger import get_logger
from ..data.mock_data_manager import get_mock_data_manager

logger = get_logger(__name__)

class EnhancedFinancialAnalyzer:
    """Enhanced financial analyzer with pricing database integration."""
    
    def __init__(self):
        """Initialize the enhanced financial analyzer."""
        self.mock_data_manager = get_mock_data_manager()
        self.pricing_data = self._load_pricing_data()
        self.logger = logger
        
    def _load_pricing_data(self) -> Dict[str, Any]:
        """Load product pricing data."""
        try:
            prices_file = Path("data/product_prices.json")
            if prices_file.exists():
                with open(prices_file, 'r') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded pricing data with {len(data.get('prices', {}))} products")
                return data
            else:
                self.logger.warning("Pricing data file not found, using defaults")
                return self._get_default_pricing()
        except Exception as e:
            self.logger.error(f"Error loading pricing data: {e}")
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
        self.logger.warning(f"Product price not found for '{product_name}', using default $500/ton")
        return 500.0
    
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
            
            # Get product price
            product_price = self.get_product_price(product_name)
            
            # Calculate daily and annual production
            daily_production = production_capacity  # Already in tons/day
            annual_production = daily_production * (operating_hours / 24) * 365
            
            # Calculate revenue
            annual_revenue = annual_production * product_price
            
            # Apply location adjustment
            location_factor = self.calculate_location_adjustment(plant_location)
            adjusted_revenue = annual_revenue * location_factor
            
            return {
                "daily_production_tons": daily_production,
                "annual_production_tons": annual_production,
                "product_name": product_name,
                "product_price_per_ton": product_price,
                "annual_revenue": adjusted_revenue,
                "location_adjustment_factor": location_factor,
                "operating_hours_per_year": operating_hours
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating annual revenue: {e}")
            return {
                "daily_production_tons": 0,
                "annual_production_tons": 0,
                "product_name": "Unknown",
                "product_price_per_ton": 0,
                "annual_revenue": 0,
                "location_adjustment_factor": 1.0,
                "operating_hours_per_year": 8000
            }
    
    def _extract_product_name(self, plant_name: str) -> str:
        """Extract product name from plant name."""
        product_mapping = {
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
        Calculate comprehensive financial metrics.
        
        Args:
            plant_config: Plant configuration dictionary
            
        Returns:
            Dictionary with financial analysis
        """
        try:
            # Get revenue calculations
            revenue_data = self.calculate_annual_revenue(plant_config)
            
            # Get cost data
            financial_params = plant_config.get("financial_parameters", {})
            capital_cost = financial_params.get("capital_cost", 0)
            operating_cost = financial_params.get("operating_cost", 0)
            
            # Calculate key metrics
            annual_revenue = revenue_data["annual_revenue"]
            annual_profit = annual_revenue - operating_cost
            
            # Calculate ratios
            profit_margin = (annual_profit / annual_revenue * 100) if annual_revenue > 0 else 0
            roi = (annual_profit / capital_cost * 100) if capital_cost > 0 else 0
            payback_period = capital_cost / annual_profit if annual_profit > 0 else float('inf')
            
            # Calculate per-unit metrics
            daily_production = revenue_data["daily_production_tons"]
            revenue_per_ton = revenue_data["product_price_per_ton"]
            operating_cost_per_ton = (operating_cost / revenue_data["annual_production_tons"]) if revenue_data["annual_production_tons"] > 0 else 0
            profit_per_ton = revenue_per_ton - operating_cost_per_ton
            
            # Risk assessment
            risk_factors = self._assess_financial_risks(plant_config, revenue_data)
            
            return {
                "revenue_analysis": revenue_data,
                "cost_analysis": {
                    "capital_cost": capital_cost,
                    "operating_cost": operating_cost,
                    "operating_cost_per_ton": operating_cost_per_ton
                },
                "profitability_metrics": {
                    "annual_revenue": annual_revenue,
                    "annual_profit": annual_profit,
                    "profit_margin_percent": profit_margin,
                    "roi_percent": roi,
                    "payback_period_years": payback_period
                },
                "unit_economics": {
                    "revenue_per_ton": revenue_per_ton,
                    "operating_cost_per_ton": operating_cost_per_ton,
                    "profit_per_ton": profit_per_ton,
                    "daily_production_tons": daily_production
                },
                "risk_assessment": risk_factors,
                "calculation_metadata": {
                    "calculation_timestamp": datetime.now().isoformat(),
                    "pricing_data_version": self.pricing_data.get("metadata", {}).get("version", "unknown"),
                    "location_adjustment_applied": revenue_data["location_adjustment_factor"]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating financial metrics: {e}")
            return {
                "error": str(e),
                "calculation_timestamp": datetime.now().isoformat()
            }
    
    def _assess_financial_risks(self, plant_config: Dict[str, Any], revenue_data: Dict[str, float]) -> List[str]:
        """Assess financial risks based on plant configuration."""
        risks = []
        
        # Revenue risks
        if revenue_data["annual_revenue"] < 10000000:  # Less than $10M
            risks.append("Low revenue scale may limit profitability")
        
        # Cost risks
        financial_params = plant_config.get("financial_parameters", {})
        capital_cost = financial_params.get("capital_cost", 0)
        operating_cost = financial_params.get("operating_cost", 0)
        
        if capital_cost > 500000000:  # More than $500M
            risks.append("High capital cost increases financial risk")
        
        if operating_cost > revenue_data["annual_revenue"] * 0.8:
            risks.append("High operating costs relative to revenue")
        
        # Market risks
        product_price = revenue_data["product_price_per_ton"]
        if product_price < 200:
            risks.append("Low product price may be vulnerable to market fluctuations")
        
        # Capacity risks
        daily_production = revenue_data["daily_production_tons"]
        if daily_production < 50:
            risks.append("Small scale may limit economies of scale")
        
        return risks
    
    def generate_financial_report(self, plant_config: Dict[str, Any]) -> str:
        """
        Generate a human-readable financial report.
        
        Args:
            plant_config: Plant configuration dictionary
            
        Returns:
            Formatted financial report string
        """
        try:
            metrics = self.calculate_financial_metrics(plant_config)
            
            if "error" in metrics:
                return f"❌ Financial analysis failed: {metrics['error']}"
            
            revenue = metrics["revenue_analysis"]
            costs = metrics["cost_analysis"]
            profitability = metrics["profitability_metrics"]
            unit_economics = metrics["unit_economics"]
            risks = metrics["risk_assessment"]
            
            report = f"""
💰 FINANCIAL ANALYSIS REPORT
{'='*50}

📊 PRODUCTION & REVENUE
• Daily Production: {revenue['daily_production_tons']:,.1f} tons/day
• Annual Production: {revenue['annual_production_tons']:,.0f} tons/year
• Product: {revenue['product_name']}
• Market Price: ${revenue['product_price_per_ton']:,.0f}/ton
• Annual Revenue: ${revenue['annual_revenue']:,.0f}
• Location Adjustment: {revenue['location_adjustment_factor']:.2f}x

💸 COSTS & INVESTMENT
• Capital Cost: ${costs['capital_cost']:,.0f}
• Annual Operating Cost: ${costs['operating_cost']:,.0f}
• Operating Cost per Ton: ${costs['operating_cost_per_ton']:,.0f}

📈 PROFITABILITY METRICS
• Annual Profit: ${profitability['annual_profit']:,.0f}
• Profit Margin: {profitability['profit_margin_percent']:.1f}%
• Return on Investment: {profitability['roi_percent']:.1f}%
• Payback Period: {profitability['payback_period_years']:.1f} years

⚖️ UNIT ECONOMICS
• Revenue per Ton: ${unit_economics['revenue_per_ton']:,.0f}
• Operating Cost per Ton: ${unit_economics['operating_cost_per_ton']:,.0f}
• Profit per Ton: ${unit_economics['profit_per_ton']:,.0f}

⚠️ RISK ASSESSMENT
"""
            
            if risks:
                for i, risk in enumerate(risks, 1):
                    report += f"• {i}. {risk}\n"
            else:
                report += "• No significant risks identified\n"
            
            report += f"""
📋 ANALYSIS METADATA
• Calculation Date: {metrics['calculation_metadata']['calculation_timestamp']}
• Pricing Data Version: {metrics['calculation_metadata']['pricing_data_version']}
• Location Factor Applied: {metrics['calculation_metadata']['location_adjustment_applied']:.2f}
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating financial report: {e}")
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
            self.logger.error(f"Error estimating capacity: {e}")
            return {
                "error": str(e),
                "target_revenue": target_revenue,
                "product_name": product_name
            } 