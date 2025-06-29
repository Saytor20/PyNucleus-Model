"""
Financial Analyzer module for modular plant financial analysis using LLM.
"""

import json
from typing import Dict, Any, List
from ..llm.llm_runner import LLMRunner
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class FinancialAnalyzer:
    """Handles financial analysis of modular plants using LLM."""
    
    def __init__(self):
        """Initialize the FinancialAnalyzer with LLM engine."""
        self.llm = LLMRunner()
    
    def _create_financial_prompt(self, plant_config: Dict[str, Any]) -> str:
        """Create the financial analysis prompt for the LLM."""
        params = plant_config["parameters"]
        financial_params = plant_config["financial_parameters"]
        
        prompt = f"""You are a financial analyst for modular chemical plants.

### PLANT CONFIGURATION:
- Feedstock: {params['feedstock']}
- Production Capacity: {params['production_capacity']} tons/year
- Location: {params['plant_location']}
- Operating Hours/Year: {params['operating_hours']}
- Estimated Capital Cost: ${financial_params['capital_cost']:,.0f} USD
- Estimated Operating Cost: ${financial_params['operating_cost']:,.0f} USD/year
- Product Price: ${financial_params['product_price']} USD/ton

### TASKS:
1. Calculate projected annual revenue.
2. Estimate annual profit margin and ROI.
3. Highlight financial risks.
4. Provide strategic recommendations.

### RESPONSE FORMAT (JSON):
{{
  "annual_revenue": float,
  "profit_margin_percent": float,
  "roi_percent": float,
  "financial_risks": ["risk1", "risk2"],
  "strategic_recommendations": "brief analysis"
}}

Provide only the JSON response, no additional text."""
        
        return prompt
    
    def _parse_financial_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into structured financial data."""
        try:
            # Extract JSON from response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[start_idx:end_idx]
            financial_data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                "annual_revenue", "profit_margin_percent", 
                "roi_percent", "financial_risks", "strategic_recommendations"
            ]
            
            for field in required_fields:
                if field not in financial_data:
                    raise ValueError(f"Missing required field: {field}")
            
            return financial_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse financial response JSON: {e}")
            # Return default analysis if parsing fails
            return self._get_default_financial_analysis()
        except Exception as e:
            logger.error(f"Error parsing financial response: {e}")
            return self._get_default_financial_analysis()
    
    def _get_default_financial_analysis(self) -> Dict[str, Any]:
        """Provide default financial analysis when LLM fails."""
        return {
            "annual_revenue": 0.0,
            "profit_margin_percent": 0.0,
            "roi_percent": 0.0,
            "financial_risks": ["Unable to analyze - using default values"],
            "strategic_recommendations": "Financial analysis unavailable. Please review plant parameters manually."
        }
    
    def _calculate_basic_financials(self, plant_config: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic financial metrics as backup."""
        params = plant_config["parameters"]
        financial_params = plant_config["financial_parameters"]
        
        # Calculate annual revenue
        annual_revenue = params["production_capacity"] * financial_params["product_price"]
        
        # Calculate profit
        annual_profit = annual_revenue - financial_params["operating_cost"]
        
        # Calculate profit margin
        profit_margin_percent = (annual_profit / annual_revenue * 100) if annual_revenue > 0 else 0
        
        # Calculate ROI
        roi_percent = (annual_profit / financial_params["capital_cost"] * 100) if financial_params["capital_cost"] > 0 else 0
        
        return {
            "annual_revenue": annual_revenue,
            "profit_margin_percent": profit_margin_percent,
            "roi_percent": roi_percent
        }
    
    def analyze_financials(self, plant_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive financial analysis of the plant."""
        try:
            # Create financial prompt
            prompt = self._create_financial_prompt(plant_config)
            
            # Get LLM analysis
            logger.info("Requesting financial analysis from LLM")
            response = self.llm.ask(prompt, max_length=400, temperature=0.3)
            
            # Parse response
            financial_analysis = self._parse_financial_response(response)
            
            # Add basic calculations as backup/verification
            basic_financials = self._calculate_basic_financials(plant_config)
            
            # Combine results
            comprehensive_analysis = {
                "llm_analysis": financial_analysis,
                "basic_calculations": basic_financials,
                "plant_configuration": {
                    "capital_cost": plant_config["financial_parameters"]["capital_cost"],
                    "operating_cost": plant_config["financial_parameters"]["operating_cost"],
                    "product_price": plant_config["financial_parameters"]["product_price"],
                    "production_capacity": plant_config["parameters"]["production_capacity"]
                },
                "analysis_metadata": {
                    "llm_used": True,
                    "analysis_timestamp": "2025-01-27T00:00:00Z"
                }
            }
            
            logger.info("Financial analysis completed successfully")
            return comprehensive_analysis
            
        except Exception as e:
            logger.error(f"Error in financial analysis: {e}")
            # Return basic analysis if LLM fails
            basic_financials = self._calculate_basic_financials(plant_config)
            
            return {
                "llm_analysis": self._get_default_financial_analysis(),
                "basic_calculations": basic_financials,
                "plant_configuration": {
                    "capital_cost": plant_config["financial_parameters"]["capital_cost"],
                    "operating_cost": plant_config["financial_parameters"]["operating_cost"],
                    "product_price": plant_config["financial_parameters"]["product_price"],
                    "production_capacity": plant_config["parameters"]["production_capacity"]
                },
                "analysis_metadata": {
                    "llm_used": False,
                    "error": str(e),
                    "analysis_timestamp": "2025-01-27T00:00:00Z"
                }
            } 