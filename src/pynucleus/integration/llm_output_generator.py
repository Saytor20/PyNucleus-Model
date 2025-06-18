"""
LLM output generator for PyNucleus system.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, Template

class LLMOutputGenerator:
    """Generate LLM-ready output from simulation data."""
    
    def __init__(self, results_dir: str = "data/05_output/llm_reports"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Setup Jinja2 environment
        try:
            self.jinja_env = Environment(loader=FileSystemLoader('prompts'))
        except:
            self.jinja_env = None
            self.logger.warning("Jinja2 template environment not available")
    
    def export_llm_ready_text(self, data: Dict[str, Any]) -> Path:
        """
        Export data as LLM-ready text format.
        
        Args:
            data: Simulation data to export
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get case name from data
        case_name = "unknown_case"
        if "original_simulation" in data:
            case_name = data["original_simulation"].get("case_name", case_name)
        elif "case_name" in data:
            case_name = data["case_name"]
            
        filename = f"llm_analysis_{case_name}_{timestamp}.md"
        output_file = self.results_dir / filename
        
        # Generate content using template if available
        if self.jinja_env:
            try:
                content = self._generate_templated_content(data)
            except Exception as e:
                self.logger.warning(f"Template generation failed, using fallback: {e}")
                content = self._generate_fallback_content(data)
        else:
            content = self._generate_fallback_content(data)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.logger.info(f"LLM-ready output generated: {output_file}")
        return output_file
    
    def export_financial_analysis(self, simulation_data_list: List[Dict[str, Any]]) -> Path:
        """
        Export financial analysis for multiple simulations.
        
        Args:
            simulation_data_list: List of simulation data
            
        Returns:
            Path to exported financial analysis file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"financial_analysis_{timestamp}.md"
        output_file = self.results_dir / filename
        
        content = self._generate_financial_analysis(simulation_data_list)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
        self.logger.info(f"Financial analysis generated: {output_file}")
        return output_file
    
    def _generate_templated_content(self, data: Dict[str, Any]) -> str:
        """Generate content using Jinja2 template."""
        try:
            # Use fallback content generation instead of template for now
            # since the template expects different data structure
            return self._generate_fallback_content(data)
            
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            raise
    
    def _generate_fallback_content(self, data: Dict[str, Any]) -> str:
        """Generate fallback content without templates."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Extract simulation data
        original_sim = data.get("original_simulation", data)
        
        content = f"""# LLM Simulation Analysis Report

Generated: {timestamp}

## Process Information
Process Type: {original_sim.get('simulation_type', 'N/A')}
Components: {original_sim.get('components', 'N/A')}
Status: {original_sim.get('status', 'N/A')}

## Performance Metrics
"""
        
        # Add specific metrics that diagnostic checks for
        conversion = original_sim.get('conversion')
        selectivity = original_sim.get('selectivity')
        yield_val = original_sim.get('yield')
        temperature = original_sim.get('temperature')
        pressure = original_sim.get('pressure')
        
        if conversion is not None:
            content += f"Conversion: {conversion:.2%}\n"
        if selectivity is not None:
            content += f"Selectivity: {selectivity:.2%}\n"
        if yield_val is not None:
            content += f"Yield: {yield_val:.2%}\n"
        if temperature is not None:
            content += f"Temperature: {temperature}°C\n"
        if pressure is not None:
            content += f"Pressure: {pressure} bar\n"
        
        # Add results if available
        results = original_sim.get("results", {})
        if results:
            content += "\n### Key Results:\n"
            for key, value in results.items():
                if isinstance(value, (int, float)):
                    content += f"- {key.title()}: {value:.3f}\n"
                else:
                    content += f"- {key.title()}: {value}\n"
        
        # Add recommendations
        recommendations = data.get("recommendations", [])
        if recommendations:
            content += "\n## Recommendations:\n"
            for i, rec in enumerate(recommendations, 1):
                content += f"{i}. {rec}\n"
        
        # Add optimization opportunities
        opportunities = data.get("optimization_opportunities", [])
        if opportunities:
            content += "\n## Optimization Opportunities\n"
            for i, opp in enumerate(opportunities, 1):
                content += f"{i}. {opp}\n"
        
        # Add RAG insights
        rag_insights = data.get("rag_insights", [])
        if rag_insights:
            content += "\n## Knowledge Base Insights\n"
            for insight in rag_insights:
                content += f"- {insight.get('text', 'N/A')} (Source: {insight.get('source', 'Unknown')})\n"
        
        content += f"\n## Analysis Summary\nThis analysis was generated automatically from simulation data and enhanced with knowledge base insights. The process shows {'successful' if original_sim.get('success', False) else 'unsuccessful'} completion.\n"
        
        return content
    
    def _calculate_key_metrics(self, simulation_data_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate key financial metrics from simulation data.
        
        Args:
            simulation_data_list: List of simulation data dictionaries
            
        Returns:
            Dict containing calculated financial metrics
        """
        if not simulation_data_list:
            return {
                'avg_recovery': 0.0,
                'estimated_revenue': 0.0,
                'net_profit': 0.0,
                'roi': 0.0
            }
        
        # Extract performance metrics from simulations
        recoveries = []
        conversions = []
        yields = []
        
        for data in simulation_data_list:
            sim = data.get("original_simulation", data)
            performance = data.get("performance_metrics", {})
            
            # Extract recovery/conversion rates
            recovery = performance.get('recovery_rate', sim.get('conversion', 0.85))
            if isinstance(recovery, (int, float)):
                recoveries.append(recovery * 100 if recovery <= 1.0 else recovery)
            
            conversion = sim.get('conversion', performance.get('conversion', 0.85))
            if isinstance(conversion, (int, float)):
                conversions.append(conversion * 100 if conversion <= 1.0 else conversion)
            
            yield_val = sim.get('yield', performance.get('yield', 0.75))
            if isinstance(yield_val, (int, float)):
                yields.append(yield_val * 100 if yield_val <= 1.0 else yield_val)
        
        # Calculate averages
        avg_recovery = sum(recoveries) / len(recoveries) if recoveries else 75.0
        avg_conversion = sum(conversions) / len(conversions) if conversions else 80.0
        avg_yield = sum(yields) / len(yields) if yields else 70.0
        
        # Estimate financial metrics (simplified model)
        # Base these on typical chemical plant economics
        base_throughput = 1000.0  # tons/day
        product_price = 1500.0    # $/ton
        raw_material_cost = 800.0 # $/ton
        operating_cost = 200.0    # $/ton
        
        # Calculate revenue based on yield and recovery
        effective_yield = (avg_yield / 100.0) * (avg_recovery / 100.0)
        daily_production = base_throughput * effective_yield
        estimated_revenue = daily_production * product_price
        
        # Calculate costs
        daily_raw_material_cost = base_throughput * raw_material_cost
        daily_operating_cost = base_throughput * operating_cost
        total_daily_cost = daily_raw_material_cost + daily_operating_cost
        
        # Calculate profit
        net_profit = estimated_revenue - total_daily_cost
        
        # Calculate ROI (simplified annual ROI)
        annual_profit = net_profit * 365
        estimated_capex = estimated_revenue * 2.5  # Typical chemical plant capex
        roi = (annual_profit / estimated_capex) * 100 if estimated_capex > 0 else 0.0
        
        return {
            'avg_recovery': avg_recovery,
            'estimated_revenue': estimated_revenue,
            'net_profit': net_profit,
            'roi': roi,
            'avg_conversion': avg_conversion,
            'avg_yield': avg_yield,
            'daily_production': daily_production
        }

    def _generate_financial_analysis(self, simulation_data_list: List[Dict[str, Any]]) -> str:
        """Generate financial analysis content."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        content = f"""# Financial Analysis Report

Generated: {timestamp}

## Overview
This report analyzes the financial implications of {len(simulation_data_list)} simulation scenarios.

## Summary Statistics
- Total Simulations: {len(simulation_data_list)}
- Successful Simulations: {sum(1 for d in simulation_data_list if d.get('original_simulation', {}).get('success', False))}

## Individual Analysis
"""
        
        for i, data in enumerate(simulation_data_list, 1):
            sim = data.get("original_simulation", data)
            content += f"\n### Simulation {i}: {sim.get('case_name', f'Case_{i}')}\n"
            content += f"- Status: {sim.get('status', 'Unknown')}\n"
            content += f"- Process Type: {sim.get('simulation_type', 'N/A')}\n"
            
            results = sim.get("results", {})
            if results:
                content += "- Key Metrics:\n"
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        content += f"  * {key.title()}: {value:.3f}\n"
        
        content += "\n## Financial Recommendations\n"
        content += "1. Focus on high-conversion processes for better economics\n"
        content += "2. Implement heat integration for energy savings\n"
        content += "3. Consider modular design for capital cost reduction\n"
        
        return content 