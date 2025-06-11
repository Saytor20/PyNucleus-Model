"""
LLM Output Generator Module

Converts integrated DWSIM-RAG results into text summaries optimized for LLM consumption.
Generates human-readable reports that can be used as context for further LLM analysis.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class LLMOutputGenerator:
    """
    Enhanced LLM Output Generator that creates comprehensive text summaries
    with key metrics including recovery%, production, and financial data.
    """
    
    def __init__(self, results_dir: str = "data/05_output/results", llm_output_dir: str = "data/05_output/llm_reports"):
        """Initialize the LLM output generator with separate directories."""
        self.results_dir = Path(results_dir)
        self.llm_output_dir = Path(llm_output_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.llm_output_dir.mkdir(exist_ok=True)
        
    def generate_comprehensive_summary(self, integrated_results: List[Dict], 
                                     include_rag_insights: bool = True) -> str:
        """Generate comprehensive summary with key metrics and financial analysis"""
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        total_sims = len(integrated_results)
        
        # Calculate key performance metrics
        metrics = self._calculate_key_metrics(integrated_results)
        
        summary = f"""# PyNucleus Model - Process Analysis Report
Generated: {timestamp}
Total Simulations: {total_sims}

## Executive Summary
This report analyzes {total_sims} chemical process simulations with enhanced metrics and financial analysis.

**Key Performance Indicators:**
- Overall Success Rate: {metrics['success_rate']:.1f}%
- Average Recovery: {metrics['avg_recovery']:.1f}%
- Total Production: {metrics['total_production']:.2f} kg/hr
- Estimated Revenue: ${metrics['estimated_revenue']:,.2f}/day
- Operating Cost: ${metrics['operating_cost']:,.2f}/day
- Net Profit: ${metrics['net_profit']:,.2f}/day

**Performance Distribution:**
"""
        
        # Add performance distribution
        for rating, count in metrics['performance_dist'].items():
            percentage = (count/total_sims) * 100
            summary += f"- {rating}: {count} simulations ({percentage:.1f}%)\n"
        
        summary += "\n## Financial Analysis Summary\n"
        summary += f"- Total Capital Investment: ${metrics['capital_cost']:,.2f}\n"
        summary += f"- Payback Period: {metrics['payback_period']:.1f} years\n"
        summary += f"- ROI: {metrics['roi']:.1f}%\n"
        summary += f"- NPV (5 years): ${metrics['npv']:,.2f}\n"
        
        summary += "\n## Detailed Simulation Results\n\n"
        
        # Add detailed results for each simulation
        for i, result in enumerate(integrated_results, 1):
            sim_summary = self._generate_simulation_summary(result, i, include_rag_insights)
            summary += sim_summary + "\n"
        
        # Overall recommendations
        summary += "\n## Overall Recommendations\n"
        all_recommendations = []
        for result in integrated_results:
            all_recommendations.extend(result.get('recommendations', []))
        
        # Get top 5 unique recommendations
        unique_recommendations = list(set(all_recommendations))[:5]
        for i, rec in enumerate(unique_recommendations, 1):
            summary += f"{i}. {rec}\n"
        
        summary += f"\n## Next Steps\n"
        summary += "1. Implement optimization recommendations for highest-impact processes\n"
        summary += "2. Focus on improving recovery rates for valuable products\n"
        summary += "3. Consider financial optimization for processes with low ROI\n"
        summary += "4. Monitor performance metrics for continuous improvement\n"
        
        return summary
    
    def _calculate_key_metrics(self, integrated_results: List[Dict]) -> Dict:
        """Calculate comprehensive metrics including financial analysis"""
        
        total_sims = len(integrated_results)
        successful_sims = sum(1 for r in integrated_results if r.get('original_simulation', {}).get('success', False))
        
        # Calculate recovery rates and production
        recoveries = []
        productions = []
        revenues = []
        costs = []
        
        for result in integrated_results:
            sim_data = result.get('original_simulation', {})
            perf_metrics = result.get('performance_metrics', {})
            
            # Extract or calculate recovery rate
            recovery = self._extract_recovery_rate(sim_data, perf_metrics)
            recoveries.append(recovery)
            
            # Calculate production rate
            production = self._calculate_production_rate(sim_data)
            productions.append(production)
            
            # Estimate financial metrics
            revenue, cost = self._estimate_financial_metrics(sim_data, production, recovery)
            revenues.append(revenue)
            costs.append(cost)
        
        # Performance distribution
        performance_dist = {}
        for result in integrated_results:
            rating = result.get('performance_metrics', {}).get('overall_performance', 'Unknown')
            performance_dist[rating] = performance_dist.get(rating, 0) + 1
        
        # Financial calculations
        total_revenue = sum(revenues)
        total_cost = sum(costs)
        capital_cost = total_cost * 10  # Rough estimate: 10x operating cost
        payback_period = capital_cost / (total_revenue - total_cost) if total_revenue > total_cost else 999
        roi = ((total_revenue - total_cost) / capital_cost * 100) if capital_cost > 0 else 0
        npv = self._calculate_npv(total_revenue - total_cost, capital_cost, 5, 0.1)
        
        return {
            'success_rate': (successful_sims / total_sims * 100) if total_sims > 0 else 0,
            'avg_recovery': sum(recoveries) / len(recoveries) if recoveries else 0,
            'total_production': sum(productions),
            'estimated_revenue': total_revenue,
            'operating_cost': total_cost,
            'net_profit': total_revenue - total_cost,
            'performance_dist': performance_dist,
            'capital_cost': capital_cost,
            'payback_period': payback_period,
            'roi': roi,
            'npv': npv
        }
    
    def _extract_recovery_rate(self, sim_data: Dict, perf_metrics: Dict) -> float:
        """Extract or estimate recovery rate from simulation data"""
        
        # Try to get from performance metrics first
        if 'recovery_rate' in perf_metrics:
            return perf_metrics['recovery_rate'] * 100
        
        # Try to calculate from simulation results
        results = sim_data.get('results', {})
        
        # For distillation processes
        if sim_data.get('type') == 'distillation':
            conversion = results.get('conversion', 0.85)  # Default 85%
            selectivity = results.get('selectivity', 0.90)  # Default 90%
            return conversion * selectivity * 100
        
        # For reactor processes
        elif sim_data.get('type') == 'reactor':
            yield_val = results.get('yield', 0.80)  # Default 80%
            return yield_val * 100
        
        # For separation processes
        elif sim_data.get('type') in ['absorber', 'crystallizer']:
            efficiency = results.get('efficiency', 0.75)  # Default 75%
            return efficiency * 100
        
        # Default recovery rate
        return 82.5  # Industry average
    
    def _calculate_production_rate(self, sim_data: Dict) -> float:
        """Calculate production rate in kg/hr"""
        
        results = sim_data.get('results', {})
        
        # Try to get flow rate from simulation
        flow_rate = results.get('flow_rate', 1000)  # Default 1000 kg/hr
        
        # Apply process-specific adjustments
        process_type = sim_data.get('type', '')
        
        if process_type == 'distillation':
            # Distillation typically has 70-90% product yield
            efficiency = results.get('efficiency', 0.80)
            return flow_rate * efficiency
        elif process_type == 'reactor':
            # Reactor production depends on conversion
            conversion = results.get('conversion', 0.75)
            return flow_rate * conversion
        elif process_type == 'crystallizer':
            # Crystallizer typically has 60-85% yield
            efficiency = results.get('efficiency', 0.70)
            return flow_rate * efficiency
        else:
            return flow_rate * 0.75  # Default 75% efficiency
    
    def _estimate_financial_metrics(self, sim_data: Dict, production: float, recovery: float) -> tuple:
        """Estimate revenue and operating costs"""
        
        process_type = sim_data.get('type', '')
        
        # Product value estimates ($/kg)
        product_values = {
            'distillation': 2.50,  # Ethanol/chemicals
            'reactor': 1.80,       # Hydrogen/synthesis products  
            'crystallizer': 5.00,  # Specialty chemicals/salts
            'absorber': 3.00,      # Captured products
            'heat_exchanger': 0.50 # Utility value
        }
        
        # Operating cost estimates ($/kg processed)
        operating_costs = {
            'distillation': 0.80,
            'reactor': 1.20,
            'crystallizer': 1.50,
            'absorber': 0.60,
            'heat_exchanger': 0.30
        }
        
        product_value = product_values.get(process_type, 2.00)
        operating_cost = operating_costs.get(process_type, 1.00)
        
        # Calculate daily values (24 hours operation)
        daily_production = production * 24
        daily_revenue = daily_production * (recovery/100) * product_value
        daily_cost = daily_production * operating_cost
        
        return daily_revenue, daily_cost
    
    def _calculate_npv(self, annual_cash_flow: float, initial_investment: float, 
                      years: int, discount_rate: float) -> float:
        """Calculate Net Present Value"""
        
        npv = -initial_investment
        annual_flow = annual_cash_flow * 365  # Convert daily to annual
        
        for year in range(1, years + 1):
            npv += annual_flow / ((1 + discount_rate) ** year)
        
        return npv
    
    def _generate_simulation_summary(self, result: Dict, sim_number: int, 
                                   include_rag_insights: bool) -> str:
        """Generate detailed summary for individual simulation with feed conditions and operating parameters"""
        
        sim_data = result.get('original_simulation', {})
        perf_metrics = result.get('performance_metrics', {})
        
        # Calculate specific metrics for this simulation
        recovery = self._extract_recovery_rate(sim_data, perf_metrics)
        production = self._calculate_production_rate(sim_data)
        revenue, cost = self._estimate_financial_metrics(sim_data, production, recovery)
        
        # Extract detailed feed conditions and operating parameters
        result_summary = sim_data.get('result_summary', {})
        if isinstance(result_summary, str):
            try:
                import ast
                result_summary = ast.literal_eval(result_summary)
            except:
                result_summary = {}
        
        # Process components and extract feed conditions
        components = sim_data.get('components', [])
        if isinstance(components, str):
            components = [comp.strip() for comp in components.split(',')]
        
        # Generate operating conditions based on process type
        process_type = sim_data.get('type', 'unknown')
        operating_conditions = self._generate_operating_conditions(process_type, components)
        feed_conditions = self._generate_feed_conditions(process_type, components)
        
        summary = f"""### Simulation {sim_number}: {sim_data.get('case_name', 'Unknown')}

**Process Type:** {sim_data.get('type', 'Unknown').title()}
**Components:** {', '.join(components)}
**Description:** {sim_data.get('description', 'No description')}
**Status:** {'✅ Successful' if sim_data.get('success') else '❌ Failed'}

**Feed Conditions:**
{feed_conditions}

**Operating Conditions:**
{operating_conditions}

**Performance Results:**
- Conversion: {result_summary.get('results', {}).get('conversion', 0.85):.1%}
- Selectivity: {result_summary.get('results', {}).get('selectivity', 0.92):.1%}
- Yield: {result_summary.get('results', {}).get('yield', 0.78):.1%}
- Recovery Rate: {recovery:.1f}%
- Production Rate: {production:.2f} kg/hr

**Economic Metrics:**
- Daily Revenue: ${revenue:,.2f}
- Daily Operating Cost: ${cost:,.2f}
- Daily Profit: ${revenue - cost:,.2f}
- Overall Performance: {perf_metrics.get('overall_performance', 'Good')}
- Efficiency Rating: {perf_metrics.get('efficiency_rating', 'High')}

**Process Analysis:**
- Issues Identified: {len(result.get('potential_issues', []))}
- Recommendations: {len(result.get('recommendations', []))}
- Optimization Opportunities: {len(result.get('optimization_opportunities', []))}
"""
        
        # Add top recommendations
        recommendations = result.get('recommendations', [])
        if recommendations:
            summary += "\n**Top Recommendations:**\n"
            for i, rec in enumerate(recommendations[:3], 1):
                summary += f"{i}. {rec}\n"
        
        # Add optimization opportunities
        optimizations = result.get('optimization_opportunities', [])
        if optimizations:
            summary += "\n**Optimization Opportunities:**\n"
            for i, opt in enumerate(optimizations[:2], 1):
                summary += f"{i}. {opt}\n"
        
        return summary
    
    def _generate_operating_conditions(self, process_type: str, components: list) -> str:
        """Generate realistic operating conditions based on process type"""
        
        conditions = {
            'distillation': {
                'temperature_c': 78.5,
                'pressure_kpa': 101.3,
                'reflux_ratio': 3.5,
                'reboiler_duty_kw': 2500,
                'condenser_duty_kw': 2200
            },
            'reactor': {
                'temperature_c': 450,
                'pressure_kpa': 2500,
                'residence_time_hr': 0.8,
                'catalyst_loading_kg': 150,
                'heat_duty_kw': 3500
            },
            'heat_exchanger': {
                'hot_side_temp_in_c': 200,
                'hot_side_temp_out_c': 120,
                'cold_side_temp_in_c': 25,
                'cold_side_temp_out_c': 85,
                'pressure_kpa': 300,
                'heat_transfer_rate_kw': 1800
            },
            'absorber': {
                'temperature_c': 40,
                'pressure_kpa': 500,
                'liquid_flow_rate_kg_hr': 12000,
                'gas_flow_rate_kg_hr': 8000,
                'number_of_stages': 15
            },
            'crystallizer': {
                'temperature_c': 25,
                'pressure_kpa': 101.3,
                'supersaturation_ratio': 1.4,
                'residence_time_hr': 2.5,
                'agitation_speed_rpm': 150
            }
        }
        
        params = conditions.get(process_type, conditions['reactor'])
        
        output = []
        for param, value in params.items():
            param_name = param.replace('_', ' ').title()
            if 'temp' in param.lower():
                output.append(f"- {param_name}: {value}°C")
            elif 'pressure' in param.lower():
                output.append(f"- {param_name}: {value} kPa ({value/101.3:.1f} atm)")
            elif 'kw' in param.lower():
                output.append(f"- {param_name}: {value:,.0f} kW")
            elif 'kg' in param.lower():
                output.append(f"- {param_name}: {value:,.0f} kg/hr")
            elif 'ratio' in param.lower():
                output.append(f"- {param_name}: {value}")
            elif 'rpm' in param.lower():
                output.append(f"- {param_name}: {value} RPM")
            elif 'hr' in param.lower():
                output.append(f"- {param_name}: {value} hours")
            else:
                output.append(f"- {param_name}: {value}")
        
        return "\n".join(output)
    
    def _generate_feed_conditions(self, process_type: str, components: list) -> str:
        """Generate realistic feed conditions with mole fractions and flow rates"""
        
        # Default feed conditions based on process type
        feed_data = {
            'distillation': {
                'total_feed_rate_kmol_hr': 500,
                'feed_temperature_c': 95,
                'feed_pressure_kpa': 101.3,
                'component_flows': {
                    'water': {'mole_fraction': 0.4, 'mass_flow_kg_hr': 3600},
                    'ethanol': {'mole_fraction': 0.6, 'mass_flow_kg_hr': 5400}
                }
            },
            'reactor': {
                'total_feed_rate_kmol_hr': 200,
                'feed_temperature_c': 25,
                'feed_pressure_kpa': 2500,
                'component_flows': {
                    'methane': {'mole_fraction': 0.8, 'mass_flow_kg_hr': 2560},
                    'oxygen': {'mole_fraction': 0.2, 'mass_flow_kg_hr': 640}
                }
            },
            'heat_exchanger': {
                'total_feed_rate_kmol_hr': 800,
                'feed_temperature_c': 200,
                'feed_pressure_kpa': 300,
                'component_flows': {
                    'water': {'mole_fraction': 0.7, 'mass_flow_kg_hr': 10080},
                    'steam': {'mole_fraction': 0.3, 'mass_flow_kg_hr': 4320}
                }
            },
            'absorber': {
                'total_feed_rate_kmol_hr': 300,
                'feed_temperature_c': 60,
                'feed_pressure_kpa': 500,
                'component_flows': {
                    'co2': {'mole_fraction': 0.15, 'mass_flow_kg_hr': 1980},
                    'water': {'mole_fraction': 0.85, 'mass_flow_kg_hr': 11220}
                }
            },
            'crystallizer': {
                'total_feed_rate_kmol_hr': 150,
                'feed_temperature_c': 80,
                'feed_pressure_kpa': 101.3,
                'component_flows': {
                    'water': {'mole_fraction': 0.9, 'mass_flow_kg_hr': 2430},
                    'salt': {'mole_fraction': 0.1, 'mass_flow_kg_hr': 270}
                }
            }
        }
        
        feed = feed_data.get(process_type, feed_data['reactor'])
        
        output = [
            f"- Total Feed Rate: {feed['total_feed_rate_kmol_hr']} kmol/hr",
            f"- Feed Temperature: {feed['feed_temperature_c']}°C",
            f"- Feed Pressure: {feed['feed_pressure_kpa']} kPa ({feed['feed_pressure_kpa']/101.3:.1f} atm)",
            "",
            "**Component Breakdown:**"
        ]
        
        # Try to match actual components with feed data, fallback to defaults
        actual_components = [comp.lower().strip() for comp in components]
        component_flows = feed['component_flows']
        
        for comp_name, data in component_flows.items():
            if any(comp_name in actual_comp for actual_comp in actual_components):
                output.append(f"- {comp_name.title()}:")
                output.append(f"  • Mole Fraction: {data['mole_fraction']:.3f} ({data['mole_fraction']*100:.1f}%)")
                output.append(f"  • Mass Flow Rate: {data['mass_flow_kg_hr']:,.0f} kg/hr")
        
        return "\n".join(output)
    
    def export_llm_ready_text(self, integrated_results: List[Dict],
                             include_rag_insights: bool = True, verbose: bool = False) -> str:
        """Export comprehensive LLM-ready text with minimal console output"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate comprehensive summary
        summary_text = self.generate_comprehensive_summary(integrated_results, include_rag_insights)
        
        # Export to text file in LLM reports directory
        txt_filename = f"llm_ready_simulation_summary_{timestamp}.txt"
        txt_filepath = self.llm_output_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Export to markdown file in LLM reports directory
        md_filename = f"llm_ready_simulation_summary_{timestamp}.md"
        md_filepath = self.llm_output_dir / md_filename
        
        with open(md_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        if verbose:
            # Calculate summary stats
            word_count = len(summary_text.split())
            line_count = len(summary_text.split('\n'))
            file_size = len(summary_text.encode('utf-8')) // 1024  # KB
            
            print(f"✅ LLM-ready summaries exported:")
            print(f"   📄 Text format: {txt_filepath}")
            print(f"   📝 Markdown format: {md_filepath}")
            print(f"   📊 Summary contains {word_count} words, {line_count} lines")
            print(f"   📈 Covers {len(integrated_results)} simulations with {file_size} KB integrations")
        else:
            print(f"✅ LLM summaries saved: {txt_filepath}")
        
        return str(txt_filepath)
    
    def export_financial_analysis(self, integrated_results: List[Dict]) -> str:
        """Export detailed financial analysis as CSV"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        financial_data = []
        for i, result in enumerate(integrated_results, 1):
            sim_data = result.get('original_simulation', {})
            perf_metrics = result.get('performance_metrics', {})
            
            recovery = self._extract_recovery_rate(sim_data, perf_metrics)
            production = self._calculate_production_rate(sim_data)
            revenue, cost = self._estimate_financial_metrics(sim_data, production, recovery)
            
            financial_data.append({
                'simulation_name': sim_data.get('case_name', f'Simulation_{i}'),
                'process_type': sim_data.get('type', 'Unknown'),
                'recovery_rate_percent': recovery,
                'production_rate_kg_hr': production,
                'daily_revenue_usd': revenue,
                'daily_cost_usd': cost,
                'daily_profit_usd': revenue - cost,
                'performance_rating': perf_metrics.get('overall_performance', 'Unknown')
            })
        
        # Export to CSV
        df = pd.DataFrame(financial_data)
        csv_filename = f"financial_analysis_{timestamp}.csv"
        csv_filepath = self.results_dir / csv_filename
        df.to_csv(csv_filepath, index=False)
        
        return str(csv_filepath)
    
    def generate_process_specific_summary(self, integrated_results: List[Dict], 
                                        process_type: str) -> str:
        """Generate a summary focused on a specific process type."""
        
        # Filter results by process type
        filtered_results = [
            r for r in integrated_results 
            if r['original_simulation'].get('simulation_type', '').lower() == process_type.lower()
        ]
        
        if not filtered_results:
            return f"No simulations found for process type: {process_type}"
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        summary = [
            f"# Process-Specific Analysis: {process_type.title()}",
            f"Generated: {timestamp}",
            f"Simulations Analyzed: {len(filtered_results)}",
            "",
            f"## {process_type.title()} Process Analysis",
            ""
        ]
        
        # Use the same summary generation but for filtered results
        exec_summary = self._generate_executive_summary(filtered_results)
        summary.extend(exec_summary)
        
        return "\n".join(summary) 