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
    
    def __init__(self, results_dir: str = "results"):
        """Initialize the LLM output generator."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
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
        """Generate detailed summary for individual simulation"""
        
        sim_data = result.get('original_simulation', {})
        perf_metrics = result.get('performance_metrics', {})
        
        # Calculate specific metrics for this simulation
        recovery = self._extract_recovery_rate(sim_data, perf_metrics)
        production = self._calculate_production_rate(sim_data)
        revenue, cost = self._estimate_financial_metrics(sim_data, production, recovery)
        
        summary = f"""### Simulation {sim_number}: {sim_data.get('case_name', 'Unknown')}

**Process Type:** {sim_data.get('type', 'Unknown').title()}
**Components:** {', '.join(sim_data.get('components', []))}
**Description:** {sim_data.get('description', 'No description')}
**Status:** {'✅ Successful' if sim_data.get('success') else '❌ Failed'}

**Key Performance Metrics:**
- Recovery Rate: {recovery:.1f}%
- Production Rate: {production:.2f} kg/hr
- Daily Revenue: ${revenue:,.2f}
- Daily Operating Cost: ${cost:,.2f}
- Daily Profit: ${revenue - cost:,.2f}
- Overall Performance: {perf_metrics.get('overall_performance', 'Unknown')}
- Efficiency Rating: {perf_metrics.get('efficiency_rating', 'Unknown')}

**Issues Identified:** {len(result.get('potential_issues', []))}
**Recommendations:** {len(result.get('recommendations', []))}
**Optimization Opportunities:** {len(result.get('optimization_opportunities', []))}
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
    
    def export_llm_ready_text(self, integrated_results: List[Dict],
                             include_rag_insights: bool = True, verbose: bool = False) -> str:
        """Export comprehensive LLM-ready text with minimal console output"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Generate comprehensive summary
        summary_text = self.generate_comprehensive_summary(integrated_results, include_rag_insights)
        
        # Export to text file
        txt_filename = f"llm_ready_simulation_summary_{timestamp}.txt"
        txt_filepath = self.results_dir / txt_filename
        
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        # Export to markdown file
        md_filename = f"llm_ready_simulation_summary_{timestamp}.md"
        md_filepath = self.results_dir / md_filename
        
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