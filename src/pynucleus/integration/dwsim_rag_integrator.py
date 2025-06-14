"""
DWSIM-RAG Integrator Module

Combines DWSIM simulation results with RAG query capabilities to provide:
- Enhanced simulation interpretation using knowledge base
- Contextual analysis of simulation results
- Automatic problem identification and suggestions
- Integration of simulation data with research knowledge
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np


class DWSIMRAGIntegrator:
    """Integrates DWSIM simulation results with RAG knowledge base."""
    
    def __init__(self, rag_pipeline=None, results_dir: str = "data/05_output/results"):
        """Initialize DWSIM-RAG integrator with specified directory."""
        self.rag_pipeline = rag_pipeline
        self.results_dir = Path(results_dir)
        
        # Create directory if it doesn't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.integrated_results = []
        
    def integrate_simulation_results(self, dwsim_results: List[Dict], 
                                   perform_rag_analysis: bool = True,
                                   verbose: bool = False) -> List[Dict]:
        """
        Integrate DWSIM results with RAG insights (quiet background operation)
        """
        if not dwsim_results:
            if verbose:
                print("⚠️ No DWSIM results provided for integration")
            return []
        
        if verbose:
            print(f"🔗 Integrating DWSIM results with RAG knowledge base...")
            print(f"📊 Processing {len(dwsim_results)} DWSIM simulation results...")
        
        integrated_results = []
        
        for i, sim_result in enumerate(dwsim_results):
            if verbose:
                print(f"📊 Processing simulation {i+1}/{len(dwsim_results)}: {sim_result.get('case_name', 'Unknown')}")
            
            enhanced_result = self._enhance_single_simulation(sim_result, perform_rag_analysis, verbose=verbose)
            integrated_results.append(enhanced_result)
        
        self.integrated_results = integrated_results
        
        if verbose:
            print(f"✅ Integration completed for {len(integrated_results)} simulations")
        else:
            print(f"✅ Enhanced {len(integrated_results)} simulations with RAG insights")
        
        return integrated_results
    
    def _enhance_single_simulation(self, sim_result: Dict, perform_rag_analysis: bool = True, verbose: bool = False) -> Dict:
        """Enhance a single simulation result (quiet operation)"""
        
        enhanced_result = {
            'original_simulation': sim_result,
            'enhancement_timestamp': datetime.now().isoformat(),
            'performance_metrics': {},
            'potential_issues': [],
            'recommendations': [],
            'optimization_opportunities': [],
            'rag_insights': [],
            'knowledge_integration': perform_rag_analysis
        }
        
        # Analyze performance metrics
        performance_metrics = self._analyze_performance(sim_result)
        enhanced_result['performance_metrics'] = performance_metrics
        
        # Identify potential issues
        issues = self._identify_issues(sim_result, performance_metrics)
        enhanced_result['potential_issues'] = issues
        
        # Generate recommendations
        recommendations = self._generate_recommendations(sim_result, performance_metrics, issues)
        enhanced_result['recommendations'] = recommendations
        
        # Identify optimization opportunities
        optimizations = self._identify_optimizations(sim_result, performance_metrics)
        enhanced_result['optimization_opportunities'] = optimizations
        
        # Perform RAG analysis if requested
        if perform_rag_analysis and self.rag_pipeline:
            try:
                rag_insights = self._get_rag_insights(sim_result, verbose=verbose)
                enhanced_result['rag_insights'] = rag_insights
            except Exception as e:
                if verbose:
                    print(f"⚠️ RAG analysis failed: {str(e)}")
                enhanced_result['rag_insights'] = []
        
        return enhanced_result
    
    def _analyze_performance(self, sim_result: Dict) -> Dict:
        """Analyze simulation performance metrics dynamically based on available data."""
        metrics = {
            'overall_performance': 'Good',
            'efficiency_rating': 'High',
            'reliability_score': 'High' if sim_result.get('success', True) else 'Low',
            'performance_indicators': {}
        }
        
        # Extract available performance data from simulation results
        sim_data = sim_result.get('results', {})
        if isinstance(sim_data, str):
            try:
                import ast
                sim_data = ast.literal_eval(sim_data)
            except:
                sim_data = {}
        
        # Add available metrics dynamically
        performance_keys = ['conversion', 'selectivity', 'yield', 'efficiency', 'purity', 'recovery']
        for key in performance_keys:
            if key in sim_result:
                value = sim_result[key]
                if isinstance(value, (int, float)):
                    # Convert to percentage if it's a decimal
                    if key == 'recovery':
                        metrics['recovery_rate'] = value * 100 if value <= 1.0 else value
                    else:
                        metrics[key] = value * 100 if value <= 1.0 else value
                else:
                    metrics[key] = value
        
        # Check simulation-specific results
        if sim_data:
            for key, value in sim_data.items():
                if isinstance(value, (int, float)) and key not in metrics:
                    if 'rate' in key.lower() or 'percentage' in key.lower():
                        metrics[key] = value * 100 if value <= 1.0 else value
                    else:
                        metrics[key] = value
        
        # Calculate derived metrics
        conversion = metrics.get('conversion', sim_result.get('conversion', 0.85))
        selectivity = metrics.get('selectivity', sim_result.get('selectivity', 0.90))
        
        # Calculate recovery rate if not already present
        if 'recovery_rate' not in metrics:
            if conversion and selectivity:
                if isinstance(conversion, (int, float)) and isinstance(selectivity, (int, float)):
                    # Ensure values are in decimal form for calculation
                    conv_decimal = conversion / 100 if conversion > 1 else conversion
                    sel_decimal = selectivity / 100 if selectivity > 1 else selectivity
                    recovery_rate = conv_decimal * sel_decimal * 100
                    metrics['recovery_rate'] = round(recovery_rate, 1)
            else:
                # Default recovery rate based on process type
                process_type = sim_result.get('simulation_type', sim_result.get('type', 'reactor'))
                default_recovery = {
                    'distillation': 85.0,
                    'reactor': 78.0,
                    'absorber': 92.0,
                    'crystallizer': 80.0,
                    'heat_exchanger': 95.0
                }.get(process_type, 82.5)
                metrics['recovery_rate'] = default_recovery
        
        # Update overall performance based on metrics
        if 'recovery_rate' in metrics:
            recovery = metrics['recovery_rate']
            if recovery >= 90:
                metrics['overall_performance'] = 'Excellent'
                metrics['efficiency_rating'] = 'Very High'
            elif recovery >= 80:
                metrics['overall_performance'] = 'Good'
                metrics['efficiency_rating'] = 'High'
            elif recovery >= 70:
                metrics['overall_performance'] = 'Fair'
                metrics['efficiency_rating'] = 'Medium'
            else:
                metrics['overall_performance'] = 'Poor'
                metrics['efficiency_rating'] = 'Low'
        
        # Add process-specific indicators
        process_type = sim_result.get('simulation_type', sim_result.get('type', 'unknown'))
        metrics['performance_indicators'] = {
            'process_type': process_type,
            'success_status': sim_result.get('success', True),
            'duration_seconds': sim_result.get('duration_seconds', 0),
            'timestamp': sim_result.get('timestamp', 'unknown')
        }
        
        return metrics
    
    def _identify_issues(self, sim_result: Dict, performance_metrics: Dict) -> List[str]:
        """Identify potential issues in simulation results."""
        issues = []
        
        # Check if simulation failed
        if not sim_result.get('success', True):
            issues.append("Simulation execution failed - check input parameters and system setup")
        
        return issues
    
    def _generate_recommendations(self, sim_result: Dict, performance_metrics: Dict, issues: List[str]) -> List[str]:
        """Generate recommendations based on simulation results."""
        recommendations = []
        
        sim_type = sim_result.get('simulation_type', '').lower()
        
        # General recommendations
        if sim_result.get('success', True):
            recommendations.append("Simulation completed successfully - results are ready for analysis")
        else:
            recommendations.append("Simulation failed - review input parameters and system configuration")
        
        return recommendations
    
    def _identify_optimizations(self, sim_result: Dict, performance_metrics: Dict) -> List[str]:
        """Find optimization opportunities in the simulation."""
        opportunities = []
        
        sim_type = sim_result.get('simulation_type', '').lower()
        if sim_type == 'distillation':
            opportunities.append("Consider heat integration for energy efficiency")
        elif sim_type == 'reactor':
            opportunities.append("Evaluate reactor design alternatives (PFR vs CSTR)")
        
        return opportunities
    
    def _get_rag_insights(self, sim_result: Dict, verbose: bool = False) -> List[Dict]:
        """Get RAG insights for simulation (quiet operation)"""
        if not self.rag_pipeline:
            return []
        
        process_type = sim_result.get('type', 'chemical process')
        components = sim_result.get('components', [])
        
        # Create process-specific queries
        queries = [
            f"optimization strategies for {process_type} with {', '.join(components)}",
            f"common issues in {process_type} operations",
            f"best practices for {process_type} efficiency improvement"
        ]
        
        insights = []
        for query in queries[:2]:  # Limit to 2 queries for efficiency
            try:
                if hasattr(self.rag_pipeline, 'query'):
                    results = self.rag_pipeline.query(query, top_k=2)
                    for result in results:
                        insights.append({
                            'query': query,
                            'content': result.get('content', ''),
                            'source': result.get('source', 'Unknown'),
                            'score': result.get('score', 0.0)
                        })
            except Exception as e:
                if verbose:
                    print(f"⚠️ RAG query failed for: {query}")
                continue
        
        return insights
    
    def export_integrated_results(self) -> str:
        """Export integrated results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"integrated_dwsim_rag_results_{timestamp}.json"
        
        # Ensure directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(self.integrated_results, f, indent=2)
        
        return str(output_file)
    
    def get_integration_summary(self) -> Dict:
        """Get summary of integration results."""
        if not self.integrated_results:
            return {'error': 'No integrated results available'}
        
        summary = {
            'total_simulations': len(self.integrated_results),
            'successful_simulations': sum(1 for r in self.integrated_results if r['original_simulation']['success']),
            'knowledge_integrated': sum(1 for r in self.integrated_results if r['knowledge_integration']),
            'performance_distribution': {},
            'common_issues': {},
            'top_recommendations': []
        }
        
        # Performance distribution
        perf_counts = {}
        for result in self.integrated_results:
            perf = result['performance_metrics'].get('overall_performance', 'Unknown')
            perf_counts[perf] = perf_counts.get(perf, 0) + 1
        summary['performance_distribution'] = perf_counts
        
        # Common issues
        all_issues = []
        for result in self.integrated_results:
            all_issues.extend(result['potential_issues'])
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        summary['common_issues'] = dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Top recommendations
        all_recommendations = []
        for result in self.integrated_results:
            all_recommendations.extend(result['recommendations'])
        
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        summary['top_recommendations'] = list(dict(sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]).keys())
        
        return summary 