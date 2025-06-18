"""
DWSIM-RAG integration for PyNucleus system.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

class DWSIMRAGIntegrator:
    """Integrate DWSIM simulation results with RAG knowledge base."""
    
    def __init__(self, results_dir: Optional[str] = None):
        """
        Initialize DWSIM-RAG integrator.
        
        Args:
            results_dir: Optional directory for storing integration results
        """
        self.logger = logging.getLogger(__name__)
        
        # Set up results directory
        if results_dir:
            self.results_dir = Path(results_dir)
            self.results_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.results_dir = Path("data/05_output/integration_results")
            self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Store integrated results for export
        self.integrated_results = []
        
        self.logger.info(f"DWSIMRAGIntegrator initialized with results_dir: {self.results_dir}")
        
    def integrate_simulation_with_knowledge(
        self, 
        simulation_results: Dict[str, Any], 
        rag_insights: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Integrate DWSIM simulation results with RAG knowledge.
        
        Args:
            simulation_results: Results from DWSIM simulation
            rag_insights: Optional RAG-generated insights
            
        Returns:
            Enhanced results with RAG integration
        """
        try:
            # Mock RAG insights if not provided
            if not rag_insights:
                rag_insights = [
                    {
                        "text": "Modular design principles can improve process efficiency",
                        "source": "Chemical Engineering Handbook",
                        "confidence": 0.85
                    },
                    {
                        "text": "Temperature optimization is critical for distillation processes",
                        "source": "Process Optimization Guide", 
                        "confidence": 0.92
                    }
                ]
            
            enhanced_results = {
                "original_simulation": simulation_results,
                "rag_insights": rag_insights,
                "knowledge_integration": True,
                "integration_timestamp": datetime.now().isoformat(),
                "recommendations": self._generate_recommendations(simulation_results, rag_insights),
                "optimization_opportunities": self._identify_optimization_opportunities(simulation_results),
                "performance_metrics": self._calculate_performance_metrics(simulation_results),
                "results_dir": str(self.results_dir)
            }
            
            self.logger.info("DWSIM-RAG integration completed successfully")
            return enhanced_results
            
        except Exception as e:
            self.logger.error(f"DWSIM-RAG integration failed: {e}")
            return {
                "original_simulation": simulation_results,
                "rag_insights": [],
                "knowledge_integration": False,
                "error": str(e),
                "integration_timestamp": datetime.now().isoformat(),
                "results_dir": str(self.results_dir)
            }
    
    def integrate_simulation_results(
        self,
        simulation_data: List[Dict[str, Any]] | Dict[str, Any],
        perform_rag_analysis: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Integrate multiple simulation results with RAG analysis.
        
        Args:
            simulation_data: List of DWSIM simulation results or single result
            perform_rag_analysis: Whether to perform RAG analysis
            
        Returns:
            List of enhanced integration results
        """
        # Handle single result input
        if isinstance(simulation_data, dict):
            simulation_data = [simulation_data]
        
        enhanced_results = []
        
        for i, sim_result in enumerate(simulation_data):
            self.logger.info(f"Integrating simulation {i+1}/{len(simulation_data)}")
            
            # Perform RAG analysis if requested
            rag_insights = None
            if perform_rag_analysis:
                rag_insights = self._perform_rag_analysis(sim_result)
            
            # Integrate with knowledge
            enhanced_result = self.integrate_simulation_with_knowledge(sim_result, rag_insights)
            enhanced_results.append(enhanced_result)
        
        # Store for export
        self.integrated_results = enhanced_results
        
        return enhanced_results
    
    def export_integrated_results(self) -> str:
        """
        Export integrated results to JSON file.
        
        Returns:
            Path to exported file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.results_dir / f"integrated_results_{timestamp}.json"
        
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_simulations": len(self.integrated_results),
                    "integration_version": "1.0"
                },
                "integrated_results": self.integrated_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            self.logger.info(f"Integrated results exported to: {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to export integrated results: {e}")
            return f"Export failed: {e}"
    
    def _perform_rag_analysis(self, simulation_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform RAG analysis on simulation result."""
        # Mock RAG analysis based on simulation data
        case_name = simulation_result.get("case_name", "unknown")
        
        rag_insights = [
            {
                "text": f"Analysis for {case_name}: Process optimization potential identified",
                "source": "RAG Knowledge Base",
                "confidence": 0.85,
                "query": f"Optimization strategies for {case_name}"
            },
            {
                "text": "Energy integration opportunities available for modular plants",
                "source": "Process Integration Handbook",
                "confidence": 0.78,
                "query": "Energy efficiency improvements"
            }
        ]
        
        return rag_insights
    
    def _calculate_performance_metrics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate performance metrics for the simulation."""
        results = simulation_results.get("results", {})
        
        # Calculate basic performance metrics
        conversion = results.get("conversion", 0.75)  # Default 75%
        selectivity = results.get("selectivity", 0.85)  # Default 85%
        yield_value = conversion * selectivity
        
        # Overall performance rating
        if yield_value > 0.8:
            performance = "Excellent"
        elif yield_value > 0.6:
            performance = "Good"
        elif yield_value > 0.4:
            performance = "Fair"
        else:
            performance = "Poor"
        
        # Efficiency rating
        temperature = results.get("temperature", 350)
        pressure = results.get("pressure", 2.0)
        
        # Simple efficiency calculation
        if temperature < 400 and pressure < 3.0:
            efficiency = "High"
        elif temperature < 500 and pressure < 5.0:
            efficiency = "Medium"
        else:
            efficiency = "Low"
        
        return {
            "conversion": conversion,
            "selectivity": selectivity,
            "yield": yield_value,
            "overall_performance": performance,
            "efficiency_rating": efficiency,
            "temperature_rating": "Optimal" if 300 <= temperature <= 400 else "Suboptimal",
            "pressure_rating": "Optimal" if 1.0 <= pressure <= 3.0 else "Suboptimal"
        }
    
    def _generate_recommendations(
        self, 
        simulation_results: Dict[str, Any], 
        rag_insights: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on simulation and RAG data."""
        recommendations = []
        
        # Analyze simulation results
        results = simulation_results.get("results", {})
        
        if results.get("conversion", 0) < 0.8:
            recommendations.append("Consider optimizing reaction temperature for higher conversion")
            
        if results.get("selectivity", 0) < 0.9:
            recommendations.append("Evaluate catalyst performance to improve selectivity")
            
        # Add RAG-based recommendations
        for insight in rag_insights:
            if "optimization" in insight.get("text", "").lower():
                recommendations.append(f"Knowledge base suggests: {insight['text']}")
                
        return recommendations
    
    def _identify_optimization_opportunities(self, simulation_results: Dict[str, Any]) -> List[str]:
        """Identify optimization opportunities."""
        opportunities = []
        
        results = simulation_results.get("results", {})
        
        if results.get("temperature", 0) > 100:
            opportunities.append("Heat integration potential")
            
        if results.get("pressure", 0) > 5:
            opportunities.append("Energy recovery from pressure reduction")
            
        opportunities.append("Process intensification evaluation")
        opportunities.append("Advanced control system implementation")
        
        return opportunities 