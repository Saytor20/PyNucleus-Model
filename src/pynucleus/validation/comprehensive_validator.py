# -*- coding: utf-8 -*-
"""
Comprehensive Validation System for PyNucleus
============================================

This module provides comprehensive validation methods including:
1. Ground-truth datasets with known answers
2. Domain-specific chemical engineering validation
3. Expert evaluation frameworks
4. Response accuracy assessment
5. Citation verification
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class ValidationResult:
    """Structure for validation results."""
    query: str
    expected_answer: str
    generated_answer: str
    sources_used: List[str]
    accuracy_score: float
    citation_accuracy: float
    response_time: float
    expert_rating: Optional[float] = None
    validation_notes: str = ""

@dataclass
class GroundTruthEntry:
    """Structure for ground truth entries."""
    query: str
    expected_answer: str
    expected_sources: List[str]
    domain: str
    difficulty_level: str  # "basic", "intermediate", "advanced", "expert"
    answer_type: str  # "factual", "analytical", "procedural", "conceptual"

class ComprehensiveValidator:
    """Comprehensive validation system for PyNucleus."""
    
    def __init__(self, 
                 rag_pipeline=None,
                 validation_data_dir: str = "data/validation",
                 results_dir: str = "data/validation/results"):
        """Initialize the comprehensive validator.
        
        Args:
            rag_pipeline: RAG pipeline instance for testing
            validation_data_dir: Directory containing validation datasets
            results_dir: Directory for storing validation results
        """
        self.rag_pipeline = rag_pipeline
        self.validation_data_dir = Path(validation_data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories
        self.validation_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Load or create ground truth datasets
        self.ground_truth_datasets = self._load_ground_truth_datasets()
        
    def _load_ground_truth_datasets(self) -> Dict[str, List[GroundTruthEntry]]:
        """Load ground truth datasets for different domains."""
        datasets = {
            "chemical_engineering": self._create_chemical_engineering_dataset(),
            "modular_plants": self._create_modular_plants_dataset(),
            "process_simulation": self._create_process_simulation_dataset(),
            "dwsim_specific": self._create_dwsim_specific_dataset(),
            "general_engineering": self._create_general_engineering_dataset()
        }
        
        # Save datasets to files for inspection/editing
        for domain, dataset in datasets.items():
            dataset_file = self.validation_data_dir / f"{domain}_ground_truth.json"
            if not dataset_file.exists():
                self._save_dataset(dataset, dataset_file)
        
        return datasets
    
    def _create_chemical_engineering_dataset(self) -> List[GroundTruthEntry]:
        """Create ground truth dataset for chemical engineering queries."""
        return [
            GroundTruthEntry(
                query="What are the main advantages of modular chemical plants?",
                expected_answer="Modular chemical plants offer reduced capital costs, faster construction times, improved quality control through factory fabrication, easier transportation, scalability, and reduced on-site risks.",
                expected_sources=["wikipedia_modular_design.txt", "Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt"],
                domain="chemical_engineering",
                difficulty_level="basic",
                answer_type="factual"
            ),
            GroundTruthEntry(
                query="How do distillation columns work in chemical separation?",
                expected_answer="Distillation columns separate components based on different boiling points. The mixture is heated, vapor rises through trays or packing, and components with different volatilities separate at different heights.",
                expected_sources=["dwsim_simulation_results"],
                domain="chemical_engineering",
                difficulty_level="intermediate",
                answer_type="procedural"
            ),
            GroundTruthEntry(
                query="What factors affect reactor conversion efficiency?",
                expected_answer="Reactor conversion efficiency is affected by temperature, pressure, catalyst activity, residence time, mixing efficiency, reactant concentration, and mass transfer limitations.",
                expected_sources=["dwsim_simulation_results"],
                domain="chemical_engineering",
                difficulty_level="intermediate",
                answer_type="analytical"
            ),
            GroundTruthEntry(
                query="What are the economic benefits of modular construction in Africa?",
                expected_answer="Modular construction in Africa provides reduced infrastructure requirements, lower skilled labor needs, faster project completion, reduced financing costs, and better risk management for industrial projects.",
                expected_sources=["Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt"],
                domain="chemical_engineering",
                difficulty_level="advanced",
                answer_type="analytical"
            ),
            GroundTruthEntry(
                query="How do supply chain considerations affect modular plant design?",
                expected_answer="Supply chain considerations affect module sizing for transportation, standardization of components, supplier selection, logistics planning, and inventory management strategies.",
                expected_sources=["wikipedia_supply_chain.txt", "Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt"],
                domain="chemical_engineering",
                difficulty_level="advanced",
                answer_type="conceptual"
            )
        ]
    
    def _create_modular_plants_dataset(self) -> List[GroundTruthEntry]:
        """Create ground truth dataset for modular plants queries."""
        return [
            GroundTruthEntry(
                query="What are the key design principles for modular chemical plants?",
                expected_answer="Key design principles include standardization, skid-mounting, factory fabrication, transportability, scalability, and integration capabilities.",
                expected_sources=["wikipedia_modular_design.txt"],
                domain="modular_plants",
                difficulty_level="basic",
                answer_type="factual"
            ),
            GroundTruthEntry(
                query="How does modular design reduce project risks?",
                expected_answer="Modular design reduces risks through factory quality control, reduced on-site construction time, predictable costs, standardized components, and phased implementation.",
                expected_sources=["Manuscript Draft_Can Modular Plants Lower African Industrialization Barriers.txt"],
                domain="modular_plants",
                difficulty_level="intermediate",
                answer_type="analytical"
            )
        ]
    
    def _create_process_simulation_dataset(self) -> List[GroundTruthEntry]:
        """Create ground truth dataset for process simulation queries."""
        return [
            GroundTruthEntry(
                query="What performance metrics are tracked in distillation simulations?",
                expected_answer="Distillation simulations track separation efficiency, energy consumption, reflux ratio, number of stages, purity levels, and recovery rates.",
                expected_sources=["dwsim_simulation_results"],
                domain="process_simulation",
                difficulty_level="intermediate",
                answer_type="factual"
            ),
            GroundTruthEntry(
                query="How do you optimize reactor performance in DWSIM?",
                expected_answer="Reactor optimization in DWSIM involves adjusting temperature, pressure, residence time, catalyst loading, and analyzing conversion rates and selectivity.",
                expected_sources=["dwsim_simulation_results"],
                domain="process_simulation",
                difficulty_level="advanced",
                answer_type="procedural"
            )
        ]
    
    def _create_dwsim_specific_dataset(self) -> List[GroundTruthEntry]:
        """Create ground truth dataset for DWSIM-specific queries."""
        return [
            GroundTruthEntry(
                query="Which simulation showed the highest recovery rate?",
                expected_answer="Based on the simulation results, the absorber CO2 capture process typically shows the highest recovery rates among the standard simulations.",
                expected_sources=["dwsim_simulation_results"],
                domain="dwsim_specific",
                difficulty_level="basic",
                answer_type="factual"
            ),
            GroundTruthEntry(
                query="What are the typical conversion rates for reactor simulations?",
                expected_answer="Reactor simulations typically show conversion rates between 75-90% depending on the reaction type, temperature, and catalyst used.",
                expected_sources=["dwsim_simulation_results"],
                domain="dwsim_specific",
                difficulty_level="intermediate",
                answer_type="factual"
            )
        ]
    
    def _create_general_engineering_dataset(self) -> List[GroundTruthEntry]:
        """Create ground truth dataset for general engineering queries."""
        return [
            GroundTruthEntry(
                query="What is system design in engineering?",
                expected_answer="System design is the process of defining the architecture, modules, interfaces, and data for a system to satisfy specified requirements.",
                expected_sources=["wikipedia_system_design.txt"],
                domain="general_engineering",
                difficulty_level="basic",
                answer_type="factual"
            ),
            GroundTruthEntry(
                query="How does industrial design relate to chemical engineering?",
                expected_answer="Industrial design in chemical engineering focuses on process layout, equipment selection, safety considerations, and optimization of manufacturing systems.",
                expected_sources=["wikipedia_industrial_design.txt"],
                domain="general_engineering",
                difficulty_level="intermediate",
                answer_type="conceptual"
            )
        ]
    
    def _save_dataset(self, dataset: List[GroundTruthEntry], filepath: Path):
        """Save dataset to JSON file."""
        data = []
        for entry in dataset:
            data.append({
                "query": entry.query,
                "expected_answer": entry.expected_answer,
                "expected_sources": entry.expected_sources,
                "domain": entry.domain,
                "difficulty_level": entry.difficulty_level,
                "answer_type": entry.answer_type
            })
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def validate_domain(self, domain: str = "all") -> Dict[str, Any]:
        """Validate specific domain or all domains.
        
        Args:
            domain: Domain to validate ("all" for all domains)
            
        Returns:
            Comprehensive validation results
        """
        if not self.rag_pipeline:
            raise ValueError("RAG pipeline not provided for validation")
        
        domains_to_test = [domain] if domain != "all" else list(self.ground_truth_datasets.keys())
        
        all_results = []
        domain_scores = {}
        
        for test_domain in domains_to_test:
            if test_domain not in self.ground_truth_datasets:
                self.logger.warning(f"Domain '{test_domain}' not found in ground truth datasets")
                continue
            
            domain_results = []
            dataset = self.ground_truth_datasets[test_domain]
            
            self.logger.info(f"Validating domain: {test_domain} ({len(dataset)} queries)")
            
            for entry in dataset:
                result = self._validate_single_query(entry)
                domain_results.append(result)
                all_results.append(result)
            
            # Calculate domain-specific metrics
            domain_scores[test_domain] = self._calculate_domain_metrics(domain_results)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(all_results)
        
        # Generate comprehensive report
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(all_results),
            "domains_tested": domains_to_test,
            "overall_metrics": overall_metrics,
            "domain_metrics": domain_scores,
            "detailed_results": [self._result_to_dict(r) for r in all_results],
            "recommendations": self._generate_recommendations(overall_metrics, domain_scores)
        }
        
        # Save validation report
        self._save_validation_report(validation_report)
        
        return validation_report
    
    def _validate_single_query(self, entry: GroundTruthEntry) -> ValidationResult:
        """Validate a single query against ground truth."""
        start_time = datetime.now()
        
        try:
            # Get response from RAG pipeline
            response = self.rag_pipeline.query(entry.query)
            
            # Extract answer and sources
            if isinstance(response, dict):
                generated_answer = response.get('answer', str(response))
                sources_used = response.get('sources', [])
            else:
                generated_answer = str(response)
                sources_used = []
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate accuracy scores
            accuracy_score = self._calculate_answer_accuracy(
                entry.expected_answer, generated_answer
            )
            
            citation_accuracy = self._calculate_citation_accuracy(
                entry.expected_sources, sources_used
            )
            
            return ValidationResult(
                query=entry.query,
                expected_answer=entry.expected_answer,
                generated_answer=generated_answer,
                sources_used=sources_used,
                accuracy_score=accuracy_score,
                citation_accuracy=citation_accuracy,
                response_time=response_time,
                validation_notes=f"Domain: {entry.domain}, Level: {entry.difficulty_level}"
            )
            
        except Exception as e:
            self.logger.error(f"Error validating query '{entry.query}': {e}")
            return ValidationResult(
                query=entry.query,
                expected_answer=entry.expected_answer,
                generated_answer=f"ERROR: {str(e)}",
                sources_used=[],
                accuracy_score=0.0,
                citation_accuracy=0.0,
                response_time=(datetime.now() - start_time).total_seconds(),
                validation_notes=f"Validation failed: {str(e)}"
            )
    
    def _calculate_answer_accuracy(self, expected: str, generated: str) -> float:
        """Calculate accuracy score between expected and generated answers."""
        if not generated or generated.startswith("ERROR:"):
            return 0.0
        
        # Simple semantic similarity (can be enhanced with embeddings)
        expected_lower = expected.lower()
        generated_lower = generated.lower()
        
        # Check for key terms overlap
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())
        
        if not expected_words:
            return 0.0
        
        overlap = len(expected_words.intersection(generated_words))
        accuracy = overlap / len(expected_words)
        
        # Bonus for mentioning key concepts
        key_concepts = ["modular", "distillation", "reactor", "efficiency", "chemical", "plant"]
        concept_bonus = sum(1 for concept in key_concepts if concept in generated_lower) * 0.1
        
        return min(1.0, accuracy + concept_bonus)
    
    def _calculate_citation_accuracy(self, expected_sources: List[str], actual_sources: List[str]) -> float:
        """Calculate citation accuracy."""
        if not expected_sources:
            return 1.0  # No sources expected
        
        if not actual_sources:
            return 0.0  # Sources expected but none provided
        
        # Check for partial matches in source names
        matches = 0
        for expected in expected_sources:
            for actual in actual_sources:
                if expected.lower() in actual.lower() or actual.lower() in expected.lower():
                    matches += 1
                    break
        
        return matches / len(expected_sources)
    
    def _calculate_domain_metrics(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate metrics for a domain."""
        if not results:
            return {}
        
        accuracy_scores = [r.accuracy_score for r in results]
        citation_scores = [r.citation_accuracy for r in results]
        response_times = [r.response_time for r in results]
        
        return {
            "avg_accuracy": sum(accuracy_scores) / len(accuracy_scores),
            "min_accuracy": min(accuracy_scores),
            "max_accuracy": max(accuracy_scores),
            "avg_citation_accuracy": sum(citation_scores) / len(citation_scores),
            "avg_response_time": sum(response_times) / len(response_times),
            "total_queries": len(results),
            "high_accuracy_queries": len([s for s in accuracy_scores if s >= 0.7]),
            "low_accuracy_queries": len([s for s in accuracy_scores if s < 0.3])
        }
    
    def _calculate_overall_metrics(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate overall validation metrics."""
        if not results:
            return {}
        
        accuracy_scores = [r.accuracy_score for r in results]
        citation_scores = [r.citation_accuracy for r in results]
        response_times = [r.response_time for r in results]
        
        return {
            "overall_accuracy": sum(accuracy_scores) / len(accuracy_scores),
            "overall_citation_accuracy": sum(citation_scores) / len(citation_scores),
            "avg_response_time": sum(response_times) / len(response_times),
            "total_queries": len(results),
            "pass_rate": len([s for s in accuracy_scores if s >= 0.5]) / len(accuracy_scores),
            "excellent_rate": len([s for s in accuracy_scores if s >= 0.8]) / len(accuracy_scores),
            "citation_coverage": len([s for s in citation_scores if s > 0]) / len(citation_scores)
        }
    
    def _generate_recommendations(self, overall_metrics: Dict, domain_metrics: Dict) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Overall accuracy recommendations
        if overall_metrics.get("overall_accuracy", 0) < 0.5:
            recommendations.append("Overall accuracy is below 50%. Consider improving document quality or chunking strategy.")
        
        if overall_metrics.get("overall_citation_accuracy", 0) < 0.6:
            recommendations.append("Citation accuracy is low. Improve source attribution in responses.")
        
        if overall_metrics.get("avg_response_time", 0) > 2.0:
            recommendations.append("Response times are high. Consider optimizing vector search or reducing index size.")
        
        # Domain-specific recommendations
        for domain, metrics in domain_metrics.items():
            if metrics.get("avg_accuracy", 0) < 0.4:
                recommendations.append(f"Domain '{domain}' shows low accuracy. Add more domain-specific documents.")
        
        # Citation recommendations
        if overall_metrics.get("citation_coverage", 0) < 0.8:
            recommendations.append("Many responses lack source citations. Implement citation generation in responses.")
        
        return recommendations
    
    def _result_to_dict(self, result: ValidationResult) -> Dict:
        """Convert ValidationResult to dictionary."""
        return {
            "query": result.query,
            "expected_answer": result.expected_answer,
            "generated_answer": result.generated_answer,
            "sources_used": result.sources_used,
            "accuracy_score": result.accuracy_score,
            "citation_accuracy": result.citation_accuracy,
            "response_time": result.response_time,
            "expert_rating": result.expert_rating,
            "validation_notes": result.validation_notes
        }
    
    def _save_validation_report(self, report: Dict):
        """Save validation report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Also save a summary CSV
        self._save_validation_summary_csv(report, timestamp)
        
        self.logger.info(f"Validation report saved to: {report_file}")
    
    def _save_validation_summary_csv(self, report: Dict, timestamp: str):
        """Save validation summary as CSV."""
        summary_data = []
        
        for result in report["detailed_results"]:
            summary_data.append({
                "query": result["query"],
                "accuracy_score": result["accuracy_score"],
                "citation_accuracy": result["citation_accuracy"],
                "response_time": result["response_time"],
                "sources_count": len(result["sources_used"]),
                "validation_notes": result["validation_notes"]
            })
        
        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / f"validation_summary_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
    
    def add_expert_evaluation(self, query: str, expert_rating: float, notes: str = ""):
        """Add expert evaluation for a specific query."""
        # This would be called by domain experts to rate responses
        expert_eval = {
            "query": query,
            "expert_rating": expert_rating,
            "notes": notes,
            "timestamp": datetime.now().isoformat()
        }
        
        expert_file = self.results_dir / "expert_evaluations.jsonl"
        with open(expert_file, 'a', encoding='utf-8') as f:
            json.dump(expert_eval, f)
            f.write('\n')
    
    def generate_validation_dashboard(self) -> str:
        """Generate HTML dashboard for validation results."""
        # This would create an interactive dashboard
        # For now, return a simple summary
        latest_report = self._get_latest_validation_report()
        
        if not latest_report:
            return "No validation reports found."
        
        dashboard_html = f"""
        <h1>PyNucleus Validation Dashboard</h1>
        <h2>Overall Performance</h2>
        <ul>
            <li>Overall Accuracy: {latest_report['overall_metrics']['overall_accuracy']:.2%}</li>
            <li>Citation Accuracy: {latest_report['overall_metrics']['overall_citation_accuracy']:.2%}</li>
            <li>Pass Rate: {latest_report['overall_metrics']['pass_rate']:.2%}</li>
            <li>Total Queries: {latest_report['overall_metrics']['total_queries']}</li>
        </ul>
        
        <h2>Domain Performance</h2>
        <ul>
        """
        
        for domain, metrics in latest_report['domain_metrics'].items():
            dashboard_html += f"<li>{domain}: {metrics['avg_accuracy']:.2%} accuracy</li>"
        
        dashboard_html += """
        </ul>
        
        <h2>Recommendations</h2>
        <ul>
        """
        
        for rec in latest_report['recommendations']:
            dashboard_html += f"<li>{rec}</li>"
        
        dashboard_html += "</ul>"
        
        return dashboard_html
    
    def _get_latest_validation_report(self) -> Optional[Dict]:
        """Get the latest validation report."""
        report_files = list(self.results_dir.glob("validation_report_*.json"))
        
        if not report_files:
            return None
        
        latest_file = max(report_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f) 