"""
Comprehensive Validation System Integration

This module integrates all validation components into a unified system that works
with the existing plant builder and financial analyzer:

- Feedstock Selection Validation
- Location Factor Adjustment
- Quantitative Risk Assessment
- Economic Benchmarking
- Operational Hours Validation
- Expert Review System

Provides a single interface for comprehensive plant validation and analysis.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from .feedstock_validator import FeedstockValidator, FeedstockType, PlantType as FeedstockPlantType
from .location_factor_analyzer import LocationFactorAnalyzer
from .quantitative_risk_assessor import QuantitativeRiskAssessor, PlantCategory
from .economic_benchmarking import EconomicBenchmarkingSystem, PlantCategory as EconPlantCategory
from .operational_hours_validator import OperationalHoursValidator, PlantType as OHPlantType
from .expert_review_system import ExpertReviewSystem, ValidationComponent
from .plant_builder import PlantBuilder
from .financial_analyzer import FinancialAnalyzer

from ..utils.logger import logger


class ValidationStatus(Enum):
    """Overall validation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass
class ComprehensiveValidationResult:
    """Comprehensive validation results"""
    plant_name: str
    location: str
    capacity: float
    technology: str
    feedstock: str
    
    # Individual validation results
    feedstock_validation: Dict[str, Any]
    location_factor_analysis: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    economic_benchmarking: Dict[str, Any]
    operational_hours_validation: Dict[str, Any]
    
    # Plant configuration and financial analysis
    plant_configuration: Dict[str, Any]
    financial_analysis: Dict[str, Any]
    
    # Overall assessment
    overall_status: ValidationStatus
    overall_score: float  # 0-100
    confidence_level: float  # 0-1
    
    # Key findings and recommendations
    key_findings: List[str]
    recommendations: List[str]
    risk_factors: List[str]
    
    # Expert review (if conducted)
    expert_review: Optional[Dict[str, Any]] = None
    
    # Validation metadata
    validation_timestamp: datetime = field(default_factory=datetime.now)
    validation_version: str = "1.0"


class ComprehensiveValidationSystem:
    """
    Comprehensive validation system that integrates all validation components
    
    Provides a unified interface for validating plant configurations using
    all available validation modules and generating comprehensive reports.
    """
    
    def __init__(self, data_dir: str = "data/comprehensive_validation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all validation components
        self.feedstock_validator = FeedstockValidator()
        self.location_factor_analyzer = LocationFactorAnalyzer()
        self.risk_assessor = QuantitativeRiskAssessor()
        self.economic_benchmarking = EconomicBenchmarkingSystem()
        self.operational_hours_validator = OperationalHoursValidator()
        self.expert_review_system = ExpertReviewSystem()
        
        # Initialize plant builder and financial analyzer
        self.plant_builder = PlantBuilder()
        self.financial_analyzer = FinancialAnalyzer()
        
        # Validation weights for overall scoring
        self.validation_weights = {
            'feedstock': 0.20,
            'location_factors': 0.15,
            'risk_assessment': 0.25,
            'economic_benchmarking': 0.25,
            'operational_hours': 0.15
        }
        
        # Plant type mappings
        self.plant_type_mappings = self._initialize_plant_type_mappings()
        
        logger.info("ComprehensiveValidationSystem initialized with all components")
    
    def _initialize_plant_type_mappings(self) -> Dict[str, Dict[str, Any]]:
        """Initialize plant type mappings for different validation modules"""
        return {
            "Haber-Bosch Process": {
                "feedstock_plant_type": FeedstockPlantType.FERTILIZER,
                "risk_plant_category": PlantCategory.FERTILIZER,
                "econ_plant_category": EconPlantCategory.FERTILIZER,
                "oh_plant_type": OHPlantType.FERTILIZER
            },
            "Biomass Conversion": {
                "feedstock_plant_type": FeedstockPlantType.BIOFUEL,
                "risk_plant_category": PlantCategory.BIOFUEL,
                "econ_plant_category": EconPlantCategory.BIOFUEL,
                "oh_plant_type": OHPlantType.BIOFUEL
            },
            "Fischer-Tropsch": {
                "feedstock_plant_type": FeedstockPlantType.PETROCHEMICAL,
                "risk_plant_category": PlantCategory.PETROCHEMICAL,
                "econ_plant_category": EconPlantCategory.PETROCHEMICAL,
                "oh_plant_type": OHPlantType.PETROCHEMICAL
            },
            "Catalytic Cracking": {
                "feedstock_plant_type": FeedstockPlantType.PETROCHEMICAL,
                "risk_plant_category": PlantCategory.PETROCHEMICAL,
                "econ_plant_category": EconPlantCategory.PETROCHEMICAL,
                "oh_plant_type": OHPlantType.PETROCHEMICAL
            },
            "Transesterification": {
                "feedstock_plant_type": FeedstockPlantType.BIOFUEL,
                "risk_plant_category": PlantCategory.BIOFUEL,
                "econ_plant_category": EconPlantCategory.BIOFUEL,
                "oh_plant_type": OHPlantType.BIOFUEL
            }
        }
    
    def validate_plant_comprehensive(self, 
                                   template_id: int,
                                   custom_parameters: Dict[str, Any],
                                   expert_id: Optional[str] = None) -> ComprehensiveValidationResult:
        """
        Conduct comprehensive validation of a plant configuration
        
        Args:
            template_id: Plant template ID
            custom_parameters: Custom plant parameters
            expert_id: Optional expert ID for expert review
            
        Returns:
            ComprehensiveValidationResult with all validation results
        """
        logger.info(f"Starting comprehensive validation for template {template_id}")
        
        # Step 1: Build plant configuration
        plant_config = self.plant_builder.build_plant(template_id, custom_parameters)
        
        # Step 2: Conduct financial analysis
        financial_analysis = self.financial_analyzer.calculate_financial_metrics(plant_config)
        
        # Extract key parameters
        template_info = plant_config.get("template_info", {})
        parameters = plant_config.get("parameters", {})
        
        plant_name = template_info.get("name", "Unknown Plant")
        location = parameters.get("plant_location", "Unknown Location")
        capacity = parameters.get("production_capacity", 0)
        technology = template_info.get("technology", "Unknown Technology")
        feedstock = parameters.get("feedstock", "Unknown Feedstock")
        operating_hours = parameters.get("operating_hours", 8000)
        
        # Get plant type mappings
        type_mappings = self.plant_type_mappings.get(technology, {})
        
        # Step 3: Conduct individual validations
        validation_results = {}
        
        # Feedstock validation
        try:
            feedstock_plant_type = type_mappings.get("feedstock_plant_type", FeedstockPlantType.FERTILIZER)
            feedstock_options = template_info.get("feedstock_options", [feedstock])
            
            feedstock_evaluations = self.feedstock_validator.evaluate_feedstock_options(
                technology, location, feedstock_options, capacity, operating_hours
            )
            
            validation_results["feedstock_validation"] = {
                "evaluations": [
                    {
                        "feedstock": eval_result.feedstock.value,
                        "composite_score": eval_result.composite_score,
                        "availability_score": eval_result.availability_score,
                        "technical_compatibility_score": eval_result.technical_compatibility_score,
                        "cost_efficiency_score": eval_result.cost_efficiency_score,
                        "recommendations": eval_result.recommendations,
                        "risk_factors": eval_result.risk_factors
                    }
                    for eval_result in feedstock_evaluations
                ],
                "status": "completed",
                "best_option": feedstock_evaluations[0].feedstock.value if feedstock_evaluations else feedstock,
                "best_score": feedstock_evaluations[0].composite_score if feedstock_evaluations else 0.5
            }
        except Exception as e:
            logger.error(f"Feedstock validation failed: {e}")
            validation_results["feedstock_validation"] = {
                "status": "failed",
                "error": str(e),
                "best_score": 0.5
            }
        
        # Location factor analysis
        try:
            location_analysis = self.location_factor_analyzer.analyze_location_factors(location)
            
            validation_results["location_factor_analysis"] = {
                "capital_cost_factor": location_analysis.capital_cost_factor,
                "operational_cost_factor": location_analysis.operational_cost_factor,
                "total_risk_score": location_analysis.total_risk_score,
                "key_advantages": location_analysis.key_advantages,
                "key_challenges": location_analysis.key_challenges,
                "mitigation_strategies": location_analysis.mitigation_strategies,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Location factor analysis failed: {e}")
            validation_results["location_factor_analysis"] = {
                "status": "failed",
                "error": str(e),
                "capital_cost_factor": 1.0,
                "operational_cost_factor": 1.0,
                "total_risk_score": 0.5
            }
        
        # Risk assessment
        try:
            risk_plant_category = type_mappings.get("risk_plant_category", PlantCategory.FERTILIZER)
            product_name = self._extract_product_name(plant_name)
            
            risk_assessment = self.risk_assessor.assess_comprehensive_risk(
                location, product_name, technology, "medium"
            )
            
            validation_results["risk_assessment"] = {
                "composite_score": risk_assessment.composite_score,
                "overall_risk_level": risk_assessment.overall_risk_level.value,
                "risk_breakdown": risk_assessment.risk_breakdown,
                "critical_risk_factors": risk_assessment.critical_risk_factors,
                "recommended_actions": risk_assessment.recommended_actions,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            validation_results["risk_assessment"] = {
                "status": "failed",
                "error": str(e),
                "composite_score": 5.0,
                "overall_risk_level": "Medium"
            }
        
        # Economic benchmarking
        try:
            econ_plant_category = type_mappings.get("econ_plant_category", EconPlantCategory.FERTILIZER)
            financial_params = plant_config.get("financial_parameters", {})
            
            capital_cost = financial_params.get("capital_cost", 0)
            operating_cost = financial_params.get("operating_cost", 0)
            
            economic_benchmark = self.economic_benchmarking.benchmark_plant_costs(
                plant_name, location, capacity, econ_plant_category, capital_cost, operating_cost
            )
            
            validation_results["economic_benchmarking"] = {
                "overall_assessment": economic_benchmark.overall_assessment,
                "competitiveness_score": economic_benchmark.competitiveness_score,
                "data_quality_score": economic_benchmark.data_quality_score,
                "capital_cost_deviation": economic_benchmark.capital_cost_comparison.deviation_percentage,
                "operating_cost_deviation": economic_benchmark.operating_cost_comparison.deviation_percentage,
                "recommendations": economic_benchmark.recommendations,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Economic benchmarking failed: {e}")
            validation_results["economic_benchmarking"] = {
                "status": "failed",
                "error": str(e),
                "competitiveness_score": 50.0,
                "data_quality_score": 0.5
            }
        
        # Operational hours validation
        try:
            oh_plant_type = type_mappings.get("oh_plant_type", OHPlantType.FERTILIZER)
            
            oh_validation = self.operational_hours_validator.validate_operational_hours(
                oh_plant_type, location, capacity, operating_hours
            )
            
            validation_results["operational_hours_validation"] = {
                "proposed_hours": oh_validation.proposed_hours,
                "validated_range": oh_validation.validated_range,
                "recommended_hours": oh_validation.recommended_hours,
                "confidence_level": oh_validation.confidence_level,
                "supporting_evidence": oh_validation.supporting_evidence,
                "risk_factors": oh_validation.risk_factors,
                "recommendations": oh_validation.recommendations,
                "status": "completed"
            }
        except Exception as e:
            logger.error(f"Operational hours validation failed: {e}")
            validation_results["operational_hours_validation"] = {
                "status": "failed",
                "error": str(e),
                "proposed_hours": operating_hours,
                "recommended_hours": operating_hours,
                "confidence_level": 0.5
            }
        
        # Step 4: Calculate overall assessment
        overall_assessment = self._calculate_overall_assessment(validation_results)
        
        # Step 5: Generate comprehensive findings and recommendations
        key_findings = self._generate_key_findings(validation_results, financial_analysis)
        recommendations = self._generate_comprehensive_recommendations(validation_results)
        risk_factors = self._identify_comprehensive_risk_factors(validation_results)
        
        # Step 6: Conduct expert review if requested
        expert_review = None
        if expert_id:
            try:
                expert_review = self._conduct_expert_review(
                    plant_name, location, capacity, expert_id, plant_config, validation_results
                )
            except Exception as e:
                logger.error(f"Expert review failed: {e}")
                expert_review = {"status": "failed", "error": str(e)}
        
        # Create comprehensive result
        result = ComprehensiveValidationResult(
            plant_name=plant_name,
            location=location,
            capacity=capacity,
            technology=technology,
            feedstock=feedstock,
            feedstock_validation=validation_results.get("feedstock_validation", {}),
            location_factor_analysis=validation_results.get("location_factor_analysis", {}),
            risk_assessment=validation_results.get("risk_assessment", {}),
            economic_benchmarking=validation_results.get("economic_benchmarking", {}),
            operational_hours_validation=validation_results.get("operational_hours_validation", {}),
            plant_configuration=plant_config,
            financial_analysis=financial_analysis,
            overall_status=overall_assessment["status"],
            overall_score=overall_assessment["score"],
            confidence_level=overall_assessment["confidence"],
            key_findings=key_findings,
            recommendations=recommendations,
            risk_factors=risk_factors,
            expert_review=expert_review
        )
        
        logger.info(f"Comprehensive validation completed with overall score: {overall_assessment['score']:.1f}")
        
        return result
    
    def _extract_product_name(self, plant_name: str) -> str:
        """Extract product name from plant name"""
        product_mappings = {
            "fertilizer": "Fertilizer",
            "biofuel": "Biofuel",
            "petrochemical": "Petrochemical",
            "ammonia": "Ammonia",
            "ethanol": "Ethanol",
            "methanol": "Methanol"
        }
        
        plant_name_lower = plant_name.lower()
        for key, value in product_mappings.items():
            if key in plant_name_lower:
                return value
        
        return "Chemical Product"
    
    def _calculate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall assessment from individual validation results"""
        component_scores = {}
        
        # Extract scores from each validation
        feedstock_result = validation_results.get("feedstock_validation", {})
        if feedstock_result.get("status") == "completed":
            component_scores["feedstock"] = feedstock_result.get("best_score", 0.5) * 100
        
        location_result = validation_results.get("location_factor_analysis", {})
        if location_result.get("status") == "completed":
            # Convert risk score to quality score (invert)
            risk_score = location_result.get("total_risk_score", 0.5)
            component_scores["location_factors"] = (1.0 - risk_score) * 100
        
        risk_result = validation_results.get("risk_assessment", {})
        if risk_result.get("status") == "completed":
            # Convert risk score to quality score (invert)
            risk_score = risk_result.get("composite_score", 5.0)
            component_scores["risk_assessment"] = (10.0 - risk_score) * 10
        
        econ_result = validation_results.get("economic_benchmarking", {})
        if econ_result.get("status") == "completed":
            component_scores["economic_benchmarking"] = econ_result.get("competitiveness_score", 50.0)
        
        oh_result = validation_results.get("operational_hours_validation", {})
        if oh_result.get("status") == "completed":
            component_scores["operational_hours"] = oh_result.get("confidence_level", 0.5) * 100
        
        # Calculate weighted overall score
        total_score = 0
        total_weight = 0
        
        for component, score in component_scores.items():
            weight = self.validation_weights.get(component, 0.2)
            total_score += score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 50.0
        
        # Calculate confidence level
        completed_validations = sum(1 for result in validation_results.values() 
                                  if result.get("status") == "completed")
        total_validations = len(validation_results)
        confidence_level = completed_validations / total_validations if total_validations > 0 else 0.5
        
        # Determine overall status
        if overall_score >= 80 and confidence_level >= 0.8:
            status = ValidationStatus.APPROVED
        elif overall_score >= 60 and confidence_level >= 0.6:
            status = ValidationStatus.NEEDS_REVISION
        elif overall_score >= 40:
            status = ValidationStatus.COMPLETED
        else:
            status = ValidationStatus.REJECTED
        
        return {
            "score": overall_score,
            "confidence": confidence_level,
            "status": status,
            "component_scores": component_scores
        }
    
    def _generate_key_findings(self, 
                             validation_results: Dict[str, Any],
                             financial_analysis: Dict[str, Any]) -> List[str]:
        """Generate key findings from all validation results"""
        findings = []
        
        # Feedstock findings
        feedstock_result = validation_results.get("feedstock_validation", {})
        if feedstock_result.get("status") == "completed":
            best_score = feedstock_result.get("best_score", 0.5)
            if best_score >= 0.8:
                findings.append("Excellent feedstock options available with strong technical compatibility")
            elif best_score >= 0.6:
                findings.append("Good feedstock options identified with some considerations")
            else:
                findings.append("Feedstock challenges identified - alternative options should be explored")
        
        # Location findings
        location_result = validation_results.get("location_factor_analysis", {})
        if location_result.get("status") == "completed":
            advantages = location_result.get("key_advantages", [])
            challenges = location_result.get("key_challenges", [])
            if len(advantages) > len(challenges):
                findings.append("Location offers good advantages for plant development")
            elif len(challenges) > len(advantages):
                findings.append("Location presents significant challenges requiring mitigation")
        
        # Risk findings
        risk_result = validation_results.get("risk_assessment", {})
        if risk_result.get("status") == "completed":
            risk_level = risk_result.get("overall_risk_level", "Medium")
            findings.append(f"Overall project risk level assessed as {risk_level}")
        
        # Economic findings
        econ_result = validation_results.get("economic_benchmarking", {})
        if econ_result.get("status") == "completed":
            competitiveness = econ_result.get("competitiveness_score", 50.0)
            if competitiveness >= 70:
                findings.append("Strong economic competitiveness compared to industry benchmarks")
            elif competitiveness >= 50:
                findings.append("Moderate economic competitiveness with room for improvement")
            else:
                findings.append("Below-average economic competitiveness - significant improvements needed")
        
        # Operational hours findings
        oh_result = validation_results.get("operational_hours_validation", {})
        if oh_result.get("status") == "completed":
            confidence = oh_result.get("confidence_level", 0.5)
            if confidence >= 0.8:
                findings.append("High confidence in operational hours assumptions")
            elif confidence >= 0.6:
                findings.append("Moderate confidence in operational hours with some validation")
            else:
                findings.append("Low confidence in operational hours - further validation needed")
        
        # Financial findings
        financial_metrics = financial_analysis.get("financial_metrics", {})
        roi = financial_metrics.get("roi", 0)
        if roi > 0.15:
            findings.append("Strong financial returns projected")
        elif roi > 0.10:
            findings.append("Moderate financial returns projected")
        else:
            findings.append("Weak financial returns - project viability concerns")
        
        return findings
    
    def _generate_comprehensive_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations from all validation results"""
        recommendations = []
        
        # Collect recommendations from each validation
        for component, result in validation_results.items():
            if result.get("status") == "completed":
                component_recs = result.get("recommendations", [])
                if component_recs:
                    recommendations.extend(component_recs)
        
        # Add general recommendations
        recommendations.extend([
            "Implement comprehensive monitoring system for all validation metrics",
            "Establish regular review cycles for validation assumptions",
            "Develop contingency plans for identified risk factors",
            "Engage with local stakeholders throughout implementation",
            "Consider phased implementation approach to manage risks"
        ])
        
        return recommendations
    
    def _identify_comprehensive_risk_factors(self, validation_results: Dict[str, Any]) -> List[str]:
        """Identify comprehensive risk factors from all validation results"""
        risk_factors = []
        
        # Collect risk factors from each validation
        for component, result in validation_results.items():
            if result.get("status") == "completed":
                component_risks = result.get("risk_factors", [])
                if component_risks:
                    risk_factors.extend(component_risks)
        
        # Add general risk factors
        failed_validations = [comp for comp, result in validation_results.items() 
                            if result.get("status") == "failed"]
        if failed_validations:
            risk_factors.append(f"Validation failures in: {', '.join(failed_validations)}")
        
        return risk_factors
    
    def _conduct_expert_review(self, 
                             plant_name: str,
                             location: str,
                             capacity: float,
                             expert_id: str,
                             plant_config: Dict[str, Any],
                             validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct expert review of validation results"""
        # This would typically involve human expert input
        # Create placeholder structure
        
        expert_review = {
            "expert_id": expert_id,
            "review_timestamp": datetime.now().isoformat(),
            "plant_name": plant_name,
            "location": location,
            "capacity": capacity,
            "component_reviews": {},
            "overall_assessment": "Expert review template generated - requires human input",
            "status": "pending_expert_input"
        }
        
        # Generate review templates for each component
        for component_name, result in validation_results.items():
            if result.get("status") == "completed":
                component_enum = self._map_component_name_to_enum(component_name)
                if component_enum:
                    review_template = self.expert_review_system.generate_review_template(
                        component_enum, plant_config, result
                    )
                    expert_review["component_reviews"][component_name] = review_template
        
        return expert_review
    
    def _map_component_name_to_enum(self, component_name: str) -> Optional[ValidationComponent]:
        """Map component name to ValidationComponent enum"""
        mapping = {
            "feedstock_validation": ValidationComponent.FEEDSTOCK_SELECTION,
            "location_factor_analysis": ValidationComponent.LOCATION_FACTORS,
            "risk_assessment": ValidationComponent.RISK_ASSESSMENT,
            "economic_benchmarking": ValidationComponent.ECONOMIC_BENCHMARKING,
            "operational_hours_validation": ValidationComponent.OPERATIONAL_HOURS
        }
        return mapping.get(component_name)
    
    def generate_comprehensive_report(self, 
                                    validation_result: ComprehensiveValidationResult,
                                    include_detailed_analysis: bool = True) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("="*100)
        report.append("COMPREHENSIVE PLANT VALIDATION REPORT")
        report.append("="*100)
        report.append(f"Plant: {validation_result.plant_name}")
        report.append(f"Location: {validation_result.location}")
        report.append(f"Capacity: {validation_result.capacity:,.0f} tons/year")
        report.append(f"Technology: {validation_result.technology}")
        report.append(f"Feedstock: {validation_result.feedstock}")
        report.append(f"Validation Date: {validation_result.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Validation Version: {validation_result.validation_version}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Overall Status: {validation_result.overall_status.value}")
        report.append(f"Overall Score: {validation_result.overall_score:.1f}/100")
        report.append(f"Confidence Level: {validation_result.confidence_level:.1%}")
        report.append("")
        
        # Key Findings
        report.append("KEY FINDINGS")
        report.append("-" * 50)
        for finding in validation_result.key_findings:
            report.append(f"• {finding}")
        report.append("")
        
        # Risk Factors
        if validation_result.risk_factors:
            report.append("RISK FACTORS")
            report.append("-" * 50)
            for risk in validation_result.risk_factors:
                report.append(f"• {risk}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 50)
        for rec in validation_result.recommendations:
            report.append(f"• {rec}")
        report.append("")
        
        # Validation Component Summary
        report.append("VALIDATION COMPONENT SUMMARY")
        report.append("-" * 50)
        
        # Feedstock validation
        fs_result = validation_result.feedstock_validation
        if fs_result.get("status") == "completed":
            report.append(f"Feedstock Selection: {fs_result.get('best_score', 0.5):.2f} - {fs_result.get('best_option', 'Unknown')}")
        else:
            report.append(f"Feedstock Selection: Failed - {fs_result.get('error', 'Unknown error')}")
        
        # Location factors
        lf_result = validation_result.location_factor_analysis
        if lf_result.get("status") == "completed":
            report.append(f"Location Factors: Capital {lf_result.get('capital_cost_factor', 1.0):.2f}, Operational {lf_result.get('operational_cost_factor', 1.0):.2f}")
        else:
            report.append(f"Location Factors: Failed - {lf_result.get('error', 'Unknown error')}")
        
        # Risk assessment
        ra_result = validation_result.risk_assessment
        if ra_result.get("status") == "completed":
            report.append(f"Risk Assessment: {ra_result.get('overall_risk_level', 'Unknown')} - Score {ra_result.get('composite_score', 5.0):.1f}/10")
        else:
            report.append(f"Risk Assessment: Failed - {ra_result.get('error', 'Unknown error')}")
        
        # Economic benchmarking
        eb_result = validation_result.economic_benchmarking
        if eb_result.get("status") == "completed":
            report.append(f"Economic Benchmarking: {eb_result.get('competitiveness_score', 50.0):.1f}/100 - {eb_result.get('overall_assessment', 'Unknown')}")
        else:
            report.append(f"Economic Benchmarking: Failed - {eb_result.get('error', 'Unknown error')}")
        
        # Operational hours
        oh_result = validation_result.operational_hours_validation
        if oh_result.get("status") == "completed":
            report.append(f"Operational Hours: {oh_result.get('recommended_hours', 8000):,} hours/year (Confidence: {oh_result.get('confidence_level', 0.5):.1%})")
        else:
            report.append(f"Operational Hours: Failed - {oh_result.get('error', 'Unknown error')}")
        
        report.append("")
        
        # Financial Analysis Summary
        financial_metrics = validation_result.financial_analysis.get("financial_metrics", {})
        if financial_metrics:
            report.append("FINANCIAL ANALYSIS SUMMARY")
            report.append("-" * 50)
            report.append(f"Capital Cost: ${financial_metrics.get('capital_cost', 0):,.0f}")
            report.append(f"Operating Cost: ${financial_metrics.get('operating_cost', 0):,.0f}/year")
            report.append(f"Annual Revenue: ${financial_metrics.get('annual_revenue', 0):,.0f}")
            report.append(f"Net Profit: ${financial_metrics.get('net_profit', 0):,.0f}")
            report.append(f"ROI: {financial_metrics.get('roi', 0):.1%}")
            payback_period = financial_metrics.get('payback_period', 'N/A')
            report.append(f"Payback Period: {payback_period}")
            report.append("")
        
        if include_detailed_analysis:
            report.append("DETAILED VALIDATION ANALYSIS")
            report.append("-" * 50)
            
            # Detailed component analysis would go here
            # For brevity, we'll include key details from each component
            
            # Feedstock details
            if fs_result.get("status") == "completed":
                report.append("Feedstock Selection Analysis:")
                evaluations = fs_result.get("evaluations", [])
                for eval_data in evaluations[:3]:  # Top 3 options
                    report.append(f"  {eval_data['feedstock']}: {eval_data['composite_score']:.3f}")
                report.append("")
            
            # Location factor details
            if lf_result.get("status") == "completed":
                report.append("Location Factor Analysis:")
                advantages = lf_result.get("key_advantages", [])
                challenges = lf_result.get("key_challenges", [])
                report.append(f"  Advantages: {len(advantages)}, Challenges: {len(challenges)}")
                report.append("")
            
            # Risk assessment details
            if ra_result.get("status") == "completed":
                report.append("Risk Assessment Details:")
                risk_breakdown = ra_result.get("risk_breakdown", {})
                for category, score in risk_breakdown.items():
                    report.append(f"  {category.capitalize()}: {score:.1f}/10")
                report.append("")
            
            # Economic benchmarking details
            if eb_result.get("status") == "completed":
                report.append("Economic Benchmarking Details:")
                capital_deviation = eb_result.get("capital_cost_deviation", 0)
                operating_deviation = eb_result.get("operating_cost_deviation", 0)
                report.append(f"  Capital Cost Deviation: {capital_deviation:+.1f}%")
                report.append(f"  Operating Cost Deviation: {operating_deviation:+.1f}%")
                report.append("")
            
            # Operational hours details
            if oh_result.get("status") == "completed":
                report.append("Operational Hours Analysis:")
                validated_range = oh_result.get("validated_range", (0, 0))
                report.append(f"  Validated Range: {validated_range[0]:,} - {validated_range[1]:,} hours/year")
                report.append("")
        
        # Expert Review Status
        if validation_result.expert_review:
            report.append("EXPERT REVIEW STATUS")
            report.append("-" * 50)
            expert_status = validation_result.expert_review.get("status", "unknown")
            report.append(f"Expert Review Status: {expert_status}")
            if expert_status == "pending_expert_input":
                report.append("Expert review templates generated - awaiting expert input")
            report.append("")
        
        return "\n".join(report)
    
    def save_validation_results(self, 
                              validation_result: ComprehensiveValidationResult,
                              filename: Optional[str] = None) -> Path:
        """Save comprehensive validation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_validation_{validation_result.plant_name.replace(' ', '_')}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            "plant_name": validation_result.plant_name,
            "location": validation_result.location,
            "capacity": validation_result.capacity,
            "technology": validation_result.technology,
            "feedstock": validation_result.feedstock,
            "overall_status": validation_result.overall_status.value,
            "overall_score": validation_result.overall_score,
            "confidence_level": validation_result.confidence_level,
            "key_findings": validation_result.key_findings,
            "recommendations": validation_result.recommendations,
            "risk_factors": validation_result.risk_factors,
            "validation_timestamp": validation_result.validation_timestamp.isoformat(),
            "validation_version": validation_result.validation_version,
            "feedstock_validation": validation_result.feedstock_validation,
            "location_factor_analysis": validation_result.location_factor_analysis,
            "risk_assessment": validation_result.risk_assessment,
            "economic_benchmarking": validation_result.economic_benchmarking,
            "operational_hours_validation": validation_result.operational_hours_validation,
            "plant_configuration": validation_result.plant_configuration,
            "financial_analysis": validation_result.financial_analysis,
            "expert_review": validation_result.expert_review
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Comprehensive validation results saved to {filepath}")
        return filepath
    
    def get_validation_summary(self, validation_result: ComprehensiveValidationResult) -> Dict[str, Any]:
        """Get a summary of validation results for quick overview"""
        return {
            "plant_name": validation_result.plant_name,
            "location": validation_result.location,
            "capacity": validation_result.capacity,
            "overall_status": validation_result.overall_status.value,
            "overall_score": validation_result.overall_score,
            "confidence_level": validation_result.confidence_level,
            "component_status": {
                "feedstock_validation": validation_result.feedstock_validation.get("status", "unknown"),
                "location_factor_analysis": validation_result.location_factor_analysis.get("status", "unknown"),
                "risk_assessment": validation_result.risk_assessment.get("status", "unknown"),
                "economic_benchmarking": validation_result.economic_benchmarking.get("status", "unknown"),
                "operational_hours_validation": validation_result.operational_hours_validation.get("status", "unknown")
            },
            "key_findings_count": len(validation_result.key_findings),
            "recommendations_count": len(validation_result.recommendations),
            "risk_factors_count": len(validation_result.risk_factors),
            "expert_review_status": validation_result.expert_review.get("status", "not_conducted") if validation_result.expert_review else "not_conducted"
        } 