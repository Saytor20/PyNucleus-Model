"""
Expert Review Template System

This module provides structured expert review workflows for the comprehensive
validation modules including feedstock selection, location factors, risk assessment,
economic benchmarking, and operational hours validation.

Integrates with the existing validation manager to provide comprehensive review capabilities.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..utils.logger import logger
from ..eval.validation_manager import ValidationManager, ExpertProfile, ValidationRecord, ValidationStatus


class ValidationComponent(Enum):
    """Components available for expert review"""
    FEEDSTOCK_SELECTION = "feedstock_selection"
    LOCATION_FACTORS = "location_factors"
    RISK_ASSESSMENT = "risk_assessment"
    ECONOMIC_BENCHMARKING = "economic_benchmarking"
    OPERATIONAL_HOURS = "operational_hours"
    OVERALL_ASSESSMENT = "overall_assessment"


class ReviewCriteria(Enum):
    """Review criteria for validation components"""
    TECHNICAL_ACCURACY = "technical_accuracy"
    DATA_QUALITY = "data_quality"
    METHODOLOGY = "methodology"
    REGIONAL_RELEVANCE = "regional_relevance"
    PRACTICAL_APPLICABILITY = "practical_applicability"
    COMPLETENESS = "completeness"


@dataclass
class ReviewQuestion:
    """Individual review question"""
    question_id: str
    component: ValidationComponent
    criteria: ReviewCriteria
    question_text: str
    required: bool
    response_type: str  # "rating", "text", "boolean", "multiple_choice"
    rating_scale: Optional[Tuple[int, int]] = None
    choices: Optional[List[str]] = None
    guidance: str = ""


@dataclass
class ReviewResponse:
    """Expert response to a review question"""
    question_id: str
    expert_id: str
    response_value: Any
    confidence_level: float  # 0-1
    comments: str
    evidence_sources: List[str]
    response_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComponentReview:
    """Complete review of a validation component"""
    component: ValidationComponent
    expert_id: str
    plant_configuration: Dict[str, Any]
    validation_results: Dict[str, Any]
    review_responses: List[ReviewResponse]
    overall_rating: float  # 0-10
    overall_confidence: float  # 0-1
    key_findings: List[str]
    recommendations: List[str]
    approval_status: ValidationStatus
    review_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ComprehensiveReview:
    """Comprehensive review of all validation components"""
    plant_name: str
    location: str
    capacity: float
    expert_id: str
    component_reviews: List[ComponentReview]
    overall_assessment: str
    overall_confidence: float
    critical_issues: List[str]
    recommendations: List[str]
    approval_status: ValidationStatus
    review_timestamp: datetime = field(default_factory=datetime.now)


class ExpertReviewSystem:
    """
    Expert review system for comprehensive validation components
    
    Provides structured review workflows and integrates with the existing
    validation manager for comprehensive quality assurance.
    """
    
    def __init__(self, 
                 validation_manager: Optional[ValidationManager] = None,
                 data_dir: str = "data/expert_reviews"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validation manager
        self.validation_manager = validation_manager or ValidationManager()
        
        # Initialize review questions
        self.review_questions = self._initialize_review_questions()
        
        # Review templates
        self.review_templates = self._initialize_review_templates()
        
        # Scoring weights
        self.component_weights = {
            ValidationComponent.FEEDSTOCK_SELECTION: 0.25,
            ValidationComponent.LOCATION_FACTORS: 0.20,
            ValidationComponent.RISK_ASSESSMENT: 0.25,
            ValidationComponent.ECONOMIC_BENCHMARKING: 0.20,
            ValidationComponent.OPERATIONAL_HOURS: 0.10
        }
        
        logger.info("ExpertReviewSystem initialized")
    
    def _initialize_review_questions(self) -> List[ReviewQuestion]:
        """Initialize structured review questions for each component"""
        questions = []
        
        # Feedstock Selection Questions
        questions.extend([
            ReviewQuestion(
                question_id="fs_001",
                component=ValidationComponent.FEEDSTOCK_SELECTION,
                criteria=ReviewCriteria.TECHNICAL_ACCURACY,
                question_text="Does the ranked feedstock list accurately reflect regional economic and technical realities?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Consider local availability, market conditions, and technical feasibility in African contexts"
            ),
            ReviewQuestion(
                question_id="fs_002",
                component=ValidationComponent.FEEDSTOCK_SELECTION,
                criteria=ReviewCriteria.DATA_QUALITY,
                question_text="Are the availability consistency scores based on reliable annual data sources?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate data sources, collection methods, and temporal coverage"
            ),
            ReviewQuestion(
                question_id="fs_003",
                component=ValidationComponent.FEEDSTOCK_SELECTION,
                criteria=ReviewCriteria.METHODOLOGY,
                question_text="Is the technical compatibility scoring methodology appropriate for the selected plant technology?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess compatibility matrix, conversion efficiencies, and technical risk factors"
            ),
            ReviewQuestion(
                question_id="fs_004",
                component=ValidationComponent.FEEDSTOCK_SELECTION,
                criteria=ReviewCriteria.REGIONAL_RELEVANCE,
                question_text="Are the cost efficiency calculations representative of regional market conditions?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Review cost data sources, regional adjustments, and market dynamics"
            ),
            ReviewQuestion(
                question_id="fs_005",
                component=ValidationComponent.FEEDSTOCK_SELECTION,
                criteria=ReviewCriteria.PRACTICAL_APPLICABILITY,
                question_text="Are the feedstock recommendations practically implementable in the target region?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Consider logistics, infrastructure, regulatory, and supply chain factors"
            )
        ])
        
        # Location Factors Questions
        questions.extend([
            ReviewQuestion(
                question_id="lf_001",
                component=ValidationComponent.LOCATION_FACTORS,
                criteria=ReviewCriteria.TECHNICAL_ACCURACY,
                question_text="Are separate capital vs. operating cost factors more accurate for financial modeling?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate the separation of cost components and their individual adjustment factors"
            ),
            ReviewQuestion(
                question_id="lf_002",
                component=ValidationComponent.LOCATION_FACTORS,
                criteria=ReviewCriteria.DATA_QUALITY,
                question_text="Are the infrastructure quality indexes based on reliable and current data?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess data sources, update frequency, and measurement methodologies"
            ),
            ReviewQuestion(
                question_id="lf_003",
                component=ValidationComponent.LOCATION_FACTORS,
                criteria=ReviewCriteria.METHODOLOGY,
                question_text="Is the weighting scheme for infrastructure, labor, and regulatory factors appropriate?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Review factor weights, calculation methods, and sensitivity analysis"
            ),
            ReviewQuestion(
                question_id="lf_004",
                component=ValidationComponent.LOCATION_FACTORS,
                criteria=ReviewCriteria.REGIONAL_RELEVANCE,
                question_text="Do the location adjustments accurately reflect regional differences?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate regional multipliers, local market conditions, and comparative analysis"
            )
        ])
        
        # Risk Assessment Questions
        questions.extend([
            ReviewQuestion(
                question_id="ra_001",
                component=ValidationComponent.RISK_ASSESSMENT,
                criteria=ReviewCriteria.TECHNICAL_ACCURACY,
                question_text="Do the quantifiable thresholds provide actionable insights for risk mitigation?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess threshold levels, action points, and practical applicability"
            ),
            ReviewQuestion(
                question_id="ra_002",
                component=ValidationComponent.RISK_ASSESSMENT,
                criteria=ReviewCriteria.DATA_QUALITY,
                question_text="Are the political stability and infrastructure quality scores based on credible sources?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate data sources, methodologies, and update mechanisms"
            ),
            ReviewQuestion(
                question_id="ra_003",
                component=ValidationComponent.RISK_ASSESSMENT,
                criteria=ReviewCriteria.METHODOLOGY,
                question_text="Is the composite risk scoring methodology robust and well-balanced?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Review risk categories, weighting schemes, and aggregation methods"
            ),
            ReviewQuestion(
                question_id="ra_004",
                component=ValidationComponent.RISK_ASSESSMENT,
                criteria=ReviewCriteria.COMPLETENESS,
                question_text="Are all major risk categories adequately covered in the assessment?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Check coverage of political, economic, operational, and technical risks"
            )
        ])
        
        # Economic Benchmarking Questions
        questions.extend([
            ReviewQuestion(
                question_id="eb_001",
                component=ValidationComponent.ECONOMIC_BENCHMARKING,
                criteria=ReviewCriteria.DATA_QUALITY,
                question_text="Are the benchmarking sources robust enough for accurate cost estimation?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate source credibility, data quality, and industry representativeness"
            ),
            ReviewQuestion(
                question_id="eb_002",
                component=ValidationComponent.ECONOMIC_BENCHMARKING,
                criteria=ReviewCriteria.METHODOLOGY,
                question_text="Is the percentage deviation calculation methodology appropriate?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess calculation methods, statistical approaches, and confidence intervals"
            ),
            ReviewQuestion(
                question_id="eb_003",
                component=ValidationComponent.ECONOMIC_BENCHMARKING,
                criteria=ReviewCriteria.REGIONAL_RELEVANCE,
                question_text="Are the regional cost adjustments and scaling factors appropriate?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Review regional multipliers, capacity scaling, and local market conditions"
            ),
            ReviewQuestion(
                question_id="eb_004",
                component=ValidationComponent.ECONOMIC_BENCHMARKING,
                criteria=ReviewCriteria.PRACTICAL_APPLICABILITY,
                question_text="Do the benchmarking results provide actionable insights for cost optimization?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess practical utility, recommendation quality, and implementation feasibility"
            )
        ])
        
        # Operational Hours Questions
        questions.extend([
            ReviewQuestion(
                question_id="oh_001",
                component=ValidationComponent.OPERATIONAL_HOURS,
                criteria=ReviewCriteria.DATA_QUALITY,
                question_text="Does historical data adequately justify the default operational hour range?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Evaluate historical data quality, sample size, and representativeness"
            ),
            ReviewQuestion(
                question_id="oh_002",
                component=ValidationComponent.OPERATIONAL_HOURS,
                criteria=ReviewCriteria.METHODOLOGY,
                question_text="Are maintenance downtime and seasonal variance calculations appropriate?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Review maintenance schedules, seasonal factors, and calculation methods"
            ),
            ReviewQuestion(
                question_id="oh_003",
                component=ValidationComponent.OPERATIONAL_HOURS,
                criteria=ReviewCriteria.REGIONAL_RELEVANCE,
                question_text="Are the operational hour recommendations realistic for the target region?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Consider local conditions, infrastructure, and operational constraints"
            ),
            ReviewQuestion(
                question_id="oh_004",
                component=ValidationComponent.OPERATIONAL_HOURS,
                criteria=ReviewCriteria.PRACTICAL_APPLICABILITY,
                question_text="Are the evidence-based justifications convincing and actionable?",
                required=True,
                response_type="rating",
                rating_scale=(1, 10),
                guidance="Assess supporting evidence, justification quality, and implementation guidance"
            )
        ])
        
        return questions
    
    def _initialize_review_templates(self) -> Dict[ValidationComponent, Dict[str, Any]]:
        """Initialize review templates for each component"""
        return {
            ValidationComponent.FEEDSTOCK_SELECTION: {
                "title": "Feedstock Selection Validation Review",
                "description": "Review of feedstock ranking based on availability, technical compatibility, and cost efficiency",
                "key_outputs": [
                    "Ranked feedstock list with numerical scores",
                    "Availability consistency analysis",
                    "Technical compatibility assessment",
                    "Cost efficiency calculations"
                ],
                "review_focus": [
                    "Regional economic and technical realities",
                    "Data quality and sources",
                    "Methodology appropriateness",
                    "Practical implementability"
                ]
            },
            ValidationComponent.LOCATION_FACTORS: {
                "title": "Location Factor Adjustment Review",
                "description": "Review of enhanced location factors with separate capital and operational components",
                "key_outputs": [
                    "Separate capital and operational cost factors",
                    "Infrastructure quality indices",
                    "Labor market assessments",
                    "Regulatory complexity analysis"
                ],
                "review_focus": [
                    "Cost component separation accuracy",
                    "Infrastructure data quality",
                    "Weighting methodology",
                    "Regional relevance"
                ]
            },
            ValidationComponent.RISK_ASSESSMENT: {
                "title": "Quantitative Risk Assessment Review",
                "description": "Review of comprehensive risk assessment with numerical thresholds",
                "key_outputs": [
                    "Composite risk scores",
                    "Political stability indices",
                    "Infrastructure quality scores",
                    "Market volatility assessments"
                ],
                "review_focus": [
                    "Threshold actionability",
                    "Data source credibility",
                    "Methodology robustness",
                    "Risk category completeness"
                ]
            },
            ValidationComponent.ECONOMIC_BENCHMARKING: {
                "title": "Economic Benchmarking Review",
                "description": "Review of cost benchmarking against industry standards",
                "key_outputs": [
                    "Percentage deviation from industry standards",
                    "Confidence intervals",
                    "Benchmark source analysis",
                    "Regional adjustment factors"
                ],
                "review_focus": [
                    "Benchmarking source robustness",
                    "Calculation methodology",
                    "Regional adjustments",
                    "Actionable insights"
                ]
            },
            ValidationComponent.OPERATIONAL_HOURS: {
                "title": "Operational Hours Validation Review",
                "description": "Review of evidence-based operational hours validation",
                "key_outputs": [
                    "Validated operational hour ranges",
                    "Historical data analysis",
                    "Maintenance schedule impact",
                    "Seasonal variance assessment"
                ],
                "review_focus": [
                    "Historical data adequacy",
                    "Methodology appropriateness",
                    "Regional realism",
                    "Evidence-based justification"
                ]
            }
        }
    
    def generate_review_template(self, 
                               component: ValidationComponent,
                               plant_configuration: Dict[str, Any],
                               validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a structured review template for a component"""
        template = self.review_templates.get(component, {})
        component_questions = [q for q in self.review_questions if q.component == component]
        
        review_template = {
            "component": component.value,
            "template_info": template,
            "plant_configuration": plant_configuration,
            "validation_results": validation_results,
            "review_questions": [],
            "review_instructions": self._generate_review_instructions(component),
            "evaluation_criteria": self._generate_evaluation_criteria(component),
            "template_timestamp": datetime.now().isoformat()
        }
        
        # Add structured questions
        for question in component_questions:
            question_data = {
                "question_id": question.question_id,
                "criteria": question.criteria.value,
                "question_text": question.question_text,
                "required": question.required,
                "response_type": question.response_type,
                "rating_scale": question.rating_scale,
                "choices": question.choices,
                "guidance": question.guidance
            }
            review_template["review_questions"].append(question_data)
        
        return review_template
    
    def _generate_review_instructions(self, component: ValidationComponent) -> List[str]:
        """Generate specific review instructions for each component"""
        instructions = {
            ValidationComponent.FEEDSTOCK_SELECTION: [
                "Review the feedstock ranking methodology and scoring criteria",
                "Assess the quality and reliability of availability data sources",
                "Evaluate technical compatibility scoring for regional conditions",
                "Verify cost efficiency calculations and regional adjustments",
                "Consider practical implementation challenges and mitigation strategies"
            ],
            ValidationComponent.LOCATION_FACTORS: [
                "Evaluate the separation of capital and operational cost components",
                "Review infrastructure quality scoring methodology and data sources",
                "Assess labor market analysis and regional wage adjustments",
                "Verify regulatory complexity scoring and compliance requirements",
                "Consider practical implications of location factor recommendations"
            ],
            ValidationComponent.RISK_ASSESSMENT: [
                "Review the quantitative risk thresholds and action points",
                "Assess political stability index components and data sources",
                "Evaluate infrastructure quality scoring methodology",
                "Verify market volatility calculations and historical analysis",
                "Consider risk mitigation strategies and their effectiveness"
            ],
            ValidationComponent.ECONOMIC_BENCHMARKING: [
                "Review benchmarking data sources and their credibility",
                "Assess calculation methodology for percentage deviations",
                "Evaluate regional adjustment factors and scaling methods",
                "Verify confidence intervals and statistical approaches",
                "Consider practical utility of benchmarking insights"
            ],
            ValidationComponent.OPERATIONAL_HOURS: [
                "Review historical data quality and representativeness",
                "Assess maintenance schedule analysis and downtime calculations",
                "Evaluate seasonal variance factors and their impact",
                "Verify evidence-based justifications for hour recommendations",
                "Consider practical feasibility of operational hour targets"
            ]
        }
        
        return instructions.get(component, [])
    
    def _generate_evaluation_criteria(self, component: ValidationComponent) -> Dict[str, str]:
        """Generate evaluation criteria descriptions for each component"""
        criteria = {
            ValidationComponent.FEEDSTOCK_SELECTION: {
                "technical_accuracy": "Accuracy of technical compatibility assessments and conversion efficiency calculations",
                "data_quality": "Quality, reliability, and currency of feedstock availability and cost data",
                "methodology": "Appropriateness of ranking methodology and scoring algorithms",
                "regional_relevance": "Relevance and accuracy of regional market conditions and adjustments",
                "practical_applicability": "Feasibility of implementing feedstock recommendations in target region"
            },
            ValidationComponent.LOCATION_FACTORS: {
                "technical_accuracy": "Accuracy of cost component separation and factor calculations",
                "data_quality": "Quality and reliability of infrastructure, labor, and regulatory data",
                "methodology": "Appropriateness of weighting schemes and calculation methods",
                "regional_relevance": "Accuracy of regional differences and local market conditions"
            },
            ValidationComponent.RISK_ASSESSMENT: {
                "technical_accuracy": "Accuracy of risk quantification and threshold setting",
                "data_quality": "Quality of political, economic, and operational risk data",
                "methodology": "Robustness of composite risk scoring methodology",
                "completeness": "Completeness of risk category coverage and factor inclusion"
            },
            ValidationComponent.ECONOMIC_BENCHMARKING: {
                "data_quality": "Quality and credibility of industry benchmark sources",
                "methodology": "Appropriateness of deviation calculations and statistical methods",
                "regional_relevance": "Accuracy of regional adjustments and market conditions",
                "practical_applicability": "Utility of benchmarking insights for cost optimization"
            },
            ValidationComponent.OPERATIONAL_HOURS: {
                "data_quality": "Quality and representativeness of historical operational data",
                "methodology": "Appropriateness of maintenance and seasonal variance calculations",
                "regional_relevance": "Realism of operational hour recommendations for target region",
                "practical_applicability": "Quality of evidence-based justifications and implementation guidance"
            }
        }
        
        return criteria.get(component, {})
    
    def conduct_component_review(self, 
                               component: ValidationComponent,
                               expert_id: str,
                               plant_configuration: Dict[str, Any],
                               validation_results: Dict[str, Any],
                               review_responses: List[Dict[str, Any]]) -> ComponentReview:
        """Conduct a complete component review"""
        # Process review responses
        processed_responses = []
        for response_data in review_responses:
            response = ReviewResponse(
                question_id=response_data["question_id"],
                expert_id=expert_id,
                response_value=response_data["response_value"],
                confidence_level=response_data.get("confidence_level", 0.8),
                comments=response_data.get("comments", ""),
                evidence_sources=response_data.get("evidence_sources", [])
            )
            processed_responses.append(response)
        
        # Calculate overall rating
        rating_responses = [r for r in processed_responses if isinstance(r.response_value, (int, float))]
        if rating_responses:
            overall_rating = sum(r.response_value for r in rating_responses) / len(rating_responses)
        else:
            overall_rating = 5.0
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence_level for r in processed_responses) / len(processed_responses)
        
        # Generate findings and recommendations
        key_findings = self._generate_key_findings(component, processed_responses, validation_results)
        recommendations = self._generate_component_recommendations(component, processed_responses, overall_rating)
        
        # Determine approval status
        approval_status = self._determine_approval_status(overall_rating, overall_confidence)
        
        return ComponentReview(
            component=component,
            expert_id=expert_id,
            plant_configuration=plant_configuration,
            validation_results=validation_results,
            review_responses=processed_responses,
            overall_rating=overall_rating,
            overall_confidence=overall_confidence,
            key_findings=key_findings,
            recommendations=recommendations,
            approval_status=approval_status
        )
    
    def conduct_comprehensive_review(self, 
                                   plant_name: str,
                                   location: str,
                                   capacity: float,
                                   expert_id: str,
                                   component_reviews: List[ComponentReview]) -> ComprehensiveReview:
        """Conduct a comprehensive review of all components"""
        # Calculate weighted overall confidence
        weighted_confidence = 0
        total_weight = 0
        
        for review in component_reviews:
            weight = self.component_weights.get(review.component, 0.2)
            weighted_confidence += review.overall_confidence * weight
            total_weight += weight
        
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # Generate overall assessment
        overall_assessment = self._generate_overall_assessment(component_reviews, overall_confidence)
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues(component_reviews)
        
        # Generate comprehensive recommendations
        recommendations = self._generate_comprehensive_recommendations(component_reviews, critical_issues)
        
        # Determine overall approval status
        approval_status = self._determine_comprehensive_approval_status(component_reviews, critical_issues)
        
        return ComprehensiveReview(
            plant_name=plant_name,
            location=location,
            capacity=capacity,
            expert_id=expert_id,
            component_reviews=component_reviews,
            overall_assessment=overall_assessment,
            overall_confidence=overall_confidence,
            critical_issues=critical_issues,
            recommendations=recommendations,
            approval_status=approval_status
        )
    
    def _generate_key_findings(self, 
                             component: ValidationComponent,
                             responses: List[ReviewResponse],
                             validation_results: Dict[str, Any]) -> List[str]:
        """Generate key findings from component review"""
        findings = []
        
        # Analyze ratings
        ratings = [r.response_value for r in responses if isinstance(r.response_value, (int, float))]
        if ratings:
            avg_rating = sum(ratings) / len(ratings)
            if avg_rating >= 8:
                findings.append("High quality validation with strong methodology and data")
            elif avg_rating >= 6:
                findings.append("Good quality validation with minor areas for improvement")
            else:
                findings.append("Validation requires significant improvements")
        
        # Analyze comments for key themes
        comments = [r.comments for r in responses if r.comments]
        if comments:
            # Simple keyword analysis
            common_themes = []
            if any("data quality" in comment.lower() for comment in comments):
                common_themes.append("Data quality concerns identified")
            if any("methodology" in comment.lower() for comment in comments):
                common_themes.append("Methodology refinements needed")
            if any("regional" in comment.lower() for comment in comments):
                common_themes.append("Regional relevance considerations")
            
            findings.extend(common_themes)
        
        return findings
    
    def _generate_component_recommendations(self, 
                                          component: ValidationComponent,
                                          responses: List[ReviewResponse],
                                          overall_rating: float) -> List[str]:
        """Generate component-specific recommendations"""
        recommendations = []
        
        # Rating-based recommendations
        if overall_rating < 5:
            recommendations.append("Significant improvements required before implementation")
        elif overall_rating < 7:
            recommendations.append("Address identified issues before final approval")
        else:
            recommendations.append("Good quality validation - minor refinements suggested")
        
        # Component-specific recommendations
        component_recs = {
            ValidationComponent.FEEDSTOCK_SELECTION: [
                "Validate feedstock availability with local suppliers",
                "Update cost data regularly to reflect market changes",
                "Consider seasonal variations in feedstock quality"
            ],
            ValidationComponent.LOCATION_FACTORS: [
                "Verify infrastructure data with local assessments",
                "Update labor market data regularly",
                "Consider regulatory changes and updates"
            ],
            ValidationComponent.RISK_ASSESSMENT: [
                "Monitor political and economic indicators regularly",
                "Update risk assessments based on current events",
                "Develop contingency plans for high-risk scenarios"
            ],
            ValidationComponent.ECONOMIC_BENCHMARKING: [
                "Update benchmark data with latest industry reports",
                "Verify regional cost adjustments with local data",
                "Consider multiple benchmarking sources"
            ],
            ValidationComponent.OPERATIONAL_HOURS: [
                "Validate assumptions with plant operators",
                "Monitor actual performance against predictions",
                "Update maintenance schedules based on experience"
            ]
        }
        
        recommendations.extend(component_recs.get(component, []))
        
        return recommendations
    
    def _determine_approval_status(self, 
                                 overall_rating: float,
                                 overall_confidence: float) -> ValidationStatus:
        """Determine approval status based on rating and confidence"""
        if overall_rating >= 8 and overall_confidence >= 0.8:
            return ValidationStatus.APPROVED
        elif overall_rating >= 6 and overall_confidence >= 0.6:
            return ValidationStatus.NEEDS_REVISION
        else:
            return ValidationStatus.REJECTED
    
    def _generate_overall_assessment(self, 
                                   component_reviews: List[ComponentReview],
                                   overall_confidence: float) -> str:
        """Generate overall assessment text"""
        approved_count = sum(1 for r in component_reviews if r.approval_status == ValidationStatus.APPROVED)
        total_count = len(component_reviews)
        
        if approved_count == total_count:
            return "All validation components meet quality standards with high confidence"
        elif approved_count >= total_count * 0.8:
            return "Most validation components are satisfactory with minor improvements needed"
        elif approved_count >= total_count * 0.6:
            return "Several validation components require improvements before approval"
        else:
            return "Significant improvements required across multiple validation components"
    
    def _identify_critical_issues(self, component_reviews: List[ComponentReview]) -> List[str]:
        """Identify critical issues across component reviews"""
        critical_issues = []
        
        for review in component_reviews:
            if review.approval_status == ValidationStatus.REJECTED:
                critical_issues.append(f"{review.component.value}: Validation rejected - significant issues identified")
            elif review.overall_rating < 5:
                critical_issues.append(f"{review.component.value}: Low quality rating - requires major improvements")
        
        return critical_issues
    
    def _generate_comprehensive_recommendations(self, 
                                              component_reviews: List[ComponentReview],
                                              critical_issues: List[str]) -> List[str]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Overall recommendations
        if critical_issues:
            recommendations.append("Address critical issues before proceeding with plant development")
        
        # Component-specific recommendations
        for review in component_reviews:
            if review.approval_status != ValidationStatus.APPROVED:
                recommendations.extend(review.recommendations)
        
        # General recommendations
        recommendations.extend([
            "Regular monitoring and updates of validation components",
            "Continuous improvement based on operational experience",
            "Stakeholder engagement throughout implementation"
        ])
        
        return recommendations
    
    def _determine_comprehensive_approval_status(self, 
                                               component_reviews: List[ComponentReview],
                                               critical_issues: List[str]) -> ValidationStatus:
        """Determine comprehensive approval status"""
        if critical_issues:
            return ValidationStatus.REJECTED
        
        approved_count = sum(1 for r in component_reviews if r.approval_status == ValidationStatus.APPROVED)
        total_count = len(component_reviews)
        
        if approved_count == total_count:
            return ValidationStatus.APPROVED
        elif approved_count >= total_count * 0.8:
            return ValidationStatus.NEEDS_REVISION
        else:
            return ValidationStatus.REJECTED
    
    def generate_review_report(self, 
                             review: ComprehensiveReview,
                             include_component_details: bool = True) -> str:
        """Generate comprehensive review report"""
        report = []
        report.append("="*80)
        report.append("COMPREHENSIVE EXPERT REVIEW REPORT")
        report.append("="*80)
        report.append(f"Plant: {review.plant_name}")
        report.append(f"Location: {review.location}")
        report.append(f"Capacity: {review.capacity:,.0f} tons/year")
        report.append(f"Expert: {review.expert_id}")
        report.append(f"Review Date: {review.review_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Assessment: {review.overall_assessment}")
        report.append(f"Overall Confidence: {review.overall_confidence:.1%}")
        report.append(f"Approval Status: {review.approval_status.value}")
        report.append("")
        
        # Component Summary
        report.append("COMPONENT REVIEW SUMMARY")
        report.append("-" * 40)
        for component_review in review.component_reviews:
            report.append(f"{component_review.component.value.replace('_', ' ').title()}:")
            report.append(f"  Rating: {component_review.overall_rating:.1f}/10")
            report.append(f"  Confidence: {component_review.overall_confidence:.1%}")
            report.append(f"  Status: {component_review.approval_status.value}")
            report.append("")
        
        # Critical Issues
        if review.critical_issues:
            report.append("CRITICAL ISSUES")
            report.append("-" * 40)
            for issue in review.critical_issues:
                report.append(f"• {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        for rec in review.recommendations:
            report.append(f"• {rec}")
        report.append("")
        
        if include_component_details:
            # Component Details
            report.append("DETAILED COMPONENT REVIEWS")
            report.append("-" * 40)
            
            for component_review in review.component_reviews:
                report.append(f"\n{component_review.component.value.replace('_', ' ').title().upper()}")
                report.append("-" * 40)
                
                # Key Findings
                report.append("Key Findings:")
                for finding in component_review.key_findings:
                    report.append(f"• {finding}")
                report.append("")
                
                # Recommendations
                report.append("Recommendations:")
                for rec in component_review.recommendations:
                    report.append(f"• {rec}")
                report.append("")
                
                # Question Responses
                report.append("Review Responses:")
                for response in component_review.review_responses:
                    if isinstance(response.response_value, (int, float)):
                        report.append(f"• Q{response.question_id}: {response.response_value}/10")
                    else:
                        report.append(f"• Q{response.question_id}: {response.response_value}")
                    if response.comments:
                        report.append(f"  Comments: {response.comments}")
                report.append("")
        
        return "\n".join(report)
    
    def save_review_results(self, 
                          review: ComprehensiveReview,
                          filename: Optional[str] = None) -> Path:
        """Save review results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"expert_review_{review.plant_name.replace(' ', '_')}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            "plant_name": review.plant_name,
            "location": review.location,
            "capacity": review.capacity,
            "expert_id": review.expert_id,
            "overall_assessment": review.overall_assessment,
            "overall_confidence": review.overall_confidence,
            "approval_status": review.approval_status.value,
            "critical_issues": review.critical_issues,
            "recommendations": review.recommendations,
            "review_timestamp": review.review_timestamp.isoformat(),
            "component_reviews": []
        }
        
        for component_review in review.component_reviews:
            component_data = {
                "component": component_review.component.value,
                "overall_rating": component_review.overall_rating,
                "overall_confidence": component_review.overall_confidence,
                "approval_status": component_review.approval_status.value,
                "key_findings": component_review.key_findings,
                "recommendations": component_review.recommendations,
                "review_responses": []
            }
            
            for response in component_review.review_responses:
                response_data = {
                    "question_id": response.question_id,
                    "response_value": response.response_value,
                    "confidence_level": response.confidence_level,
                    "comments": response.comments,
                    "evidence_sources": response.evidence_sources
                }
                component_data["review_responses"].append(response_data)
            
            data["component_reviews"].append(component_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Expert review results saved to {filepath}")
        return filepath 