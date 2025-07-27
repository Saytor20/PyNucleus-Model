"""
Enhanced Location Factor Adjustment Module

This module provides advanced location-based cost adjustments that distinguish between
capital and operational cost components with detailed factor analysis:

- Infrastructure quality index (capital cost impact)
- Labor market index (operational cost impact)
- Regulatory complexity index (both capital & operational impact)

Output: Separate adjustment factors for capital and operating expenses
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..utils.logger import logger


class InfrastructureCategory(Enum):
    """Infrastructure categories for assessment"""
    TRANSPORTATION = "transportation"
    UTILITIES = "utilities"
    TELECOMMUNICATIONS = "telecommunications"
    INDUSTRIAL_PARKS = "industrial_parks"
    PORTS_LOGISTICS = "ports_logistics"
    FINANCIAL_SERVICES = "financial_services"


class LaborCategory(Enum):
    """Labor market categories"""
    SKILLED_TECHNICAL = "skilled_technical"
    MANAGEMENT = "management"
    OPERATORS = "operators"
    MAINTENANCE = "maintenance"
    ADMINISTRATIVE = "administrative"
    CONSTRUCTION = "construction"


class RegulatoryCategory(Enum):
    """Regulatory complexity categories"""
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"
    ZONING = "zoning"
    PERMITS = "permits"
    TAXATION = "taxation"
    LABOR_LAWS = "labor_laws"
    IMPORT_EXPORT = "import_export"


@dataclass
class InfrastructureIndex:
    """Infrastructure quality assessment for a location"""
    location: str
    transportation_score: float  # 0-10
    utilities_score: float  # 0-10
    telecommunications_score: float  # 0-10
    industrial_parks_score: float  # 0-10
    ports_logistics_score: float  # 0-10
    financial_services_score: float  # 0-10
    composite_score: float = field(init=False)
    
    def __post_init__(self):
        """Calculate composite infrastructure score"""
        weights = {
            'transportation': 0.25,
            'utilities': 0.25,
            'telecommunications': 0.15,
            'industrial_parks': 0.15,
            'ports_logistics': 0.10,
            'financial_services': 0.10
        }
        
        self.composite_score = (
            self.transportation_score * weights['transportation'] +
            self.utilities_score * weights['utilities'] +
            self.telecommunications_score * weights['telecommunications'] +
            self.industrial_parks_score * weights['industrial_parks'] +
            self.ports_logistics_score * weights['ports_logistics'] +
            self.financial_services_score * weights['financial_services']
        )


@dataclass
class LaborMarketIndex:
    """Labor market assessment for a location"""
    location: str
    skilled_technical_availability: float  # 0-10
    skilled_technical_cost: float  # Relative to baseline
    management_availability: float  # 0-10
    management_cost: float  # Relative to baseline
    operators_availability: float  # 0-10
    operators_cost: float  # Relative to baseline
    maintenance_availability: float  # 0-10
    maintenance_cost: float  # Relative to baseline
    administrative_availability: float  # 0-10
    administrative_cost: float  # Relative to baseline
    construction_availability: float  # 0-10
    construction_cost: float  # Relative to baseline
    productivity_index: float  # 0-10
    labor_flexibility: float  # 0-10
    composite_availability: float = field(init=False)
    composite_cost: float = field(init=False)
    
    def __post_init__(self):
        """Calculate composite labor market scores"""
        # Availability weights
        availability_weights = {
            'skilled_technical': 0.25,
            'management': 0.15,
            'operators': 0.20,
            'maintenance': 0.15,
            'administrative': 0.10,
            'construction': 0.15
        }
        
        # Cost weights
        cost_weights = {
            'skilled_technical': 0.30,
            'management': 0.20,
            'operators': 0.20,
            'maintenance': 0.15,
            'administrative': 0.10,
            'construction': 0.05
        }
        
        self.composite_availability = (
            self.skilled_technical_availability * availability_weights['skilled_technical'] +
            self.management_availability * availability_weights['management'] +
            self.operators_availability * availability_weights['operators'] +
            self.maintenance_availability * availability_weights['maintenance'] +
            self.administrative_availability * availability_weights['administrative'] +
            self.construction_availability * availability_weights['construction']
        )
        
        self.composite_cost = (
            self.skilled_technical_cost * cost_weights['skilled_technical'] +
            self.management_cost * cost_weights['management'] +
            self.operators_cost * cost_weights['operators'] +
            self.maintenance_cost * cost_weights['maintenance'] +
            self.administrative_cost * cost_weights['administrative'] +
            self.construction_cost * cost_weights['construction']
        )


@dataclass
class RegulatoryComplexityIndex:
    """Regulatory complexity assessment for a location"""
    location: str
    environmental_complexity: float  # 0-10
    safety_complexity: float  # 0-10
    zoning_complexity: float  # 0-10
    permits_complexity: float  # 0-10
    taxation_complexity: float  # 0-10
    labor_laws_complexity: float  # 0-10
    import_export_complexity: float  # 0-10
    approval_timeframes: float  # Months for typical project
    compliance_costs: float  # Relative to baseline
    political_stability: float  # 0-10
    composite_complexity: float = field(init=False)
    
    def __post_init__(self):
        """Calculate composite regulatory complexity score"""
        weights = {
            'environmental': 0.20,
            'safety': 0.15,
            'zoning': 0.10,
            'permits': 0.15,
            'taxation': 0.15,
            'labor_laws': 0.10,
            'import_export': 0.15
        }
        
        self.composite_complexity = (
            self.environmental_complexity * weights['environmental'] +
            self.safety_complexity * weights['safety'] +
            self.zoning_complexity * weights['zoning'] +
            self.permits_complexity * weights['permits'] +
            self.taxation_complexity * weights['taxation'] +
            self.labor_laws_complexity * weights['labor_laws'] +
            self.import_export_complexity * weights['import_export']
        )


@dataclass
class LocationFactorAnalysis:
    """Comprehensive location factor analysis result"""
    location: str
    infrastructure_index: InfrastructureIndex
    labor_market_index: LaborMarketIndex
    regulatory_index: RegulatoryComplexityIndex
    capital_cost_factor: float
    operational_cost_factor: float
    total_risk_score: float
    key_advantages: List[str]
    key_challenges: List[str]
    mitigation_strategies: List[str]
    analysis_timestamp: datetime = field(default_factory=datetime.now)


class LocationFactorAnalyzer:
    """
    Enhanced location factor analyzer for capital and operational cost separation
    
    Provides detailed analysis of location-specific factors that impact plant
    costs, with separate treatment of capital and operational components.
    """
    
    def __init__(self, data_dir: str = "data/location_factors"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize location databases
        self.infrastructure_data = self._initialize_infrastructure_data()
        self.labor_market_data = self._initialize_labor_market_data()
        self.regulatory_data = self._initialize_regulatory_data()
        
        # Cost factor models
        self.capital_cost_model = self._initialize_capital_cost_model()
        self.operational_cost_model = self._initialize_operational_cost_model()
        
        logger.info("LocationFactorAnalyzer initialized with comprehensive databases")
    
    def _initialize_infrastructure_data(self) -> Dict[str, InfrastructureIndex]:
        """Initialize infrastructure quality data for African locations"""
        return {
            "South Africa": InfrastructureIndex(
                location="South Africa",
                transportation_score=7.5,
                utilities_score=7.0,
                telecommunications_score=8.0,
                industrial_parks_score=8.5,
                ports_logistics_score=8.0,
                financial_services_score=9.0
            ),
            "Nigeria": InfrastructureIndex(
                location="Nigeria",
                transportation_score=5.0,
                utilities_score=4.5,
                telecommunications_score=6.5,
                industrial_parks_score=5.5,
                ports_logistics_score=6.0,
                financial_services_score=6.5
            ),
            "Kenya": InfrastructureIndex(
                location="Kenya",
                transportation_score=6.0,
                utilities_score=5.5,
                telecommunications_score=7.5,
                industrial_parks_score=6.5,
                ports_logistics_score=7.0,
                financial_services_score=7.0
            ),
            "Ghana": InfrastructureIndex(
                location="Ghana",
                transportation_score=6.5,
                utilities_score=6.0,
                telecommunications_score=7.0,
                industrial_parks_score=6.0,
                ports_logistics_score=6.5,
                financial_services_score=6.5
            ),
            "Morocco": InfrastructureIndex(
                location="Morocco",
                transportation_score=7.0,
                utilities_score=6.5,
                telecommunications_score=7.5,
                industrial_parks_score=7.0,
                ports_logistics_score=7.5,
                financial_services_score=7.0
            ),
            "Egypt": InfrastructureIndex(
                location="Egypt",
                transportation_score=6.0,
                utilities_score=5.5,
                telecommunications_score=7.0,
                industrial_parks_score=6.5,
                ports_logistics_score=8.0,
                financial_services_score=6.5
            ),
            "Tanzania": InfrastructureIndex(
                location="Tanzania",
                transportation_score=4.5,
                utilities_score=4.0,
                telecommunications_score=6.0,
                industrial_parks_score=5.0,
                ports_logistics_score=6.0,
                financial_services_score=5.5
            ),
            "Ethiopia": InfrastructureIndex(
                location="Ethiopia",
                transportation_score=4.0,
                utilities_score=3.5,
                telecommunications_score=5.5,
                industrial_parks_score=4.5,
                ports_logistics_score=4.0,
                financial_services_score=4.5
            )
        }
    
    def _initialize_labor_market_data(self) -> Dict[str, LaborMarketIndex]:
        """Initialize labor market data for African locations"""
        return {
            "South Africa": LaborMarketIndex(
                location="South Africa",
                skilled_technical_availability=8.0,
                skilled_technical_cost=1.0,  # Baseline
                management_availability=8.5,
                management_cost=1.0,
                operators_availability=7.5,
                operators_cost=1.0,
                maintenance_availability=7.0,
                maintenance_cost=1.0,
                administrative_availability=8.0,
                administrative_cost=1.0,
                construction_availability=7.5,
                construction_cost=1.0,
                productivity_index=7.5,
                labor_flexibility=6.0
            ),
            "Nigeria": LaborMarketIndex(
                location="Nigeria",
                skilled_technical_availability=6.5,
                skilled_technical_cost=0.7,
                management_availability=7.0,
                management_cost=0.8,
                operators_availability=7.5,
                operators_cost=0.6,
                maintenance_availability=6.0,
                maintenance_cost=0.6,
                administrative_availability=7.0,
                administrative_cost=0.5,
                construction_availability=8.0,
                construction_cost=0.5,
                productivity_index=6.5,
                labor_flexibility=7.5
            ),
            "Kenya": LaborMarketIndex(
                location="Kenya",
                skilled_technical_availability=7.0,
                skilled_technical_cost=0.6,
                management_availability=7.5,
                management_cost=0.7,
                operators_availability=7.0,
                operators_cost=0.5,
                maintenance_availability=6.5,
                maintenance_cost=0.5,
                administrative_availability=7.5,
                administrative_cost=0.4,
                construction_availability=7.5,
                construction_cost=0.4,
                productivity_index=7.0,
                labor_flexibility=8.0
            ),
            "Ghana": LaborMarketIndex(
                location="Ghana",
                skilled_technical_availability=6.5,
                skilled_technical_cost=0.65,
                management_availability=7.0,
                management_cost=0.75,
                operators_availability=7.5,
                operators_cost=0.55,
                maintenance_availability=6.5,
                maintenance_cost=0.55,
                administrative_availability=7.0,
                administrative_cost=0.45,
                construction_availability=7.5,
                construction_cost=0.45,
                productivity_index=6.8,
                labor_flexibility=7.5
            ),
            "Morocco": LaborMarketIndex(
                location="Morocco",
                skilled_technical_availability=7.5,
                skilled_technical_cost=0.8,
                management_availability=7.5,
                management_cost=0.85,
                operators_availability=7.0,
                operators_cost=0.7,
                maintenance_availability=7.0,
                maintenance_cost=0.7,
                administrative_availability=7.5,
                administrative_cost=0.6,
                construction_availability=7.0,
                construction_cost=0.6,
                productivity_index=7.2,
                labor_flexibility=6.5
            ),
            "Egypt": LaborMarketIndex(
                location="Egypt",
                skilled_technical_availability=7.0,
                skilled_technical_cost=0.7,
                management_availability=7.5,
                management_cost=0.8,
                operators_availability=7.5,
                operators_cost=0.6,
                maintenance_availability=6.5,
                maintenance_cost=0.6,
                administrative_availability=7.0,
                administrative_cost=0.5,
                construction_availability=8.0,
                construction_cost=0.5,
                productivity_index=6.8,
                labor_flexibility=7.0
            ),
            "Tanzania": LaborMarketIndex(
                location="Tanzania",
                skilled_technical_availability=5.5,
                skilled_technical_cost=0.5,
                management_availability=6.0,
                management_cost=0.6,
                operators_availability=6.5,
                operators_cost=0.4,
                maintenance_availability=5.5,
                maintenance_cost=0.4,
                administrative_availability=6.0,
                administrative_cost=0.3,
                construction_availability=7.0,
                construction_cost=0.3,
                productivity_index=5.8,
                labor_flexibility=8.5
            ),
            "Ethiopia": LaborMarketIndex(
                location="Ethiopia",
                skilled_technical_availability=5.0,
                skilled_technical_cost=0.4,
                management_availability=5.5,
                management_cost=0.5,
                operators_availability=6.0,
                operators_cost=0.3,
                maintenance_availability=5.0,
                maintenance_cost=0.3,
                administrative_availability=5.5,
                administrative_cost=0.25,
                construction_availability=6.5,
                construction_cost=0.25,
                productivity_index=5.0,
                labor_flexibility=9.0
            )
        }
    
    def _initialize_regulatory_data(self) -> Dict[str, RegulatoryComplexityIndex]:
        """Initialize regulatory complexity data for African locations"""
        return {
            "South Africa": RegulatoryComplexityIndex(
                location="South Africa",
                environmental_complexity=7.0,
                safety_complexity=8.0,
                zoning_complexity=6.5,
                permits_complexity=7.5,
                taxation_complexity=8.5,
                labor_laws_complexity=8.0,
                import_export_complexity=6.0,
                approval_timeframes=18.0,
                compliance_costs=1.0,
                political_stability=7.5
            ),
            "Nigeria": RegulatoryComplexityIndex(
                location="Nigeria",
                environmental_complexity=5.5,
                safety_complexity=6.0,
                zoning_complexity=7.0,
                permits_complexity=8.0,
                taxation_complexity=7.5,
                labor_laws_complexity=6.5,
                import_export_complexity=7.5,
                approval_timeframes=24.0,
                compliance_costs=1.2,
                political_stability=5.5
            ),
            "Kenya": RegulatoryComplexityIndex(
                location="Kenya",
                environmental_complexity=6.0,
                safety_complexity=6.5,
                zoning_complexity=6.0,
                permits_complexity=7.0,
                taxation_complexity=7.0,
                labor_laws_complexity=6.0,
                import_export_complexity=6.5,
                approval_timeframes=20.0,
                compliance_costs=1.1,
                political_stability=6.5
            ),
            "Ghana": RegulatoryComplexityIndex(
                location="Ghana",
                environmental_complexity=6.0,
                safety_complexity=6.0,
                zoning_complexity=5.5,
                permits_complexity=6.5,
                taxation_complexity=6.5,
                labor_laws_complexity=5.5,
                import_export_complexity=6.0,
                approval_timeframes=16.0,
                compliance_costs=1.0,
                political_stability=7.0
            ),
            "Morocco": RegulatoryComplexityIndex(
                location="Morocco",
                environmental_complexity=7.5,
                safety_complexity=7.0,
                zoning_complexity=6.5,
                permits_complexity=7.0,
                taxation_complexity=7.5,
                labor_laws_complexity=6.5,
                import_export_complexity=5.5,
                approval_timeframes=15.0,
                compliance_costs=1.05,
                political_stability=7.5
            ),
            "Egypt": RegulatoryComplexityIndex(
                location="Egypt",
                environmental_complexity=6.5,
                safety_complexity=6.0,
                zoning_complexity=7.0,
                permits_complexity=7.5,
                taxation_complexity=7.0,
                labor_laws_complexity=6.0,
                import_export_complexity=6.5,
                approval_timeframes=22.0,
                compliance_costs=1.15,
                political_stability=6.0
            ),
            "Tanzania": RegulatoryComplexityIndex(
                location="Tanzania",
                environmental_complexity=5.0,
                safety_complexity=5.5,
                zoning_complexity=5.0,
                permits_complexity=6.0,
                taxation_complexity=6.0,
                labor_laws_complexity=5.0,
                import_export_complexity=6.5,
                approval_timeframes=20.0,
                compliance_costs=1.1,
                political_stability=6.5
            ),
            "Ethiopia": RegulatoryComplexityIndex(
                location="Ethiopia",
                environmental_complexity=4.5,
                safety_complexity=5.0,
                zoning_complexity=4.5,
                permits_complexity=5.5,
                taxation_complexity=5.5,
                labor_laws_complexity=4.5,
                import_export_complexity=7.0,
                approval_timeframes=18.0,
                compliance_costs=1.0,
                political_stability=5.5
            )
        }
    
    def _initialize_capital_cost_model(self) -> Dict[str, Any]:
        """Initialize capital cost factor model"""
        return {
            "infrastructure_weight": 0.40,
            "regulatory_weight": 0.35,
            "labor_construction_weight": 0.25,
            "base_factors": {
                "infrastructure_excellent": 0.95,  # 5% reduction for excellent infrastructure
                "infrastructure_poor": 1.40,      # 40% increase for poor infrastructure
                "regulatory_simple": 0.90,        # 10% reduction for simple regulations
                "regulatory_complex": 1.50,       # 50% increase for complex regulations
                "labor_cheap": 0.85,              # 15% reduction for cheap construction labor
                "labor_expensive": 1.25           # 25% increase for expensive labor
            },
            "infrastructure_thresholds": {
                "excellent": 8.0,
                "good": 6.5,
                "moderate": 5.0,
                "poor": 3.5
            },
            "regulatory_thresholds": {
                "simple": 4.0,
                "moderate": 6.0,
                "complex": 8.0
            }
        }
    
    def _initialize_operational_cost_model(self) -> Dict[str, Any]:
        """Initialize operational cost factor model"""
        return {
            "labor_cost_weight": 0.50,
            "utilities_weight": 0.25,
            "regulatory_compliance_weight": 0.15,
            "logistics_weight": 0.10,
            "base_factors": {
                "labor_very_cheap": 0.60,
                "labor_cheap": 0.80,
                "labor_moderate": 1.00,
                "labor_expensive": 1.20,
                "utilities_excellent": 0.90,
                "utilities_poor": 1.30,
                "compliance_simple": 0.95,
                "compliance_complex": 1.15,
                "logistics_excellent": 0.95,
                "logistics_poor": 1.25
            },
            "labor_cost_thresholds": {
                "very_cheap": 0.5,
                "cheap": 0.7,
                "moderate": 0.9,
                "expensive": 1.1
            },
            "utilities_thresholds": {
                "excellent": 7.0,
                "good": 5.5,
                "moderate": 4.0,
                "poor": 2.5
            }
        }
    
    def analyze_location_factors(self, 
                               location: str,
                               plant_type: str = "chemical_processing",
                               capacity_scale: str = "medium") -> LocationFactorAnalysis:
        """
        Analyze comprehensive location factors for a specific location
        
        Args:
            location: Location name
            plant_type: Type of plant (affects factor weights)
            capacity_scale: Scale of plant (small, medium, large)
            
        Returns:
            LocationFactorAnalysis object with detailed analysis
        """
        # Get base data
        infrastructure = self.infrastructure_data.get(location)
        labor_market = self.labor_market_data.get(location)
        regulatory = self.regulatory_data.get(location)
        
        if not all([infrastructure, labor_market, regulatory]):
            logger.error(f"Incomplete data for location: {location}")
            return self._create_default_analysis(location)
        
        # Calculate capital cost factor
        capital_factor = self._calculate_capital_cost_factor(
            infrastructure, labor_market, regulatory, plant_type, capacity_scale
        )
        
        # Calculate operational cost factor
        operational_factor = self._calculate_operational_cost_factor(
            infrastructure, labor_market, regulatory, plant_type, capacity_scale
        )
        
        # Calculate total risk score
        risk_score = self._calculate_total_risk_score(
            infrastructure, labor_market, regulatory
        )
        
        # Generate insights
        advantages = self._identify_key_advantages(infrastructure, labor_market, regulatory)
        challenges = self._identify_key_challenges(infrastructure, labor_market, regulatory)
        strategies = self._generate_mitigation_strategies(challenges, infrastructure, labor_market, regulatory)
        
        return LocationFactorAnalysis(
            location=location,
            infrastructure_index=infrastructure,
            labor_market_index=labor_market,
            regulatory_index=regulatory,
            capital_cost_factor=capital_factor,
            operational_cost_factor=operational_factor,
            total_risk_score=risk_score,
            key_advantages=advantages,
            key_challenges=challenges,
            mitigation_strategies=strategies
        )
    
    def _calculate_capital_cost_factor(self, 
                                     infrastructure: InfrastructureIndex,
                                     labor_market: LaborMarketIndex,
                                     regulatory: RegulatoryComplexityIndex,
                                     plant_type: str,
                                     capacity_scale: str) -> float:
        """Calculate capital cost adjustment factor"""
        model = self.capital_cost_model
        
        # Infrastructure factor
        infra_score = infrastructure.composite_score
        if infra_score >= model["infrastructure_thresholds"]["excellent"]:
            infra_factor = model["base_factors"]["infrastructure_excellent"]
        elif infra_score >= model["infrastructure_thresholds"]["good"]:
            infra_factor = 1.0
        elif infra_score >= model["infrastructure_thresholds"]["moderate"]:
            infra_factor = 1.15
        else:
            infra_factor = model["base_factors"]["infrastructure_poor"]
        
        # Regulatory factor
        reg_score = regulatory.composite_complexity
        if reg_score <= model["regulatory_thresholds"]["simple"]:
            reg_factor = model["base_factors"]["regulatory_simple"]
        elif reg_score <= model["regulatory_thresholds"]["moderate"]:
            reg_factor = 1.0
        else:
            reg_factor = model["base_factors"]["regulatory_complex"]
        
        # Construction labor factor
        construction_cost = labor_market.construction_cost
        if construction_cost <= 0.5:
            labor_factor = model["base_factors"]["labor_cheap"]
        elif construction_cost <= 1.0:
            labor_factor = 1.0
        else:
            labor_factor = model["base_factors"]["labor_expensive"]
        
        # Weighted combination
        capital_factor = (
            infra_factor * model["infrastructure_weight"] +
            reg_factor * model["regulatory_weight"] +
            labor_factor * model["labor_construction_weight"]
        )
        
        # Adjust for plant type and capacity scale
        capital_factor *= self._get_plant_type_adjustment(plant_type, "capital")
        capital_factor *= self._get_capacity_scale_adjustment(capacity_scale, "capital")
        
        return capital_factor
    
    def _calculate_operational_cost_factor(self, 
                                         infrastructure: InfrastructureIndex,
                                         labor_market: LaborMarketIndex,
                                         regulatory: RegulatoryComplexityIndex,
                                         plant_type: str,
                                         capacity_scale: str) -> float:
        """Calculate operational cost adjustment factor"""
        model = self.operational_cost_model
        
        # Labor cost factor (most significant for operations)
        labor_cost = labor_market.composite_cost
        if labor_cost <= model["labor_cost_thresholds"]["very_cheap"]:
            labor_factor = model["base_factors"]["labor_very_cheap"]
        elif labor_cost <= model["labor_cost_thresholds"]["cheap"]:
            labor_factor = model["base_factors"]["labor_cheap"]
        elif labor_cost <= model["labor_cost_thresholds"]["moderate"]:
            labor_factor = model["base_factors"]["labor_moderate"]
        else:
            labor_factor = model["base_factors"]["labor_expensive"]
        
        # Utilities factor
        utilities_score = infrastructure.utilities_score
        if utilities_score >= model["utilities_thresholds"]["excellent"]:
            utilities_factor = model["base_factors"]["utilities_excellent"]
        elif utilities_score >= model["utilities_thresholds"]["good"]:
            utilities_factor = 1.0
        elif utilities_score >= model["utilities_thresholds"]["moderate"]:
            utilities_factor = 1.15
        else:
            utilities_factor = model["base_factors"]["utilities_poor"]
        
        # Regulatory compliance factor
        compliance_cost = regulatory.compliance_costs
        if compliance_cost <= 1.0:
            compliance_factor = model["base_factors"]["compliance_simple"]
        else:
            compliance_factor = model["base_factors"]["compliance_complex"]
        
        # Logistics factor
        logistics_score = infrastructure.ports_logistics_score
        if logistics_score >= 7.0:
            logistics_factor = model["base_factors"]["logistics_excellent"]
        elif logistics_score >= 5.0:
            logistics_factor = 1.0
        else:
            logistics_factor = model["base_factors"]["logistics_poor"]
        
        # Weighted combination
        operational_factor = (
            labor_factor * model["labor_cost_weight"] +
            utilities_factor * model["utilities_weight"] +
            compliance_factor * model["regulatory_compliance_weight"] +
            logistics_factor * model["logistics_weight"]
        )
        
        # Adjust for plant type and capacity scale
        operational_factor *= self._get_plant_type_adjustment(plant_type, "operational")
        operational_factor *= self._get_capacity_scale_adjustment(capacity_scale, "operational")
        
        return operational_factor
    
    def _calculate_total_risk_score(self, 
                                  infrastructure: InfrastructureIndex,
                                  labor_market: LaborMarketIndex,
                                  regulatory: RegulatoryComplexityIndex) -> float:
        """Calculate total risk score for the location"""
        # Infrastructure risk (inverted - lower score = higher risk)
        infra_risk = (10 - infrastructure.composite_score) / 10
        
        # Labor market risk (combination of availability and cost volatility)
        labor_availability_risk = (10 - labor_market.composite_availability) / 10
        labor_cost_risk = abs(labor_market.composite_cost - 1.0) * 0.5  # Deviation from baseline
        labor_risk = (labor_availability_risk + labor_cost_risk) / 2
        
        # Regulatory risk (higher complexity = higher risk)
        regulatory_risk = regulatory.composite_complexity / 10
        
        # Political stability risk (inverted)
        political_risk = (10 - regulatory.political_stability) / 10
        
        # Weighted combination
        total_risk = (
            infra_risk * 0.25 +
            labor_risk * 0.25 +
            regulatory_risk * 0.25 +
            political_risk * 0.25
        )
        
        return min(1.0, max(0.0, total_risk))
    
    def _get_plant_type_adjustment(self, plant_type: str, cost_type: str) -> float:
        """Get adjustment factor based on plant type"""
        adjustments = {
            "chemical_processing": {"capital": 1.0, "operational": 1.0},
            "petroleum_refining": {"capital": 1.1, "operational": 1.05},
            "pharmaceutical": {"capital": 1.2, "operational": 1.1},
            "food_processing": {"capital": 0.9, "operational": 0.95},
            "fertilizer": {"capital": 1.0, "operational": 1.0},
            "biofuel": {"capital": 0.95, "operational": 0.98}
        }
        return adjustments.get(plant_type, {"capital": 1.0, "operational": 1.0})[cost_type]
    
    def _get_capacity_scale_adjustment(self, capacity_scale: str, cost_type: str) -> float:
        """Get adjustment factor based on capacity scale"""
        adjustments = {
            "small": {"capital": 1.05, "operational": 1.02},
            "medium": {"capital": 1.0, "operational": 1.0},
            "large": {"capital": 0.95, "operational": 0.98}
        }
        return adjustments.get(capacity_scale, {"capital": 1.0, "operational": 1.0})[cost_type]
    
    def _identify_key_advantages(self, 
                               infrastructure: InfrastructureIndex,
                               labor_market: LaborMarketIndex,
                               regulatory: RegulatoryComplexityIndex) -> List[str]:
        """Identify key advantages of the location"""
        advantages = []
        
        # Infrastructure advantages
        if infrastructure.composite_score >= 7.0:
            advantages.append("Excellent overall infrastructure quality")
        if infrastructure.transportation_score >= 7.0:
            advantages.append("Well-developed transportation network")
        if infrastructure.utilities_score >= 7.0:
            advantages.append("Reliable utilities infrastructure")
        if infrastructure.ports_logistics_score >= 7.0:
            advantages.append("Efficient port and logistics facilities")
        
        # Labor market advantages
        if labor_market.composite_availability >= 7.0:
            advantages.append("Good availability of skilled workforce")
        if labor_market.composite_cost <= 0.7:
            advantages.append("Competitive labor costs")
        if labor_market.labor_flexibility >= 7.5:
            advantages.append("Flexible labor market conditions")
        
        # Regulatory advantages
        if regulatory.composite_complexity <= 5.0:
            advantages.append("Relatively simple regulatory environment")
        if regulatory.political_stability >= 7.0:
            advantages.append("Stable political environment")
        if regulatory.approval_timeframes <= 18.0:
            advantages.append("Reasonable project approval timeframes")
        
        return advantages
    
    def _identify_key_challenges(self, 
                               infrastructure: InfrastructureIndex,
                               labor_market: LaborMarketIndex,
                               regulatory: RegulatoryComplexityIndex) -> List[str]:
        """Identify key challenges of the location"""
        challenges = []
        
        # Infrastructure challenges
        if infrastructure.composite_score <= 5.0:
            challenges.append("Limited infrastructure quality")
        if infrastructure.utilities_score <= 5.0:
            challenges.append("Unreliable utilities supply")
        if infrastructure.transportation_score <= 5.0:
            challenges.append("Poor transportation infrastructure")
        
        # Labor market challenges
        if labor_market.composite_availability <= 6.0:
            challenges.append("Limited skilled workforce availability")
        if labor_market.productivity_index <= 6.0:
            challenges.append("Low labor productivity levels")
        
        # Regulatory challenges
        if regulatory.composite_complexity >= 7.0:
            challenges.append("Complex regulatory environment")
        if regulatory.political_stability <= 6.0:
            challenges.append("Political instability risks")
        if regulatory.approval_timeframes >= 24.0:
            challenges.append("Long project approval processes")
        
        return challenges
    
    def _generate_mitigation_strategies(self, 
                                      challenges: List[str],
                                      infrastructure: InfrastructureIndex,
                                      labor_market: LaborMarketIndex,
                                      regulatory: RegulatoryComplexityIndex) -> List[str]:
        """Generate mitigation strategies for identified challenges"""
        strategies = []
        
        # Infrastructure mitigation
        if infrastructure.utilities_score <= 5.0:
            strategies.append("Invest in backup power generation and water treatment")
        if infrastructure.transportation_score <= 5.0:
            strategies.append("Establish dedicated logistics partnerships")
        
        # Labor market mitigation
        if labor_market.composite_availability <= 6.0:
            strategies.append("Implement comprehensive training programs")
            strategies.append("Partner with local educational institutions")
        
        # Regulatory mitigation
        if regulatory.composite_complexity >= 7.0:
            strategies.append("Engage local regulatory consultants early")
            strategies.append("Build relationships with regulatory authorities")
        if regulatory.political_stability <= 6.0:
            strategies.append("Obtain political risk insurance")
            strategies.append("Diversify operational locations")
        
        return strategies
    
    def _create_default_analysis(self, location: str) -> LocationFactorAnalysis:
        """Create default analysis for unknown locations"""
        return LocationFactorAnalysis(
            location=location,
            infrastructure_index=InfrastructureIndex(
                location=location,
                transportation_score=5.0,
                utilities_score=5.0,
                telecommunications_score=5.0,
                industrial_parks_score=5.0,
                ports_logistics_score=5.0,
                financial_services_score=5.0
            ),
            labor_market_index=LaborMarketIndex(
                location=location,
                skilled_technical_availability=5.0,
                skilled_technical_cost=1.0,
                management_availability=5.0,
                management_cost=1.0,
                operators_availability=5.0,
                operators_cost=1.0,
                maintenance_availability=5.0,
                maintenance_cost=1.0,
                administrative_availability=5.0,
                administrative_cost=1.0,
                construction_availability=5.0,
                construction_cost=1.0,
                productivity_index=5.0,
                labor_flexibility=5.0
            ),
            regulatory_index=RegulatoryComplexityIndex(
                location=location,
                environmental_complexity=5.0,
                safety_complexity=5.0,
                zoning_complexity=5.0,
                permits_complexity=5.0,
                taxation_complexity=5.0,
                labor_laws_complexity=5.0,
                import_export_complexity=5.0,
                approval_timeframes=18.0,
                compliance_costs=1.0,
                political_stability=5.0
            ),
            capital_cost_factor=1.0,
            operational_cost_factor=1.0,
            total_risk_score=0.5,
            key_advantages=["Location data not available"],
            key_challenges=["Limited location information"],
            mitigation_strategies=["Conduct detailed location assessment"]
        )
    
    def compare_locations(self, locations: List[str]) -> Dict[str, Any]:
        """Compare multiple locations across all factors"""
        analyses = {}
        for location in locations:
            analyses[location] = self.analyze_location_factors(location)
        
        # Create comparison matrix
        comparison = {
            "locations": locations,
            "capital_cost_factors": {loc: analyses[loc].capital_cost_factor for loc in locations},
            "operational_cost_factors": {loc: analyses[loc].operational_cost_factor for loc in locations},
            "risk_scores": {loc: analyses[loc].total_risk_score for loc in locations},
            "infrastructure_scores": {loc: analyses[loc].infrastructure_index.composite_score for loc in locations},
            "labor_availability": {loc: analyses[loc].labor_market_index.composite_availability for loc in locations},
            "labor_costs": {loc: analyses[loc].labor_market_index.composite_cost for loc in locations},
            "regulatory_complexity": {loc: analyses[loc].regulatory_index.composite_complexity for loc in locations},
            "detailed_analyses": analyses
        }
        
        # Rank locations
        comparison["rankings"] = {
            "lowest_capital_cost": sorted(locations, key=lambda x: analyses[x].capital_cost_factor),
            "lowest_operational_cost": sorted(locations, key=lambda x: analyses[x].operational_cost_factor),
            "lowest_risk": sorted(locations, key=lambda x: analyses[x].total_risk_score),
            "best_infrastructure": sorted(locations, key=lambda x: analyses[x].infrastructure_index.composite_score, reverse=True),
            "best_labor_availability": sorted(locations, key=lambda x: analyses[x].labor_market_index.composite_availability, reverse=True)
        }
        
        return comparison
    
    def generate_location_report(self, 
                               analysis: LocationFactorAnalysis,
                               include_detailed_breakdown: bool = True) -> str:
        """Generate comprehensive location factor report"""
        report = []
        report.append("="*80)
        report.append("ENHANCED LOCATION FACTOR ANALYSIS REPORT")
        report.append("="*80)
        report.append(f"Location: {analysis.location}")
        report.append(f"Analysis Date: {analysis.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Capital Cost Factor: {analysis.capital_cost_factor:.3f}")
        report.append(f"Operational Cost Factor: {analysis.operational_cost_factor:.3f}")
        report.append(f"Total Risk Score: {analysis.total_risk_score:.3f}")
        report.append("")
        
        # Key Insights
        report.append("KEY ADVANTAGES")
        report.append("-" * 40)
        for advantage in analysis.key_advantages:
            report.append(f"• {advantage}")
        report.append("")
        
        report.append("KEY CHALLENGES")
        report.append("-" * 40)
        for challenge in analysis.key_challenges:
            report.append(f"• {challenge}")
        report.append("")
        
        report.append("MITIGATION STRATEGIES")
        report.append("-" * 40)
        for strategy in analysis.mitigation_strategies:
            report.append(f"• {strategy}")
        report.append("")
        
        if include_detailed_breakdown:
            # Infrastructure Analysis
            report.append("INFRASTRUCTURE ANALYSIS")
            report.append("-" * 40)
            infra = analysis.infrastructure_index
            report.append(f"Composite Score: {infra.composite_score:.1f}/10")
            report.append(f"• Transportation: {infra.transportation_score:.1f}/10")
            report.append(f"• Utilities: {infra.utilities_score:.1f}/10")
            report.append(f"• Telecommunications: {infra.telecommunications_score:.1f}/10")
            report.append(f"• Industrial Parks: {infra.industrial_parks_score:.1f}/10")
            report.append(f"• Ports & Logistics: {infra.ports_logistics_score:.1f}/10")
            report.append(f"• Financial Services: {infra.financial_services_score:.1f}/10")
            report.append("")
            
            # Labor Market Analysis
            report.append("LABOR MARKET ANALYSIS")
            report.append("-" * 40)
            labor = analysis.labor_market_index
            report.append(f"Availability Score: {labor.composite_availability:.1f}/10")
            report.append(f"Cost Index: {labor.composite_cost:.2f} (baseline = 1.0)")
            report.append(f"Productivity Index: {labor.productivity_index:.1f}/10")
            report.append(f"Labor Flexibility: {labor.labor_flexibility:.1f}/10")
            report.append("")
            
            # Regulatory Analysis
            report.append("REGULATORY ANALYSIS")
            report.append("-" * 40)
            reg = analysis.regulatory_index
            report.append(f"Complexity Score: {reg.composite_complexity:.1f}/10")
            report.append(f"Political Stability: {reg.political_stability:.1f}/10")
            report.append(f"Approval Timeframes: {reg.approval_timeframes:.1f} months")
            report.append(f"Compliance Cost Factor: {reg.compliance_costs:.2f}")
            report.append("")
        
        return "\n".join(report)
    
    def save_analysis_results(self, 
                            analysis: LocationFactorAnalysis,
                            filename: Optional[str] = None) -> Path:
        """Save location factor analysis results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"location_factor_analysis_{analysis.location.replace(' ', '_')}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            "location": analysis.location,
            "analysis_timestamp": analysis.analysis_timestamp.isoformat(),
            "cost_factors": {
                "capital_cost_factor": analysis.capital_cost_factor,
                "operational_cost_factor": analysis.operational_cost_factor,
                "total_risk_score": analysis.total_risk_score
            },
            "infrastructure_index": {
                "composite_score": analysis.infrastructure_index.composite_score,
                "transportation_score": analysis.infrastructure_index.transportation_score,
                "utilities_score": analysis.infrastructure_index.utilities_score,
                "telecommunications_score": analysis.infrastructure_index.telecommunications_score,
                "industrial_parks_score": analysis.infrastructure_index.industrial_parks_score,
                "ports_logistics_score": analysis.infrastructure_index.ports_logistics_score,
                "financial_services_score": analysis.infrastructure_index.financial_services_score
            },
            "labor_market_index": {
                "composite_availability": analysis.labor_market_index.composite_availability,
                "composite_cost": analysis.labor_market_index.composite_cost,
                "productivity_index": analysis.labor_market_index.productivity_index,
                "labor_flexibility": analysis.labor_market_index.labor_flexibility
            },
            "regulatory_index": {
                "composite_complexity": analysis.regulatory_index.composite_complexity,
                "political_stability": analysis.regulatory_index.political_stability,
                "approval_timeframes": analysis.regulatory_index.approval_timeframes,
                "compliance_costs": analysis.regulatory_index.compliance_costs
            },
            "insights": {
                "key_advantages": analysis.key_advantages,
                "key_challenges": analysis.key_challenges,
                "mitigation_strategies": analysis.mitigation_strategies
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Location factor analysis saved to {filepath}")
        return filepath 