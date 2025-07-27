"""
Enhanced Quantitative Risk Assessment Module

This module provides comprehensive quantifiable risk assessment incorporating:
- Political stability index thresholds (with numerical action points)
- Infrastructure quality scoring (scale: 1-10)
- Market volatility index (historical fluctuation analysis)

Output: Composite risk score with thresholds for High, Medium, and Low risk
"""

import json
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

from ..utils.logger import logger


class RiskLevel(Enum):
    """Risk level classifications"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class RiskCategory(Enum):
    """Risk assessment categories"""
    POLITICAL = "political"
    INFRASTRUCTURE = "infrastructure"
    MARKET = "market"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    ENVIRONMENTAL = "environmental"
    REGULATORY = "regulatory"


@dataclass
class PoliticalStabilityIndex:
    """Political stability assessment with numerical thresholds"""
    location: str
    governance_effectiveness: float  # 0-10
    political_stability: float  # 0-10
    regulatory_quality: float  # 0-10
    rule_of_law: float  # 0-10
    corruption_control: float  # 0-10
    democratic_accountability: float  # 0-10
    government_stability: float  # 0-10
    conflict_risk: float  # 0-10 (higher = more stable)
    policy_continuity: float  # 0-10
    international_relations: float  # 0-10
    composite_score: float = field(init=False)
    risk_level: RiskLevel = field(init=False)
    action_points: List[str] = field(init=False)
    
    def __post_init__(self):
        """Calculate composite political stability score"""
        weights = {
            'governance_effectiveness': 0.15,
            'political_stability': 0.15,
            'regulatory_quality': 0.10,
            'rule_of_law': 0.15,
            'corruption_control': 0.10,
            'democratic_accountability': 0.10,
            'government_stability': 0.10,
            'conflict_risk': 0.10,
            'policy_continuity': 0.10,
            'international_relations': 0.05
        }
        
        self.composite_score = sum(
            getattr(self, attr) * weight
            for attr, weight in weights.items()
        )
        
        # Determine risk level and action points
        if self.composite_score >= 7.5:
            self.risk_level = RiskLevel.LOW
            self.action_points = ["Monitor quarterly", "Standard due diligence"]
        elif self.composite_score >= 5.5:
            self.risk_level = RiskLevel.MEDIUM
            self.action_points = ["Enhanced monitoring", "Political risk insurance recommended"]
        elif self.composite_score >= 3.5:
            self.risk_level = RiskLevel.HIGH
            self.action_points = ["Frequent monitoring", "Political risk insurance required", "Contingency planning"]
        else:
            self.risk_level = RiskLevel.CRITICAL
            self.action_points = ["Continuous monitoring", "Maximum insurance coverage", "Exit strategy planning"]


@dataclass
class InfrastructureQualityScore:
    """Infrastructure quality assessment (1-10 scale)"""
    location: str
    power_grid_reliability: float  # 1-10
    water_supply_quality: float  # 1-10
    transportation_network: float  # 1-10
    telecommunications: float  # 1-10
    waste_management: float  # 1-10
    industrial_zones: float  # 1-10
    port_facilities: float  # 1-10
    maintenance_standards: float  # 1-10
    expansion_capacity: float  # 1-10
    emergency_services: float  # 1-10
    composite_score: float = field(init=False)
    risk_level: RiskLevel = field(init=False)
    critical_gaps: List[str] = field(init=False)
    
    def __post_init__(self):
        """Calculate composite infrastructure quality score"""
        weights = {
            'power_grid_reliability': 0.20,
            'water_supply_quality': 0.15,
            'transportation_network': 0.15,
            'telecommunications': 0.10,
            'waste_management': 0.10,
            'industrial_zones': 0.10,
            'port_facilities': 0.10,
            'maintenance_standards': 0.05,
            'expansion_capacity': 0.05,
            'emergency_services': 0.05
        }
        
        self.composite_score = sum(
            getattr(self, attr) * weight
            for attr, weight in weights.items()
        )
        
        # Determine risk level
        if self.composite_score >= 8.0:
            self.risk_level = RiskLevel.LOW
        elif self.composite_score >= 6.0:
            self.risk_level = RiskLevel.MEDIUM
        elif self.composite_score >= 4.0:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL
        
        # Identify critical gaps
        self.critical_gaps = []
        if self.power_grid_reliability < 5.0:
            self.critical_gaps.append("Power grid unreliable")
        if self.water_supply_quality < 5.0:
            self.critical_gaps.append("Water supply inadequate")
        if self.transportation_network < 5.0:
            self.critical_gaps.append("Transportation network poor")
        if self.telecommunications < 5.0:
            self.critical_gaps.append("Telecommunications limited")


@dataclass
class MarketVolatilityIndex:
    """Market volatility assessment with historical fluctuation analysis"""
    location: str
    product_name: str
    price_volatility: float  # Coefficient of variation
    demand_volatility: float  # Coefficient of variation
    supply_volatility: float  # Coefficient of variation
    currency_volatility: float  # Coefficient of variation
    commodity_correlation: float  # Correlation with major commodities
    economic_sensitivity: float  # Sensitivity to economic cycles
    seasonal_fluctuation: float  # Seasonal variation coefficient
    trade_disruption_risk: float  # 0-10 scale
    market_maturity: float  # 0-10 scale (higher = more mature)
    competition_intensity: float  # 0-10 scale
    composite_score: float = field(init=False)
    risk_level: RiskLevel = field(init=False)
    volatility_drivers: List[str] = field(init=False)
    
    def __post_init__(self):
        """Calculate composite market volatility score"""
        # Normalize volatility measures to 0-10 scale
        normalized_price_vol = min(10, self.price_volatility * 10)
        normalized_demand_vol = min(10, self.demand_volatility * 10)
        normalized_supply_vol = min(10, self.supply_volatility * 10)
        normalized_currency_vol = min(10, self.currency_volatility * 10)
        
        # Calculate composite (higher score = higher volatility/risk)
        self.composite_score = (
            normalized_price_vol * 0.25 +
            normalized_demand_vol * 0.20 +
            normalized_supply_vol * 0.15 +
            normalized_currency_vol * 0.15 +
            (10 - self.market_maturity) * 0.10 +
            self.trade_disruption_risk * 0.10 +
            self.competition_intensity * 0.05
        )
        
        # Determine risk level
        if self.composite_score <= 3.0:
            self.risk_level = RiskLevel.LOW
        elif self.composite_score <= 6.0:
            self.risk_level = RiskLevel.MEDIUM
        elif self.composite_score <= 8.0:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL
        
        # Identify volatility drivers
        self.volatility_drivers = []
        if self.price_volatility > 0.3:
            self.volatility_drivers.append("High price volatility")
        if self.demand_volatility > 0.25:
            self.volatility_drivers.append("Unstable demand patterns")
        if self.currency_volatility > 0.2:
            self.volatility_drivers.append("Currency fluctuations")
        if self.trade_disruption_risk > 6.0:
            self.volatility_drivers.append("Trade disruption risks")


@dataclass
class OperationalRiskFactors:
    """Operational risk assessment"""
    location: str
    labor_disputes_risk: float  # 0-10
    supplier_reliability: float  # 0-10 (higher = more reliable)
    technology_risk: float  # 0-10
    maintenance_challenges: float  # 0-10
    safety_incidents_risk: float  # 0-10
    environmental_compliance: float  # 0-10 (higher = better compliance)
    capacity_utilization_risk: float  # 0-10
    quality_control_risk: float  # 0-10
    composite_score: float = field(init=False)
    risk_level: RiskLevel = field(init=False)
    
    def __post_init__(self):
        """Calculate composite operational risk score"""
        # Convert reliable factors to risk factors
        supplier_risk = 10 - self.supplier_reliability
        compliance_risk = 10 - self.environmental_compliance
        
        self.composite_score = (
            self.labor_disputes_risk * 0.20 +
            supplier_risk * 0.15 +
            self.technology_risk * 0.15 +
            self.maintenance_challenges * 0.15 +
            self.safety_incidents_risk * 0.15 +
            compliance_risk * 0.10 +
            self.capacity_utilization_risk * 0.05 +
            self.quality_control_risk * 0.05
        )
        
        # Determine risk level
        if self.composite_score <= 3.0:
            self.risk_level = RiskLevel.LOW
        elif self.composite_score <= 5.0:
            self.risk_level = RiskLevel.MEDIUM
        elif self.composite_score <= 7.0:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL


@dataclass
class FinancialRiskFactors:
    """Financial risk assessment"""
    location: str
    currency_risk: float  # 0-10
    inflation_risk: float  # 0-10
    interest_rate_risk: float  # 0-10
    credit_risk: float  # 0-10
    liquidity_risk: float  # 0-10
    commodity_price_risk: float  # 0-10
    taxation_risk: float  # 0-10
    banking_system_stability: float  # 0-10 (higher = more stable)
    composite_score: float = field(init=False)
    risk_level: RiskLevel = field(init=False)
    
    def __post_init__(self):
        """Calculate composite financial risk score"""
        banking_risk = 10 - self.banking_system_stability
        
        self.composite_score = (
            self.currency_risk * 0.20 +
            self.inflation_risk * 0.15 +
            self.interest_rate_risk * 0.15 +
            self.credit_risk * 0.15 +
            self.liquidity_risk * 0.10 +
            self.commodity_price_risk * 0.10 +
            self.taxation_risk * 0.10 +
            banking_risk * 0.05
        )
        
        # Determine risk level
        if self.composite_score <= 3.0:
            self.risk_level = RiskLevel.LOW
        elif self.composite_score <= 5.0:
            self.risk_level = RiskLevel.MEDIUM
        elif self.composite_score <= 7.0:
            self.risk_level = RiskLevel.HIGH
        else:
            self.risk_level = RiskLevel.CRITICAL


@dataclass
class CompositeRiskAssessment:
    """Comprehensive risk assessment result"""
    location: str
    product_name: str
    political_stability: PoliticalStabilityIndex
    infrastructure_quality: InfrastructureQualityScore
    market_volatility: MarketVolatilityIndex
    operational_risks: OperationalRiskFactors
    financial_risks: FinancialRiskFactors
    composite_score: float
    overall_risk_level: RiskLevel
    risk_breakdown: Dict[str, float]
    critical_risk_factors: List[str]
    risk_mitigation_priorities: List[str]
    recommended_actions: List[str]
    assessment_timestamp: datetime = field(default_factory=datetime.now)


class QuantitativeRiskAssessor:
    """
    Comprehensive quantitative risk assessment system
    
    Provides detailed risk analysis with numerical thresholds and actionable insights
    for chemical plant investment decisions in African markets.
    """
    
    def __init__(self, data_dir: str = "data/risk_assessment"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize risk databases
        self.political_data = self._initialize_political_data()
        self.infrastructure_data = self._initialize_infrastructure_data()
        self.market_data = self._initialize_market_data()
        self.operational_data = self._initialize_operational_data()
        self.financial_data = self._initialize_financial_data()
        
        # Risk thresholds and weights
        self.risk_weights = {
            'political': 0.25,
            'infrastructure': 0.20,
            'market': 0.20,
            'operational': 0.20,
            'financial': 0.15
        }
        
        # Composite risk thresholds
        self.composite_thresholds = {
            RiskLevel.LOW: 3.0,
            RiskLevel.MEDIUM: 5.0,
            RiskLevel.HIGH: 7.0,
            RiskLevel.CRITICAL: 10.0
        }
        
        logger.info("QuantitativeRiskAssessor initialized with comprehensive databases")
    
    def _initialize_political_data(self) -> Dict[str, PoliticalStabilityIndex]:
        """Initialize political stability data for African locations"""
        return {
            "South Africa": PoliticalStabilityIndex(
                location="South Africa",
                governance_effectiveness=6.5,
                political_stability=6.0,
                regulatory_quality=7.0,
                rule_of_law=5.5,
                corruption_control=4.5,
                democratic_accountability=7.5,
                government_stability=6.0,
                conflict_risk=7.0,
                policy_continuity=6.5,
                international_relations=7.5
            ),
            "Nigeria": PoliticalStabilityIndex(
                location="Nigeria",
                governance_effectiveness=4.0,
                political_stability=3.5,
                regulatory_quality=4.5,
                rule_of_law=3.0,
                corruption_control=2.5,
                democratic_accountability=5.0,
                government_stability=4.0,
                conflict_risk=4.5,
                policy_continuity=4.5,
                international_relations=5.5
            ),
            "Kenya": PoliticalStabilityIndex(
                location="Kenya",
                governance_effectiveness=5.0,
                political_stability=5.5,
                regulatory_quality=5.5,
                rule_of_law=4.5,
                corruption_control=3.5,
                democratic_accountability=6.0,
                government_stability=5.5,
                conflict_risk=6.0,
                policy_continuity=5.0,
                international_relations=6.5
            ),
            "Ghana": PoliticalStabilityIndex(
                location="Ghana",
                governance_effectiveness=6.0,
                political_stability=6.5,
                regulatory_quality=6.0,
                rule_of_law=5.5,
                corruption_control=5.0,
                democratic_accountability=7.0,
                government_stability=6.5,
                conflict_risk=7.5,
                policy_continuity=6.0,
                international_relations=7.0
            ),
            "Morocco": PoliticalStabilityIndex(
                location="Morocco",
                governance_effectiveness=5.5,
                political_stability=6.0,
                regulatory_quality=6.5,
                rule_of_law=5.0,
                corruption_control=4.5,
                democratic_accountability=4.5,
                government_stability=7.0,
                conflict_risk=7.5,
                policy_continuity=6.5,
                international_relations=7.5
            ),
            "Egypt": PoliticalStabilityIndex(
                location="Egypt",
                governance_effectiveness=4.5,
                political_stability=4.0,
                regulatory_quality=5.0,
                rule_of_law=4.0,
                corruption_control=3.5,
                democratic_accountability=3.0,
                government_stability=5.5,
                conflict_risk=5.0,
                policy_continuity=5.0,
                international_relations=6.0
            ),
            "Tanzania": PoliticalStabilityIndex(
                location="Tanzania",
                governance_effectiveness=5.0,
                political_stability=6.0,
                regulatory_quality=4.5,
                rule_of_law=4.0,
                corruption_control=3.0,
                democratic_accountability=5.5,
                government_stability=6.0,
                conflict_risk=7.0,
                policy_continuity=5.5,
                international_relations=6.0
            ),
            "Ethiopia": PoliticalStabilityIndex(
                location="Ethiopia",
                governance_effectiveness=4.0,
                political_stability=3.0,
                regulatory_quality=4.0,
                rule_of_law=3.5,
                corruption_control=3.0,
                democratic_accountability=3.5,
                government_stability=4.5,
                conflict_risk=3.5,
                policy_continuity=4.0,
                international_relations=5.0
            )
        }
    
    def _initialize_infrastructure_data(self) -> Dict[str, InfrastructureQualityScore]:
        """Initialize infrastructure quality data for African locations"""
        return {
            "South Africa": InfrastructureQualityScore(
                location="South Africa",
                power_grid_reliability=7.0,
                water_supply_quality=7.5,
                transportation_network=7.5,
                telecommunications=8.0,
                waste_management=7.0,
                industrial_zones=8.5,
                port_facilities=8.0,
                maintenance_standards=7.5,
                expansion_capacity=7.0,
                emergency_services=8.0
            ),
            "Nigeria": InfrastructureQualityScore(
                location="Nigeria",
                power_grid_reliability=3.5,
                water_supply_quality=4.0,
                transportation_network=4.5,
                telecommunications=6.0,
                waste_management=3.5,
                industrial_zones=5.0,
                port_facilities=5.5,
                maintenance_standards=4.0,
                expansion_capacity=5.5,
                emergency_services=4.5
            ),
            "Kenya": InfrastructureQualityScore(
                location="Kenya",
                power_grid_reliability=5.0,
                water_supply_quality=5.5,
                transportation_network=6.0,
                telecommunications=7.0,
                waste_management=5.0,
                industrial_zones=6.0,
                port_facilities=7.0,
                maintenance_standards=5.5,
                expansion_capacity=6.0,
                emergency_services=6.0
            ),
            "Ghana": InfrastructureQualityScore(
                location="Ghana",
                power_grid_reliability=5.5,
                water_supply_quality=6.0,
                transportation_network=6.5,
                telecommunications=7.0,
                waste_management=5.5,
                industrial_zones=6.0,
                port_facilities=6.5,
                maintenance_standards=6.0,
                expansion_capacity=6.0,
                emergency_services=6.5
            ),
            "Morocco": InfrastructureQualityScore(
                location="Morocco",
                power_grid_reliability=6.5,
                water_supply_quality=6.5,
                transportation_network=7.0,
                telecommunications=7.5,
                waste_management=6.5,
                industrial_zones=7.0,
                port_facilities=7.5,
                maintenance_standards=7.0,
                expansion_capacity=7.0,
                emergency_services=7.0
            ),
            "Egypt": InfrastructureQualityScore(
                location="Egypt",
                power_grid_reliability=5.5,
                water_supply_quality=5.0,
                transportation_network=6.0,
                telecommunications=7.0,
                waste_management=5.0,
                industrial_zones=6.0,
                port_facilities=8.0,
                maintenance_standards=5.5,
                expansion_capacity=6.5,
                emergency_services=6.0
            ),
            "Tanzania": InfrastructureQualityScore(
                location="Tanzania",
                power_grid_reliability=4.0,
                water_supply_quality=4.0,
                transportation_network=4.5,
                telecommunications=6.0,
                waste_management=4.0,
                industrial_zones=5.0,
                port_facilities=6.0,
                maintenance_standards=4.5,
                expansion_capacity=5.0,
                emergency_services=5.0
            ),
            "Ethiopia": InfrastructureQualityScore(
                location="Ethiopia",
                power_grid_reliability=3.0,
                water_supply_quality=3.5,
                transportation_network=4.0,
                telecommunications=5.5,
                waste_management=3.0,
                industrial_zones=4.5,
                port_facilities=4.0,
                maintenance_standards=3.5,
                expansion_capacity=4.0,
                emergency_services=4.0
            )
        }
    
    def _initialize_market_data(self) -> Dict[Tuple[str, str], MarketVolatilityIndex]:
        """Initialize market volatility data for location-product combinations"""
        data = {}
        
        # Fertilizer market data
        fertilizer_locations = ["South Africa", "Nigeria", "Kenya", "Ghana", "Morocco", "Egypt", "Tanzania", "Ethiopia"]
        for location in fertilizer_locations:
            volatility_factor = self._get_location_volatility_factor(location)
            data[(location, "Fertilizer")] = MarketVolatilityIndex(
                location=location,
                product_name="Fertilizer",
                price_volatility=0.25 * volatility_factor,
                demand_volatility=0.20 * volatility_factor,
                supply_volatility=0.30 * volatility_factor,
                currency_volatility=0.15 * volatility_factor,
                commodity_correlation=0.7,
                economic_sensitivity=7.0,
                seasonal_fluctuation=0.4,
                trade_disruption_risk=5.0 * volatility_factor,
                market_maturity=6.0 / volatility_factor,
                competition_intensity=6.0
            )
        
        # Biofuel market data
        biofuel_locations = ["South Africa", "Nigeria", "Kenya", "Ghana", "Tanzania"]
        for location in biofuel_locations:
            volatility_factor = self._get_location_volatility_factor(location)
            data[(location, "Biofuel")] = MarketVolatilityIndex(
                location=location,
                product_name="Biofuel",
                price_volatility=0.35 * volatility_factor,
                demand_volatility=0.25 * volatility_factor,
                supply_volatility=0.40 * volatility_factor,
                currency_volatility=0.20 * volatility_factor,
                commodity_correlation=0.8,
                economic_sensitivity=8.0,
                seasonal_fluctuation=0.6,
                trade_disruption_risk=6.0 * volatility_factor,
                market_maturity=4.0 / volatility_factor,
                competition_intensity=7.0
            )
        
        return data
    
    def _get_location_volatility_factor(self, location: str) -> float:
        """Get volatility multiplier based on location stability"""
        factors = {
            "South Africa": 1.0,
            "Morocco": 1.1,
            "Ghana": 1.2,
            "Kenya": 1.3,
            "Egypt": 1.4,
            "Tanzania": 1.5,
            "Nigeria": 1.6,
            "Ethiopia": 1.8
        }
        return factors.get(location, 1.5)
    
    def _initialize_operational_data(self) -> Dict[str, OperationalRiskFactors]:
        """Initialize operational risk data for African locations"""
        return {
            "South Africa": OperationalRiskFactors(
                location="South Africa",
                labor_disputes_risk=6.0,
                supplier_reliability=7.0,
                technology_risk=4.0,
                maintenance_challenges=5.0,
                safety_incidents_risk=5.5,
                environmental_compliance=7.5,
                capacity_utilization_risk=4.5,
                quality_control_risk=4.0
            ),
            "Nigeria": OperationalRiskFactors(
                location="Nigeria",
                labor_disputes_risk=5.0,
                supplier_reliability=4.5,
                technology_risk=6.5,
                maintenance_challenges=7.0,
                safety_incidents_risk=6.5,
                environmental_compliance=4.0,
                capacity_utilization_risk=6.0,
                quality_control_risk=6.0
            ),
            "Kenya": OperationalRiskFactors(
                location="Kenya",
                labor_disputes_risk=4.5,
                supplier_reliability=5.5,
                technology_risk=5.5,
                maintenance_challenges=6.0,
                safety_incidents_risk=5.5,
                environmental_compliance=5.5,
                capacity_utilization_risk=5.0,
                quality_control_risk=5.0
            ),
            "Ghana": OperationalRiskFactors(
                location="Ghana",
                labor_disputes_risk=4.0,
                supplier_reliability=6.0,
                technology_risk=5.0,
                maintenance_challenges=5.5,
                safety_incidents_risk=5.0,
                environmental_compliance=6.0,
                capacity_utilization_risk=4.5,
                quality_control_risk=4.5
            ),
            "Morocco": OperationalRiskFactors(
                location="Morocco",
                labor_disputes_risk=5.5,
                supplier_reliability=6.5,
                technology_risk=4.5,
                maintenance_challenges=5.0,
                safety_incidents_risk=4.5,
                environmental_compliance=6.5,
                capacity_utilization_risk=4.0,
                quality_control_risk=4.0
            ),
            "Egypt": OperationalRiskFactors(
                location="Egypt",
                labor_disputes_risk=5.0,
                supplier_reliability=5.0,
                technology_risk=5.5,
                maintenance_challenges=6.0,
                safety_incidents_risk=5.5,
                environmental_compliance=5.0,
                capacity_utilization_risk=5.5,
                quality_control_risk=5.5
            ),
            "Tanzania": OperationalRiskFactors(
                location="Tanzania",
                labor_disputes_risk=4.0,
                supplier_reliability=4.5,
                technology_risk=6.0,
                maintenance_challenges=7.0,
                safety_incidents_risk=6.0,
                environmental_compliance=4.5,
                capacity_utilization_risk=6.0,
                quality_control_risk=6.0
            ),
            "Ethiopia": OperationalRiskFactors(
                location="Ethiopia",
                labor_disputes_risk=6.0,
                supplier_reliability=4.0,
                technology_risk=7.0,
                maintenance_challenges=7.5,
                safety_incidents_risk=7.0,
                environmental_compliance=4.0,
                capacity_utilization_risk=7.0,
                quality_control_risk=7.0
            )
        }
    
    def _initialize_financial_data(self) -> Dict[str, FinancialRiskFactors]:
        """Initialize financial risk data for African locations"""
        return {
            "South Africa": FinancialRiskFactors(
                location="South Africa",
                currency_risk=6.0,
                inflation_risk=5.5,
                interest_rate_risk=5.0,
                credit_risk=4.5,
                liquidity_risk=4.0,
                commodity_price_risk=6.0,
                taxation_risk=6.5,
                banking_system_stability=7.5
            ),
            "Nigeria": FinancialRiskFactors(
                location="Nigeria",
                currency_risk=8.0,
                inflation_risk=7.5,
                interest_rate_risk=7.0,
                credit_risk=7.0,
                liquidity_risk=6.5,
                commodity_price_risk=7.5,
                taxation_risk=6.0,
                banking_system_stability=5.5
            ),
            "Kenya": FinancialRiskFactors(
                location="Kenya",
                currency_risk=6.5,
                inflation_risk=6.0,
                interest_rate_risk=6.0,
                credit_risk=5.5,
                liquidity_risk=5.0,
                commodity_price_risk=6.5,
                taxation_risk=5.5,
                banking_system_stability=6.5
            ),
            "Ghana": FinancialRiskFactors(
                location="Ghana",
                currency_risk=7.0,
                inflation_risk=6.5,
                interest_rate_risk=6.5,
                credit_risk=6.0,
                liquidity_risk=5.5,
                commodity_price_risk=6.0,
                taxation_risk=5.5,
                banking_system_stability=6.0
            ),
            "Morocco": FinancialRiskFactors(
                location="Morocco",
                currency_risk=5.0,
                inflation_risk=4.5,
                interest_rate_risk=4.0,
                credit_risk=4.5,
                liquidity_risk=4.0,
                commodity_price_risk=5.5,
                taxation_risk=5.0,
                banking_system_stability=7.0
            ),
            "Egypt": FinancialRiskFactors(
                location="Egypt",
                currency_risk=7.5,
                inflation_risk=7.0,
                interest_rate_risk=6.5,
                credit_risk=6.0,
                liquidity_risk=6.0,
                commodity_price_risk=6.5,
                taxation_risk=6.0,
                banking_system_stability=6.0
            ),
            "Tanzania": FinancialRiskFactors(
                location="Tanzania",
                currency_risk=6.0,
                inflation_risk=5.5,
                interest_rate_risk=5.5,
                credit_risk=6.0,
                liquidity_risk=5.5,
                commodity_price_risk=6.0,
                taxation_risk=5.0,
                banking_system_stability=5.5
            ),
            "Ethiopia": FinancialRiskFactors(
                location="Ethiopia",
                currency_risk=7.0,
                inflation_risk=8.0,
                interest_rate_risk=7.0,
                credit_risk=7.5,
                liquidity_risk=7.0,
                commodity_price_risk=7.0,
                taxation_risk=6.0,
                banking_system_stability=4.5
            )
        }
    
    def assess_comprehensive_risk(self, 
                                location: str,
                                product_name: str,
                                plant_type: str = "chemical_processing",
                                capacity_scale: str = "medium") -> CompositeRiskAssessment:
        """
        Conduct comprehensive quantitative risk assessment
        
        Args:
            location: Location name
            product_name: Product being manufactured
            plant_type: Type of plant
            capacity_scale: Scale of operation
            
        Returns:
            CompositeRiskAssessment with detailed risk analysis
        """
        # Get individual risk assessments
        political_risk = self.political_data.get(location)
        infrastructure_risk = self.infrastructure_data.get(location)
        market_risk = self.market_data.get((location, product_name))
        operational_risk = self.operational_data.get(location)
        financial_risk = self.financial_data.get(location)
        
        # Handle missing data
        if not all([political_risk, infrastructure_risk, operational_risk, financial_risk]):
            logger.error(f"Incomplete risk data for location: {location}")
            return self._create_default_assessment(location, product_name)
        
        if not market_risk:
            logger.warning(f"No market data for {location} + {product_name}, using default")
            market_risk = self._create_default_market_risk(location, product_name)
        
        # Calculate composite risk score
        composite_score = (
            political_risk.composite_score * self.risk_weights['political'] +
            infrastructure_risk.composite_score * self.risk_weights['infrastructure'] +
            market_risk.composite_score * self.risk_weights['market'] +
            operational_risk.composite_score * self.risk_weights['operational'] +
            financial_risk.composite_score * self.risk_weights['financial']
        )
        
        # Determine overall risk level
        if composite_score <= self.composite_thresholds[RiskLevel.LOW]:
            overall_risk_level = RiskLevel.LOW
        elif composite_score <= self.composite_thresholds[RiskLevel.MEDIUM]:
            overall_risk_level = RiskLevel.MEDIUM
        elif composite_score <= self.composite_thresholds[RiskLevel.HIGH]:
            overall_risk_level = RiskLevel.HIGH
        else:
            overall_risk_level = RiskLevel.CRITICAL
        
        # Create risk breakdown
        risk_breakdown = {
            'political': political_risk.composite_score,
            'infrastructure': infrastructure_risk.composite_score,
            'market': market_risk.composite_score,
            'operational': operational_risk.composite_score,
            'financial': financial_risk.composite_score
        }
        
        # Identify critical risk factors
        critical_factors = self._identify_critical_risk_factors(
            political_risk, infrastructure_risk, market_risk, operational_risk, financial_risk
        )
        
        # Generate mitigation priorities
        mitigation_priorities = self._generate_mitigation_priorities(
            political_risk, infrastructure_risk, market_risk, operational_risk, financial_risk
        )
        
        # Generate recommended actions
        recommended_actions = self._generate_recommended_actions(
            overall_risk_level, critical_factors, location, product_name
        )
        
        return CompositeRiskAssessment(
            location=location,
            product_name=product_name,
            political_stability=political_risk,
            infrastructure_quality=infrastructure_risk,
            market_volatility=market_risk,
            operational_risks=operational_risk,
            financial_risks=financial_risk,
            composite_score=composite_score,
            overall_risk_level=overall_risk_level,
            risk_breakdown=risk_breakdown,
            critical_risk_factors=critical_factors,
            risk_mitigation_priorities=mitigation_priorities,
            recommended_actions=recommended_actions
        )
    
    def _identify_critical_risk_factors(self, 
                                      political: PoliticalStabilityIndex,
                                      infrastructure: InfrastructureQualityScore,
                                      market: MarketVolatilityIndex,
                                      operational: OperationalRiskFactors,
                                      financial: FinancialRiskFactors) -> List[str]:
        """Identify critical risk factors across all categories"""
        critical_factors = []
        
        # Political critical factors
        if political.composite_score <= 4.0:
            critical_factors.append("Political instability")
        if political.corruption_control <= 3.0:
            critical_factors.append("High corruption levels")
        if political.rule_of_law <= 3.0:
            critical_factors.append("Weak rule of law")
        
        # Infrastructure critical factors
        if infrastructure.composite_score <= 4.0:
            critical_factors.append("Poor infrastructure quality")
        critical_factors.extend(infrastructure.critical_gaps)
        
        # Market critical factors
        if market.composite_score >= 7.0:
            critical_factors.append("High market volatility")
        critical_factors.extend(market.volatility_drivers)
        
        # Operational critical factors
        if operational.composite_score >= 6.0:
            critical_factors.append("High operational risk")
        if operational.labor_disputes_risk >= 7.0:
            critical_factors.append("High labor dispute risk")
        if operational.technology_risk >= 7.0:
            critical_factors.append("High technology risk")
        
        # Financial critical factors
        if financial.composite_score >= 6.0:
            critical_factors.append("High financial risk")
        if financial.currency_risk >= 7.0:
            critical_factors.append("High currency volatility")
        if financial.inflation_risk >= 7.0:
            critical_factors.append("High inflation risk")
        
        return critical_factors
    
    def _generate_mitigation_priorities(self, 
                                      political: PoliticalStabilityIndex,
                                      infrastructure: InfrastructureQualityScore,
                                      market: MarketVolatilityIndex,
                                      operational: OperationalRiskFactors,
                                      financial: FinancialRiskFactors) -> List[str]:
        """Generate prioritized mitigation strategies"""
        priorities = []
        
        # Priority 1: Address highest risk category
        risk_scores = {
            'political': political.composite_score,
            'infrastructure': infrastructure.composite_score,
            'market': market.composite_score,
            'operational': operational.composite_score,
            'financial': financial.composite_score
        }
        
        highest_risk = max(risk_scores, key=risk_scores.get)
        
        if highest_risk == 'political':
            priorities.append("Obtain comprehensive political risk insurance")
        elif highest_risk == 'infrastructure':
            priorities.append("Invest in backup infrastructure systems")
        elif highest_risk == 'market':
            priorities.append("Implement hedging strategies for market volatility")
        elif highest_risk == 'operational':
            priorities.append("Strengthen operational risk management systems")
        elif highest_risk == 'financial':
            priorities.append("Establish comprehensive financial risk hedging")
        
        # Priority 2: Address critical infrastructure gaps
        if infrastructure.power_grid_reliability <= 4.0:
            priorities.append("Install backup power generation systems")
        if infrastructure.water_supply_quality <= 4.0:
            priorities.append("Establish independent water treatment facilities")
        
        # Priority 3: Address market volatility
        if market.price_volatility > 0.3:
            priorities.append("Negotiate long-term supply contracts")
        if market.currency_volatility > 0.2:
            priorities.append("Implement currency hedging strategies")
        
        return priorities
    
    def _generate_recommended_actions(self, 
                                    risk_level: RiskLevel,
                                    critical_factors: List[str],
                                    location: str,
                                    product_name: str) -> List[str]:
        """Generate recommended actions based on risk assessment"""
        actions = []
        
        # Risk level specific actions
        if risk_level == RiskLevel.LOW:
            actions.append("Proceed with standard risk management procedures")
            actions.append("Conduct quarterly risk reviews")
        elif risk_level == RiskLevel.MEDIUM:
            actions.append("Implement enhanced monitoring systems")
            actions.append("Conduct monthly risk assessments")
            actions.append("Consider risk insurance for key exposures")
        elif risk_level == RiskLevel.HIGH:
            actions.append("Implement comprehensive risk mitigation strategies")
            actions.append("Conduct weekly risk monitoring")
            actions.append("Obtain extensive insurance coverage")
            actions.append("Develop contingency plans")
        else:  # CRITICAL
            actions.append("Reconsider investment viability")
            actions.append("Implement maximum risk mitigation measures")
            actions.append("Continuous risk monitoring required")
            actions.append("Maintain exit strategy readiness")
        
        # Specific actions for critical factors
        if "Political instability" in critical_factors:
            actions.append("Engage with local political stakeholders")
        if "Poor infrastructure quality" in critical_factors:
            actions.append("Invest in self-sufficient infrastructure")
        if "High market volatility" in critical_factors:
            actions.append("Diversify product portfolio")
        if "High currency volatility" in critical_factors:
            actions.append("Implement multi-currency hedging")
        
        return actions
    
    def _create_default_market_risk(self, location: str, product_name: str) -> MarketVolatilityIndex:
        """Create default market risk assessment"""
        volatility_factor = self._get_location_volatility_factor(location)
        return MarketVolatilityIndex(
            location=location,
            product_name=product_name,
            price_volatility=0.3 * volatility_factor,
            demand_volatility=0.25 * volatility_factor,
            supply_volatility=0.35 * volatility_factor,
            currency_volatility=0.2 * volatility_factor,
            commodity_correlation=0.6,
            economic_sensitivity=7.0,
            seasonal_fluctuation=0.5,
            trade_disruption_risk=5.0 * volatility_factor,
            market_maturity=5.0 / volatility_factor,
            competition_intensity=6.0
        )
    
    def _create_default_assessment(self, location: str, product_name: str) -> CompositeRiskAssessment:
        """Create default risk assessment for unknown locations"""
        return CompositeRiskAssessment(
            location=location,
            product_name=product_name,
            political_stability=PoliticalStabilityIndex(
                location=location,
                governance_effectiveness=5.0,
                political_stability=5.0,
                regulatory_quality=5.0,
                rule_of_law=5.0,
                corruption_control=5.0,
                democratic_accountability=5.0,
                government_stability=5.0,
                conflict_risk=5.0,
                policy_continuity=5.0,
                international_relations=5.0
            ),
            infrastructure_quality=InfrastructureQualityScore(
                location=location,
                power_grid_reliability=5.0,
                water_supply_quality=5.0,
                transportation_network=5.0,
                telecommunications=5.0,
                waste_management=5.0,
                industrial_zones=5.0,
                port_facilities=5.0,
                maintenance_standards=5.0,
                expansion_capacity=5.0,
                emergency_services=5.0
            ),
            market_volatility=self._create_default_market_risk(location, product_name),
            operational_risks=OperationalRiskFactors(
                location=location,
                labor_disputes_risk=5.0,
                supplier_reliability=5.0,
                technology_risk=5.0,
                maintenance_challenges=5.0,
                safety_incidents_risk=5.0,
                environmental_compliance=5.0,
                capacity_utilization_risk=5.0,
                quality_control_risk=5.0
            ),
            financial_risks=FinancialRiskFactors(
                location=location,
                currency_risk=5.0,
                inflation_risk=5.0,
                interest_rate_risk=5.0,
                credit_risk=5.0,
                liquidity_risk=5.0,
                commodity_price_risk=5.0,
                taxation_risk=5.0,
                banking_system_stability=5.0
            ),
            composite_score=5.0,
            overall_risk_level=RiskLevel.MEDIUM,
            risk_breakdown={'political': 5.0, 'infrastructure': 5.0, 'market': 5.0, 'operational': 5.0, 'financial': 5.0},
            critical_risk_factors=["Insufficient location data"],
            risk_mitigation_priorities=["Conduct detailed risk assessment"],
            recommended_actions=["Gather comprehensive location data"]
        )
    
    def generate_risk_report(self, 
                           assessment: CompositeRiskAssessment,
                           include_detailed_breakdown: bool = True) -> str:
        """Generate comprehensive risk assessment report"""
        report = []
        report.append("="*80)
        report.append("QUANTITATIVE RISK ASSESSMENT REPORT")
        report.append("="*80)
        report.append(f"Location: {assessment.location}")
        report.append(f"Product: {assessment.product_name}")
        report.append(f"Assessment Date: {assessment.assessment_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Overall Risk Level: {assessment.overall_risk_level.value}")
        report.append(f"Composite Risk Score: {assessment.composite_score:.2f}/10")
        report.append("")
        
        # Risk Breakdown
        report.append("RISK BREAKDOWN")
        report.append("-" * 40)
        for category, score in assessment.risk_breakdown.items():
            report.append(f"{category.capitalize()}: {score:.2f}/10")
        report.append("")
        
        # Critical Risk Factors
        report.append("CRITICAL RISK FACTORS")
        report.append("-" * 40)
        for factor in assessment.critical_risk_factors:
            report.append(f"• {factor}")
        report.append("")
        
        # Risk Mitigation Priorities
        report.append("RISK MITIGATION PRIORITIES")
        report.append("-" * 40)
        for priority in assessment.risk_mitigation_priorities:
            report.append(f"• {priority}")
        report.append("")
        
        # Recommended Actions
        report.append("RECOMMENDED ACTIONS")
        report.append("-" * 40)
        for action in assessment.recommended_actions:
            report.append(f"• {action}")
        report.append("")
        
        if include_detailed_breakdown:
            # Political Stability Details
            report.append("POLITICAL STABILITY ANALYSIS")
            report.append("-" * 40)
            political = assessment.political_stability
            report.append(f"Composite Score: {political.composite_score:.2f}/10")
            report.append(f"Risk Level: {political.risk_level.value}")
            report.append("Action Points:")
            for action in political.action_points:
                report.append(f"• {action}")
            report.append("")
            
            # Infrastructure Quality Details
            report.append("INFRASTRUCTURE QUALITY ANALYSIS")
            report.append("-" * 40)
            infrastructure = assessment.infrastructure_quality
            report.append(f"Composite Score: {infrastructure.composite_score:.2f}/10")
            report.append(f"Risk Level: {infrastructure.risk_level.value}")
            if infrastructure.critical_gaps:
                report.append("Critical Gaps:")
                for gap in infrastructure.critical_gaps:
                    report.append(f"• {gap}")
            report.append("")
            
            # Market Volatility Details
            report.append("MARKET VOLATILITY ANALYSIS")
            report.append("-" * 40)
            market = assessment.market_volatility
            report.append(f"Composite Score: {market.composite_score:.2f}/10")
            report.append(f"Risk Level: {market.risk_level.value}")
            if market.volatility_drivers:
                report.append("Volatility Drivers:")
                for driver in market.volatility_drivers:
                    report.append(f"• {driver}")
            report.append("")
        
        return "\n".join(report)
    
    def compare_risk_profiles(self, 
                            locations: List[str],
                            product_name: str) -> Dict[str, Any]:
        """Compare risk profiles across multiple locations"""
        assessments = {}
        for location in locations:
            assessments[location] = self.assess_comprehensive_risk(location, product_name)
        
        # Create comparison data
        comparison = {
            "product_name": product_name,
            "locations": locations,
            "composite_scores": {loc: assessments[loc].composite_score for loc in locations},
            "risk_levels": {loc: assessments[loc].overall_risk_level.value for loc in locations},
            "risk_breakdown": {},
            "rankings": {},
            "detailed_assessments": assessments
        }
        
        # Risk breakdown comparison
        for category in ['political', 'infrastructure', 'market', 'operational', 'financial']:
            comparison["risk_breakdown"][category] = {
                loc: assessments[loc].risk_breakdown[category] for loc in locations
            }
        
        # Rankings
        comparison["rankings"] = {
            "lowest_overall_risk": sorted(locations, key=lambda x: assessments[x].composite_score),
            "lowest_political_risk": sorted(locations, key=lambda x: assessments[x].political_stability.composite_score),
            "best_infrastructure": sorted(locations, key=lambda x: assessments[x].infrastructure_quality.composite_score),
            "lowest_market_volatility": sorted(locations, key=lambda x: assessments[x].market_volatility.composite_score),
            "lowest_operational_risk": sorted(locations, key=lambda x: assessments[x].operational_risks.composite_score),
            "lowest_financial_risk": sorted(locations, key=lambda x: assessments[x].financial_risks.composite_score)
        }
        
        return comparison
    
    def save_assessment_results(self, 
                              assessment: CompositeRiskAssessment,
                              filename: Optional[str] = None) -> Path:
        """Save risk assessment results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"risk_assessment_{assessment.location.replace(' ', '_')}_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            "location": assessment.location,
            "product_name": assessment.product_name,
            "assessment_timestamp": assessment.assessment_timestamp.isoformat(),
            "composite_score": assessment.composite_score,
            "overall_risk_level": assessment.overall_risk_level.value,
            "risk_breakdown": assessment.risk_breakdown,
            "critical_risk_factors": assessment.critical_risk_factors,
            "risk_mitigation_priorities": assessment.risk_mitigation_priorities,
            "recommended_actions": assessment.recommended_actions,
            "detailed_analysis": {
                "political_stability": {
                    "composite_score": assessment.political_stability.composite_score,
                    "risk_level": assessment.political_stability.risk_level.value,
                    "action_points": assessment.political_stability.action_points
                },
                "infrastructure_quality": {
                    "composite_score": assessment.infrastructure_quality.composite_score,
                    "risk_level": assessment.infrastructure_quality.risk_level.value,
                    "critical_gaps": assessment.infrastructure_quality.critical_gaps
                },
                "market_volatility": {
                    "composite_score": assessment.market_volatility.composite_score,
                    "risk_level": assessment.market_volatility.risk_level.value,
                    "volatility_drivers": assessment.market_volatility.volatility_drivers
                },
                "operational_risks": {
                    "composite_score": assessment.operational_risks.composite_score,
                    "risk_level": assessment.operational_risks.risk_level.value
                },
                "financial_risks": {
                    "composite_score": assessment.financial_risks.composite_score,
                    "risk_level": assessment.financial_risks.risk_level.value
                }
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Risk assessment results saved to {filepath}")
        return filepath 