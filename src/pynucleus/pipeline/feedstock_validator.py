"""
Feedstock Selection Validation Module

This module provides comprehensive feedstock evaluation and ranking based on:
- Availability consistency (annual data)
- Technical compatibility score with selected plant technology
- Cost efficiency (feedstock cost per energy output)

Output: Ranked list with numerical scores for decision-making
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

from ..utils.logger import logger


class FeedstockType(Enum):
    """Supported feedstock types"""
    NATURAL_GAS = "Natural Gas"
    BIOMASS = "Biomass"
    AGRICULTURAL_WASTE = "Agricultural Waste"
    CRUDE_OIL = "Crude Oil"
    COAL = "Coal"
    JATROPHA_SEEDS = "Jatropha Seeds"
    PALM_OIL = "Palm Oil"
    SUGARCANE = "Sugarcane"
    CORN = "Corn"
    ALGAE = "Algae"
    WASTE_PLASTIC = "Waste Plastic"
    MUNICIPAL_WASTE = "Municipal Waste"


class TechnologyType(Enum):
    """Plant technology types"""
    HABER_BOSCH = "Haber-Bosch Process"
    BIOMASS_CONVERSION = "Biomass Conversion"
    FISCHER_TROPSCH = "Fischer-Tropsch"
    CATALYTIC_CRACKING = "Catalytic Cracking"
    STEAM_REFORMING = "Steam Reforming"
    PYROLYSIS = "Pyrolysis"
    FERMENTATION = "Fermentation"
    TRANSESTERIFICATION = "Transesterification"
    GASIFICATION = "Gasification"


@dataclass
class FeedstockProperties:
    """Properties of a feedstock type"""
    name: str
    feedstock_type: FeedstockType
    energy_content: float  # MJ/kg
    carbon_content: float  # %
    moisture_content: float  # %
    ash_content: float  # %
    sulfur_content: float  # %
    availability_score: float  # 0-1 (regional availability)
    cost_per_ton: float  # USD/ton
    transportation_cost_factor: float  # Multiplier for logistics
    processing_complexity: float  # 0-1 (0=simple, 1=complex)
    environmental_impact: float  # 0-1 (0=low, 1=high)
    supply_chain_reliability: float  # 0-1 (0=unreliable, 1=reliable)
    seasonal_variation: float  # 0-1 (0=constant, 1=highly seasonal)
    storage_requirements: str  # Description of storage needs
    regional_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TechnologyCompatibility:
    """Technology-feedstock compatibility matrix"""
    technology: TechnologyType
    feedstock: FeedstockType
    compatibility_score: float  # 0-1
    conversion_efficiency: float  # 0-1
    pretreatment_required: bool
    catalyst_requirements: List[str]
    operating_conditions: Dict[str, Any]
    yield_factor: float  # Output per unit input
    technical_risk: float  # 0-1 (0=low, 1=high)


@dataclass
class FeedstockEvaluation:
    """Comprehensive feedstock evaluation result"""
    feedstock: FeedstockType
    availability_score: float
    technical_compatibility_score: float
    cost_efficiency_score: float
    composite_score: float
    regional_availability: Dict[str, float]
    technical_details: Dict[str, Any]
    cost_analysis: Dict[str, Any]
    risk_factors: List[str]
    recommendations: List[str]
    evaluation_timestamp: datetime = field(default_factory=datetime.now)


class FeedstockValidator:
    """
    Comprehensive feedstock validation and ranking system
    
    Evaluates feedstock options based on multiple criteria and provides
    ranked recommendations with detailed scoring breakdown.
    """
    
    def __init__(self, data_dir: str = "data/feedstock_validation"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize feedstock database
        self.feedstock_database = self._initialize_feedstock_database()
        self.compatibility_matrix = self._initialize_compatibility_matrix()
        
        # Regional availability data
        self.regional_availability = self._load_regional_availability()
        
        # Cost data and indexes
        self.cost_indexes = self._load_cost_indexes()
        
        logger.info("FeedstockValidator initialized with comprehensive database")
    
    def _initialize_feedstock_database(self) -> Dict[FeedstockType, FeedstockProperties]:
        """Initialize comprehensive feedstock properties database"""
        return {
            FeedstockType.NATURAL_GAS: FeedstockProperties(
                name="Natural Gas",
                feedstock_type=FeedstockType.NATURAL_GAS,
                energy_content=55.5,  # MJ/kg
                carbon_content=75.0,
                moisture_content=0.0,
                ash_content=0.0,
                sulfur_content=0.1,
                availability_score=0.85,
                cost_per_ton=150.0,
                transportation_cost_factor=1.2,
                processing_complexity=0.3,
                environmental_impact=0.6,
                supply_chain_reliability=0.9,
                seasonal_variation=0.1,
                storage_requirements="Pressurized tanks, pipeline infrastructure"
            ),
            
            FeedstockType.BIOMASS: FeedstockProperties(
                name="Biomass",
                feedstock_type=FeedstockType.BIOMASS,
                energy_content=18.0,  # MJ/kg
                carbon_content=45.0,
                moisture_content=25.0,
                ash_content=5.0,
                sulfur_content=0.05,
                availability_score=0.75,
                cost_per_ton=80.0,
                transportation_cost_factor=1.8,
                processing_complexity=0.6,
                environmental_impact=0.3,
                supply_chain_reliability=0.7,
                seasonal_variation=0.4,
                storage_requirements="Covered storage, moisture control"
            ),
            
            FeedstockType.AGRICULTURAL_WASTE: FeedstockProperties(
                name="Agricultural Waste",
                feedstock_type=FeedstockType.AGRICULTURAL_WASTE,
                energy_content=15.0,  # MJ/kg
                carbon_content=40.0,
                moisture_content=30.0,
                ash_content=8.0,
                sulfur_content=0.1,
                availability_score=0.70,
                cost_per_ton=60.0,
                transportation_cost_factor=2.0,
                processing_complexity=0.7,
                environmental_impact=0.2,
                supply_chain_reliability=0.6,
                seasonal_variation=0.6,
                storage_requirements="Open storage, pre-treatment required"
            ),
            
            FeedstockType.CRUDE_OIL: FeedstockProperties(
                name="Crude Oil",
                feedstock_type=FeedstockType.CRUDE_OIL,
                energy_content=42.0,  # MJ/kg
                carbon_content=85.0,
                moisture_content=0.5,
                ash_content=0.1,
                sulfur_content=2.0,
                availability_score=0.90,
                cost_per_ton=350.0,
                transportation_cost_factor=1.1,
                processing_complexity=0.5,
                environmental_impact=0.8,
                supply_chain_reliability=0.95,
                seasonal_variation=0.05,
                storage_requirements="Heated tanks, vapor recovery"
            ),
            
            FeedstockType.JATROPHA_SEEDS: FeedstockProperties(
                name="Jatropha Seeds",
                feedstock_type=FeedstockType.JATROPHA_SEEDS,
                energy_content=19.0,  # MJ/kg
                carbon_content=50.0,
                moisture_content=8.0,
                ash_content=4.0,
                sulfur_content=0.02,
                availability_score=0.65,
                cost_per_ton=120.0,
                transportation_cost_factor=1.5,
                processing_complexity=0.4,
                environmental_impact=0.2,
                supply_chain_reliability=0.65,
                seasonal_variation=0.7,
                storage_requirements="Dry storage, pest control"
            ),
            
            FeedstockType.PALM_OIL: FeedstockProperties(
                name="Palm Oil",
                feedstock_type=FeedstockType.PALM_OIL,
                energy_content=37.0,  # MJ/kg
                carbon_content=77.0,
                moisture_content=0.1,
                ash_content=0.01,
                sulfur_content=0.005,
                availability_score=0.80,
                cost_per_ton=650.0,
                transportation_cost_factor=1.3,
                processing_complexity=0.3,
                environmental_impact=0.7,
                supply_chain_reliability=0.85,
                seasonal_variation=0.3,
                storage_requirements="Temperature controlled, antioxidants"
            ),
            
            FeedstockType.MUNICIPAL_WASTE: FeedstockProperties(
                name="Municipal Waste",
                feedstock_type=FeedstockType.MUNICIPAL_WASTE,
                energy_content=10.0,  # MJ/kg
                carbon_content=30.0,
                moisture_content=40.0,
                ash_content=15.0,
                sulfur_content=0.3,
                availability_score=0.95,
                cost_per_ton=-20.0,  # Negative cost (tipping fees)
                transportation_cost_factor=1.0,
                processing_complexity=0.9,
                environmental_impact=0.4,
                supply_chain_reliability=0.9,
                seasonal_variation=0.1,
                storage_requirements="Waste management infrastructure"
            )
        }
    
    def _initialize_compatibility_matrix(self) -> Dict[Tuple[TechnologyType, FeedstockType], TechnologyCompatibility]:
        """Initialize technology-feedstock compatibility matrix"""
        matrix = {}
        
        # Haber-Bosch Process compatibilities
        matrix[(TechnologyType.HABER_BOSCH, FeedstockType.NATURAL_GAS)] = TechnologyCompatibility(
            technology=TechnologyType.HABER_BOSCH,
            feedstock=FeedstockType.NATURAL_GAS,
            compatibility_score=0.95,
            conversion_efficiency=0.85,
            pretreatment_required=True,
            catalyst_requirements=["Iron-based catalyst", "Promoters"],
            operating_conditions={"temperature": 450, "pressure": 250},
            yield_factor=0.82,
            technical_risk=0.1
        )
        
        matrix[(TechnologyType.HABER_BOSCH, FeedstockType.BIOMASS)] = TechnologyCompatibility(
            technology=TechnologyType.HABER_BOSCH,
            feedstock=FeedstockType.BIOMASS,
            compatibility_score=0.60,
            conversion_efficiency=0.65,
            pretreatment_required=True,
            catalyst_requirements=["Gasification catalyst", "Reforming catalyst"],
            operating_conditions={"temperature": 800, "pressure": 150},
            yield_factor=0.55,
            technical_risk=0.4
        )
        
        matrix[(TechnologyType.HABER_BOSCH, FeedstockType.AGRICULTURAL_WASTE)] = TechnologyCompatibility(
            technology=TechnologyType.HABER_BOSCH,
            feedstock=FeedstockType.AGRICULTURAL_WASTE,
            compatibility_score=0.45,
            conversion_efficiency=0.50,
            pretreatment_required=True,
            catalyst_requirements=["Gasification catalyst", "Cleanup catalyst"],
            operating_conditions={"temperature": 850, "pressure": 100},
            yield_factor=0.40,
            technical_risk=0.6
        )
        
        # Biomass Conversion compatibilities
        matrix[(TechnologyType.BIOMASS_CONVERSION, FeedstockType.BIOMASS)] = TechnologyCompatibility(
            technology=TechnologyType.BIOMASS_CONVERSION,
            feedstock=FeedstockType.BIOMASS,
            compatibility_score=0.90,
            conversion_efficiency=0.80,
            pretreatment_required=True,
            catalyst_requirements=["Enzyme cocktail", "Acid catalyst"],
            operating_conditions={"temperature": 180, "pressure": 10},
            yield_factor=0.75,
            technical_risk=0.2
        )
        
        matrix[(TechnologyType.BIOMASS_CONVERSION, FeedstockType.AGRICULTURAL_WASTE)] = TechnologyCompatibility(
            technology=TechnologyType.BIOMASS_CONVERSION,
            feedstock=FeedstockType.AGRICULTURAL_WASTE,
            compatibility_score=0.85,
            conversion_efficiency=0.75,
            pretreatment_required=True,
            catalyst_requirements=["Enzyme cocktail", "Pretreatment chemicals"],
            operating_conditions={"temperature": 200, "pressure": 15},
            yield_factor=0.70,
            technical_risk=0.3
        )
        
        matrix[(TechnologyType.BIOMASS_CONVERSION, FeedstockType.JATROPHA_SEEDS)] = TechnologyCompatibility(
            technology=TechnologyType.BIOMASS_CONVERSION,
            feedstock=FeedstockType.JATROPHA_SEEDS,
            compatibility_score=0.80,
            conversion_efficiency=0.78,
            pretreatment_required=False,
            catalyst_requirements=["Transesterification catalyst"],
            operating_conditions={"temperature": 60, "pressure": 1},
            yield_factor=0.85,
            technical_risk=0.15
        )
        
        # Fischer-Tropsch compatibilities
        matrix[(TechnologyType.FISCHER_TROPSCH, FeedstockType.NATURAL_GAS)] = TechnologyCompatibility(
            technology=TechnologyType.FISCHER_TROPSCH,
            feedstock=FeedstockType.NATURAL_GAS,
            compatibility_score=0.85,
            conversion_efficiency=0.75,
            pretreatment_required=True,
            catalyst_requirements=["Cobalt catalyst", "Iron catalyst"],
            operating_conditions={"temperature": 250, "pressure": 25},
            yield_factor=0.70,
            technical_risk=0.2
        )
        
        matrix[(TechnologyType.FISCHER_TROPSCH, FeedstockType.BIOMASS)] = TechnologyCompatibility(
            technology=TechnologyType.FISCHER_TROPSCH,
            feedstock=FeedstockType.BIOMASS,
            compatibility_score=0.70,
            conversion_efficiency=0.65,
            pretreatment_required=True,
            catalyst_requirements=["Gasification catalyst", "FT catalyst"],
            operating_conditions={"temperature": 800, "pressure": 20},
            yield_factor=0.60,
            technical_risk=0.35
        )
        
        return matrix
    
    def _load_regional_availability(self) -> Dict[str, Dict[FeedstockType, float]]:
        """Load regional availability data for different feedstocks"""
        return {
            "South Africa": {
                FeedstockType.NATURAL_GAS: 0.60,
                FeedstockType.BIOMASS: 0.80,
                FeedstockType.AGRICULTURAL_WASTE: 0.85,
                FeedstockType.CRUDE_OIL: 0.30,
                FeedstockType.JATROPHA_SEEDS: 0.70,
                FeedstockType.PALM_OIL: 0.20,
                FeedstockType.MUNICIPAL_WASTE: 0.90
            },
            "Nigeria": {
                FeedstockType.NATURAL_GAS: 0.95,
                FeedstockType.BIOMASS: 0.85,
                FeedstockType.AGRICULTURAL_WASTE: 0.90,
                FeedstockType.CRUDE_OIL: 0.95,
                FeedstockType.JATROPHA_SEEDS: 0.60,
                FeedstockType.PALM_OIL: 0.80,
                FeedstockType.MUNICIPAL_WASTE: 0.70
            },
            "Kenya": {
                FeedstockType.NATURAL_GAS: 0.20,
                FeedstockType.BIOMASS: 0.90,
                FeedstockType.AGRICULTURAL_WASTE: 0.95,
                FeedstockType.CRUDE_OIL: 0.10,
                FeedstockType.JATROPHA_SEEDS: 0.85,
                FeedstockType.PALM_OIL: 0.15,
                FeedstockType.MUNICIPAL_WASTE: 0.60
            },
            "Ghana": {
                FeedstockType.NATURAL_GAS: 0.70,
                FeedstockType.BIOMASS: 0.85,
                FeedstockType.AGRICULTURAL_WASTE: 0.90,
                FeedstockType.CRUDE_OIL: 0.60,
                FeedstockType.JATROPHA_SEEDS: 0.75,
                FeedstockType.PALM_OIL: 0.85,
                FeedstockType.MUNICIPAL_WASTE: 0.65
            },
            "Morocco": {
                FeedstockType.NATURAL_GAS: 0.40,
                FeedstockType.BIOMASS: 0.60,
                FeedstockType.AGRICULTURAL_WASTE: 0.75,
                FeedstockType.CRUDE_OIL: 0.05,
                FeedstockType.JATROPHA_SEEDS: 0.30,
                FeedstockType.PALM_OIL: 0.10,
                FeedstockType.MUNICIPAL_WASTE: 0.80
            },
            "Egypt": {
                FeedstockType.NATURAL_GAS: 0.85,
                FeedstockType.BIOMASS: 0.50,
                FeedstockType.AGRICULTURAL_WASTE: 0.80,
                FeedstockType.CRUDE_OIL: 0.60,
                FeedstockType.JATROPHA_SEEDS: 0.20,
                FeedstockType.PALM_OIL: 0.05,
                FeedstockType.MUNICIPAL_WASTE: 0.85
            },
            "Tanzania": {
                FeedstockType.NATURAL_GAS: 0.75,
                FeedstockType.BIOMASS: 0.90,
                FeedstockType.AGRICULTURAL_WASTE: 0.95,
                FeedstockType.CRUDE_OIL: 0.15,
                FeedstockType.JATROPHA_SEEDS: 0.80,
                FeedstockType.PALM_OIL: 0.30,
                FeedstockType.MUNICIPAL_WASTE: 0.50
            },
            "Ethiopia": {
                FeedstockType.NATURAL_GAS: 0.30,
                FeedstockType.BIOMASS: 0.85,
                FeedstockType.AGRICULTURAL_WASTE: 0.90,
                FeedstockType.CRUDE_OIL: 0.05,
                FeedstockType.JATROPHA_SEEDS: 0.70,
                FeedstockType.PALM_OIL: 0.10,
                FeedstockType.MUNICIPAL_WASTE: 0.40
            }
        }
    
    def _load_cost_indexes(self) -> Dict[str, Any]:
        """Load cost indexes for different regions and feedstock types"""
        return {
            "base_year": 2024,
            "regional_cost_multipliers": {
                "South Africa": 1.0,
                "Nigeria": 1.1,
                "Kenya": 1.2,
                "Ghana": 1.15,
                "Morocco": 1.05,
                "Egypt": 1.08,
                "Tanzania": 1.25,
                "Ethiopia": 1.30
            },
            "transportation_costs": {
                "local": 5.0,  # USD/ton/100km
                "regional": 8.0,
                "international": 25.0
            },
            "storage_costs": {
                "simple": 2.0,  # USD/ton/month
                "controlled": 5.0,
                "specialized": 10.0
            },
            "processing_costs": {
                "minimal": 10.0,  # USD/ton
                "moderate": 25.0,
                "complex": 50.0
            }
        }
    
    def evaluate_feedstock_options(self, 
                                 technology_type: str,
                                 plant_location: str,
                                 feedstock_options: List[str],
                                 production_capacity: float,
                                 operating_hours: int = 8000) -> List[FeedstockEvaluation]:
        """
        Evaluate and rank feedstock options based on comprehensive criteria
        
        Args:
            technology_type: Plant technology type
            plant_location: Plant location
            feedstock_options: List of feedstock options to evaluate
            production_capacity: Annual production capacity (tons/year)
            operating_hours: Operating hours per year
            
        Returns:
            List of FeedstockEvaluation objects ranked by composite score
        """
        evaluations = []
        
        # Convert string inputs to enums
        tech_enum = self._get_technology_enum(technology_type)
        if not tech_enum:
            logger.error(f"Unknown technology type: {technology_type}")
            return evaluations
        
        for feedstock_name in feedstock_options:
            feedstock_enum = self._get_feedstock_enum(feedstock_name)
            if not feedstock_enum:
                logger.warning(f"Unknown feedstock type: {feedstock_name}")
                continue
            
            evaluation = self._evaluate_single_feedstock(
                tech_enum, feedstock_enum, plant_location, 
                production_capacity, operating_hours
            )
            evaluations.append(evaluation)
        
        # Sort by composite score (descending)
        evaluations.sort(key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"Evaluated {len(evaluations)} feedstock options for {technology_type}")
        return evaluations
    
    def _evaluate_single_feedstock(self, 
                                 technology: TechnologyType,
                                 feedstock: FeedstockType,
                                 location: str,
                                 capacity: float,
                                 hours: int) -> FeedstockEvaluation:
        """Evaluate a single feedstock option"""
        
        # Get feedstock properties
        props = self.feedstock_database.get(feedstock)
        if not props:
            logger.error(f"No properties found for feedstock: {feedstock}")
            return self._create_empty_evaluation(feedstock)
        
        # Get compatibility data
        compatibility = self.compatibility_matrix.get((technology, feedstock))
        if not compatibility:
            logger.warning(f"No compatibility data for {technology} + {feedstock}")
            compatibility = self._create_default_compatibility(technology, feedstock)
        
        # Calculate availability score
        availability_score = self._calculate_availability_score(feedstock, location, props)
        
        # Calculate technical compatibility score
        technical_score = self._calculate_technical_compatibility_score(compatibility, props)
        
        # Calculate cost efficiency score
        cost_efficiency_score = self._calculate_cost_efficiency_score(
            props, compatibility, location, capacity, hours
        )
        
        # Calculate composite score (weighted average)
        composite_score = (
            availability_score * 0.35 +
            technical_score * 0.40 +
            cost_efficiency_score * 0.25
        )
        
        # Generate regional availability data
        regional_availability = self._get_regional_availability_data(feedstock, location)
        
        # Generate technical details
        technical_details = self._generate_technical_details(compatibility, props)
        
        # Generate cost analysis
        cost_analysis = self._generate_cost_analysis(props, location, capacity, hours)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(props, compatibility, location)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            feedstock, availability_score, technical_score, cost_efficiency_score
        )
        
        return FeedstockEvaluation(
            feedstock=feedstock,
            availability_score=availability_score,
            technical_compatibility_score=technical_score,
            cost_efficiency_score=cost_efficiency_score,
            composite_score=composite_score,
            regional_availability=regional_availability,
            technical_details=technical_details,
            cost_analysis=cost_analysis,
            risk_factors=risk_factors,
            recommendations=recommendations
        )
    
    def _calculate_availability_score(self, 
                                    feedstock: FeedstockType,
                                    location: str,
                                    props: FeedstockProperties) -> float:
        """Calculate availability consistency score"""
        # Base availability from properties
        base_availability = props.availability_score
        
        # Regional availability adjustment
        regional_data = self.regional_availability.get(location, {})
        regional_factor = regional_data.get(feedstock, 0.5)  # Default to medium if unknown
        
        # Supply chain reliability factor
        reliability_factor = props.supply_chain_reliability
        
        # Seasonal variation penalty (lower is better)
        seasonal_penalty = 1.0 - (props.seasonal_variation * 0.3)
        
        # Combine factors
        availability_score = (
            base_availability * 0.4 +
            regional_factor * 0.3 +
            reliability_factor * 0.2 +
            seasonal_penalty * 0.1
        )
        
        return min(1.0, max(0.0, availability_score))
    
    def _calculate_technical_compatibility_score(self, 
                                              compatibility: TechnologyCompatibility,
                                              props: FeedstockProperties) -> float:
        """Calculate technical compatibility score"""
        # Base compatibility
        base_score = compatibility.compatibility_score
        
        # Conversion efficiency bonus
        efficiency_bonus = compatibility.conversion_efficiency * 0.3
        
        # Processing complexity penalty
        complexity_penalty = props.processing_complexity * 0.2
        
        # Technical risk penalty
        risk_penalty = compatibility.technical_risk * 0.25
        
        # Pretreatment penalty
        pretreatment_penalty = 0.1 if compatibility.pretreatment_required else 0.0
        
        # Combine factors
        technical_score = (
            base_score + efficiency_bonus - complexity_penalty - 
            risk_penalty - pretreatment_penalty
        )
        
        return min(1.0, max(0.0, technical_score))
    
    def _calculate_cost_efficiency_score(self, 
                                       props: FeedstockProperties,
                                       compatibility: TechnologyCompatibility,
                                       location: str,
                                       capacity: float,
                                       hours: int) -> float:
        """Calculate cost efficiency score (feedstock cost per energy output)"""
        # Base feedstock cost
        base_cost = props.cost_per_ton
        
        # Regional cost adjustment
        regional_multiplier = self.cost_indexes["regional_cost_multipliers"].get(location, 1.0)
        adjusted_cost = base_cost * regional_multiplier
        
        # Transportation cost
        transport_cost = adjusted_cost * (props.transportation_cost_factor - 1.0)
        
        # Processing cost based on complexity
        processing_cost = self._get_processing_cost(props.processing_complexity)
        
        # Total cost per ton
        total_cost_per_ton = adjusted_cost + transport_cost + processing_cost
        
        # Energy output per ton (considering conversion efficiency)
        energy_output = props.energy_content * compatibility.conversion_efficiency
        
        # Cost per energy unit
        cost_per_energy = total_cost_per_ton / energy_output if energy_output > 0 else float('inf')
        
        # Normalize to 0-1 score (lower cost is better)
        # Use benchmark of $5/MJ as reference point
        benchmark_cost = 5.0
        if cost_per_energy <= 0:
            return 1.0  # Negative cost (waste materials)
        
        efficiency_score = benchmark_cost / cost_per_energy
        return min(1.0, max(0.0, efficiency_score))
    
    def _get_processing_cost(self, complexity: float) -> float:
        """Get processing cost based on complexity level"""
        if complexity < 0.3:
            return self.cost_indexes["processing_costs"]["minimal"]
        elif complexity < 0.7:
            return self.cost_indexes["processing_costs"]["moderate"]
        else:
            return self.cost_indexes["processing_costs"]["complex"]
    
    def _get_regional_availability_data(self, 
                                      feedstock: FeedstockType,
                                      location: str) -> Dict[str, float]:
        """Get regional availability data for the feedstock"""
        regional_data = {}
        for region, feedstock_data in self.regional_availability.items():
            regional_data[region] = feedstock_data.get(feedstock, 0.0)
        return regional_data
    
    def _generate_technical_details(self, 
                                  compatibility: TechnologyCompatibility,
                                  props: FeedstockProperties) -> Dict[str, Any]:
        """Generate technical details for the evaluation"""
        return {
            "compatibility_score": compatibility.compatibility_score,
            "conversion_efficiency": compatibility.conversion_efficiency,
            "yield_factor": compatibility.yield_factor,
            "pretreatment_required": compatibility.pretreatment_required,
            "catalyst_requirements": compatibility.catalyst_requirements,
            "operating_conditions": compatibility.operating_conditions,
            "technical_risk": compatibility.technical_risk,
            "energy_content": props.energy_content,
            "processing_complexity": props.processing_complexity,
            "storage_requirements": props.storage_requirements
        }
    
    def _generate_cost_analysis(self, 
                              props: FeedstockProperties,
                              location: str,
                              capacity: float,
                              hours: int) -> Dict[str, Any]:
        """Generate detailed cost analysis"""
        regional_multiplier = self.cost_indexes["regional_cost_multipliers"].get(location, 1.0)
        base_cost = props.cost_per_ton
        adjusted_cost = base_cost * regional_multiplier
        
        annual_feedstock_requirement = capacity / props.energy_content  # Rough estimate
        
        return {
            "base_cost_per_ton": base_cost,
            "regional_cost_multiplier": regional_multiplier,
            "adjusted_cost_per_ton": adjusted_cost,
            "transportation_cost_factor": props.transportation_cost_factor,
            "estimated_annual_requirement": annual_feedstock_requirement,
            "estimated_annual_cost": adjusted_cost * annual_feedstock_requirement,
            "cost_per_energy_unit": adjusted_cost / props.energy_content,
            "environmental_cost_factor": props.environmental_impact
        }
    
    def _identify_risk_factors(self, 
                             props: FeedstockProperties,
                             compatibility: TechnologyCompatibility,
                             location: str) -> List[str]:
        """Identify risk factors for the feedstock"""
        risks = []
        
        if props.supply_chain_reliability < 0.7:
            risks.append("Low supply chain reliability")
        
        if props.seasonal_variation > 0.5:
            risks.append("High seasonal availability variation")
        
        if compatibility.technical_risk > 0.4:
            risks.append("High technical implementation risk")
        
        if props.processing_complexity > 0.6:
            risks.append("Complex processing requirements")
        
        if props.environmental_impact > 0.6:
            risks.append("High environmental impact")
        
        if compatibility.pretreatment_required:
            risks.append("Pretreatment required increases complexity")
        
        regional_availability = self.regional_availability.get(location, {})
        if regional_availability.get(props.feedstock_type, 0) < 0.5:
            risks.append("Limited regional availability")
        
        return risks
    
    def _generate_recommendations(self, 
                                feedstock: FeedstockType,
                                availability_score: float,
                                technical_score: float,
                                cost_score: float) -> List[str]:
        """Generate recommendations based on scores"""
        recommendations = []
        
        if availability_score < 0.6:
            recommendations.append("Consider diversifying supply sources")
            recommendations.append("Implement inventory management system")
        
        if technical_score < 0.6:
            recommendations.append("Conduct pilot-scale testing")
            recommendations.append("Evaluate alternative processing routes")
        
        if cost_score < 0.6:
            recommendations.append("Negotiate long-term supply contracts")
            recommendations.append("Explore local sourcing options")
        
        # Overall recommendations
        overall_score = (availability_score + technical_score + cost_score) / 3
        if overall_score > 0.8:
            recommendations.append("Highly recommended feedstock option")
        elif overall_score > 0.6:
            recommendations.append("Good feedstock option with some considerations")
        else:
            recommendations.append("Consider alternative feedstock options")
        
        return recommendations
    
    def _get_technology_enum(self, technology_str: str) -> Optional[TechnologyType]:
        """Convert technology string to enum"""
        technology_map = {
            "Haber-Bosch Process": TechnologyType.HABER_BOSCH,
            "Biomass Conversion": TechnologyType.BIOMASS_CONVERSION,
            "Fischer-Tropsch": TechnologyType.FISCHER_TROPSCH,
            "Catalytic Cracking": TechnologyType.CATALYTIC_CRACKING,
            "Steam Reforming": TechnologyType.STEAM_REFORMING,
            "Pyrolysis": TechnologyType.PYROLYSIS,
            "Fermentation": TechnologyType.FERMENTATION,
            "Transesterification": TechnologyType.TRANSESTERIFICATION,
            "Gasification": TechnologyType.GASIFICATION
        }
        return technology_map.get(technology_str)
    
    def _get_feedstock_enum(self, feedstock_str: str) -> Optional[FeedstockType]:
        """Convert feedstock string to enum"""
        feedstock_map = {
            "Natural Gas": FeedstockType.NATURAL_GAS,
            "Biomass": FeedstockType.BIOMASS,
            "Agricultural Waste": FeedstockType.AGRICULTURAL_WASTE,
            "Crude Oil": FeedstockType.CRUDE_OIL,
            "Coal": FeedstockType.COAL,
            "Jatropha Seeds": FeedstockType.JATROPHA_SEEDS,
            "Palm Oil": FeedstockType.PALM_OIL,
            "Sugarcane": FeedstockType.SUGARCANE,
            "Corn": FeedstockType.CORN,
            "Algae": FeedstockType.ALGAE,
            "Waste Plastic": FeedstockType.WASTE_PLASTIC,
            "Municipal Waste": FeedstockType.MUNICIPAL_WASTE
        }
        return feedstock_map.get(feedstock_str)
    
    def _create_empty_evaluation(self, feedstock: FeedstockType) -> FeedstockEvaluation:
        """Create empty evaluation for unknown feedstock"""
        return FeedstockEvaluation(
            feedstock=feedstock,
            availability_score=0.0,
            technical_compatibility_score=0.0,
            cost_efficiency_score=0.0,
            composite_score=0.0,
            regional_availability={},
            technical_details={},
            cost_analysis={},
            risk_factors=["Unknown feedstock type"],
            recommendations=["Feedstock evaluation not available"]
        )
    
    def _create_default_compatibility(self, 
                                    technology: TechnologyType,
                                    feedstock: FeedstockType) -> TechnologyCompatibility:
        """Create default compatibility for unknown combinations"""
        return TechnologyCompatibility(
            technology=technology,
            feedstock=feedstock,
            compatibility_score=0.5,
            conversion_efficiency=0.5,
            pretreatment_required=True,
            catalyst_requirements=["Unknown"],
            operating_conditions={"temperature": 0, "pressure": 0},
            yield_factor=0.5,
            technical_risk=0.7
        )
    
    def generate_feedstock_report(self, 
                                evaluations: List[FeedstockEvaluation],
                                technology_type: str,
                                location: str) -> str:
        """Generate comprehensive feedstock evaluation report"""
        report = []
        report.append("="*80)
        report.append("FEEDSTOCK SELECTION VALIDATION REPORT")
        report.append("="*80)
        report.append(f"Technology: {technology_type}")
        report.append(f"Location: {location}")
        report.append(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("RANKING SUMMARY")
        report.append("-" * 40)
        for i, eval_result in enumerate(evaluations, 1):
            report.append(f"{i}. {eval_result.feedstock.value}")
            report.append(f"   Composite Score: {eval_result.composite_score:.3f}")
            report.append(f"   Availability: {eval_result.availability_score:.3f}")
            report.append(f"   Technical: {eval_result.technical_compatibility_score:.3f}")
            report.append(f"   Cost Efficiency: {eval_result.cost_efficiency_score:.3f}")
            report.append("")
        
        report.append("\nDETAILED ANALYSIS")
        report.append("-" * 40)
        
        for eval_result in evaluations:
            report.append(f"\n{eval_result.feedstock.value.upper()}")
            report.append("-" * len(eval_result.feedstock.value))
            
            # Scores
            report.append(f"Composite Score: {eval_result.composite_score:.3f}")
            report.append(f"• Availability Score: {eval_result.availability_score:.3f}")
            report.append(f"• Technical Compatibility: {eval_result.technical_compatibility_score:.3f}")
            report.append(f"• Cost Efficiency: {eval_result.cost_efficiency_score:.3f}")
            
            # Technical details
            report.append(f"\nTechnical Details:")
            technical = eval_result.technical_details
            report.append(f"• Conversion Efficiency: {technical.get('conversion_efficiency', 0):.2f}")
            report.append(f"• Yield Factor: {technical.get('yield_factor', 0):.2f}")
            report.append(f"• Pretreatment Required: {technical.get('pretreatment_required', False)}")
            report.append(f"• Technical Risk: {technical.get('technical_risk', 0):.2f}")
            
            # Cost analysis
            report.append(f"\nCost Analysis:")
            cost = eval_result.cost_analysis
            report.append(f"• Base Cost: ${cost.get('base_cost_per_ton', 0):.2f}/ton")
            report.append(f"• Regional Multiplier: {cost.get('regional_cost_multiplier', 1):.2f}")
            report.append(f"• Adjusted Cost: ${cost.get('adjusted_cost_per_ton', 0):.2f}/ton")
            report.append(f"• Estimated Annual Cost: ${cost.get('estimated_annual_cost', 0):,.0f}")
            
            # Risk factors
            if eval_result.risk_factors:
                report.append(f"\nRisk Factors:")
                for risk in eval_result.risk_factors:
                    report.append(f"• {risk}")
            
            # Recommendations
            if eval_result.recommendations:
                report.append(f"\nRecommendations:")
                for rec in eval_result.recommendations:
                    report.append(f"• {rec}")
            
            report.append("")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, 
                               evaluations: List[FeedstockEvaluation],
                               technology_type: str,
                               location: str,
                               filename: Optional[str] = None) -> Path:
        """Save evaluation results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedstock_evaluation_{timestamp}.json"
        
        filepath = self.data_dir / filename
        
        # Convert to serializable format
        data = {
            "metadata": {
                "technology_type": technology_type,
                "location": location,
                "evaluation_date": datetime.now().isoformat(),
                "total_options": len(evaluations)
            },
            "evaluations": []
        }
        
        for eval_result in evaluations:
            eval_data = {
                "feedstock": eval_result.feedstock.value,
                "scores": {
                    "composite": eval_result.composite_score,
                    "availability": eval_result.availability_score,
                    "technical_compatibility": eval_result.technical_compatibility_score,
                    "cost_efficiency": eval_result.cost_efficiency_score
                },
                "regional_availability": eval_result.regional_availability,
                "technical_details": eval_result.technical_details,
                "cost_analysis": eval_result.cost_analysis,
                "risk_factors": eval_result.risk_factors,
                "recommendations": eval_result.recommendations
            }
            data["evaluations"].append(eval_data)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Feedstock evaluation results saved to {filepath}")
        return filepath 