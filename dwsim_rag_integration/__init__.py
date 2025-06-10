"""
PyNucleus-Model: Enhanced DWSIM Bridge with RAG Integration

This module provides a comprehensive chemical engineering simulation platform
that integrates DWSIM simulation capabilities with RAG-based intelligent
process design for modular plants.

Main Components:
- Enhanced DWSIM Bridge: Core simulation interface with chemical engineering calculations
- Enhanced DWSIM Service: Microservice for simulation execution with REST API
- RAG Integration: Intelligent process design queries and recommendations
- Modular Plant Design: Specialized tools for modular chemical plant optimization

Version: 2.0
"""

from .core.enhanced_dwsim_bridge import (
    DWSimBridge,
    ProcessConditions,
    EconomicParameters, 
    OptimizationResult,
    SimulationResult,
    create_default_feed_conditions,
    create_default_economic_params
)

# Aliases for clarity
EnhancedDWSimBridge = DWSimBridge

__version__ = "2.0.0"
__author__ = "PyNucleus-Model Enhanced DWSIM Integration"

__all__ = [
    "DWSimBridge",
    "EnhancedDWSimBridge",
    "ProcessConditions", 
    "EconomicParameters",
    "OptimizationResult",
    "SimulationResult",
    "create_default_feed_conditions",
    "create_default_economic_params"
] 